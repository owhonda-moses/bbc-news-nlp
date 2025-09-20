import os
import json
import random
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_
from transformers import (
    DistilBertForTokenClassification, RobertaForTokenClassification,
    DistilBertTokenizerFast, RobertaTokenizerFast,
    get_linear_schedule_with_warmup, DataCollatorForTokenClassification
)
from torch.optim import AdamW
from tqdm import tqdm
from src.ner.utils import NERDataset, evaluate_model, to_py

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL = 'roberta'
# MODEL = 'distilbert'

VERSION = 'v2'
DATA_PATH = '././data/output/ner'
MODELS_PATH = '././models'
PARAMS_PATH = '././params'
NER_TRAIN = f"{DATA_PATH}/train_{MODEL}_{VERSION}.csv"
NER_VAL = f"{DATA_PATH}/val_{MODEL}.csv"
OUTPUT_DIR = f"{MODELS_PATH}/{MODEL}-model-{VERSION}"
SCORE_FILE = f"{PARAMS_PATH}/{MODEL}_score_{VERSION}.json"
TAGS_FILE = f"{PARAMS_PATH}/tags.json"
THR_FILE = f"{PARAMS_PATH}/{MODEL}_thresholds_{VERSION}.json"

CONFIGS = {
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'tokenizer': DistilBertTokenizerFast, 'model': DistilBertForTokenClassification,
        'batch_size': 4, 'learning_rate': 3e-5
    },
    'roberta': {
        'model_name': 'roberta-base',
        'tokenizer': RobertaTokenizerFast, 'model': RobertaForTokenClassification,
        'batch_size': 4, 'learning_rate': 2e-5
    }
}
config = CONFIGS[MODEL]

MAX_LEN = 512
BATCH_SIZE = config['batch_size']
MIN_EPOCHS = 10
MAX_EPOCHS = 30
LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VAL = 1.0
WARM_UP = 0.06
PATIENCE = 6
MIN_DELTA = 1e-3
ACCUM_STEPS = 2
SET_SEED = 0  # 1:use best seed, 0:new random seed
PAD_TO = 8

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_or_build_tags(train_df, val_df):
    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE, 'r') as f:
            tag2id = json.load(f)
        id2tag = {int(v): k for k, v in tag2id.items()}
        return tag2id, id2tag
    all_tags = set(tag for tags in (train_df['ner_tags'].tolist() + val_df['ner_tags'].tolist()) for tag in str(tags).split())
    tag_list = sorted(list(all_tags))
    tag2id = {tag: i for i, tag in enumerate(tag_list)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    os.makedirs(PARAMS_PATH, exist_ok=True)
    with open(TAGS_FILE, 'w') as f:
        json.dump(tag2id, f, indent=2)
    return tag2id, id2tag

def compute_class_weights(train_df, tag2id, eps=1e-6):
    counts = np.zeros(len(tag2id), dtype=np.float64)
    for tags in train_df['ner_tags'].tolist():
        for t in str(tags).split():
            idx = tag2id.get(t)
            if idx is not None and idx >= 0:
                counts[idx] += 1
    inv = 1.0 / np.maximum(counts, eps)
    inv /= inv.mean()
    if 'O' in tag2id:
        inv[tag2id['O']] = min(inv[tag2id['O']], 0.25)
    return torch.tensor(inv, dtype=torch.float32)

def token_balanced_sentence_weights(train_df):
    cls_counts = {}
    for tags in train_df['ner_tags']:
        for t in str(tags).split():
            if t == 'O': continue
            cls = t.split('-', 1)[-1]
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
    inv = {c: 1.0 / v for c, v in cls_counts.items()} if cls_counts else {}
    weights = []
    for tags in train_df['ner_tags']:
        w = 1.0
        for t in str(tags).split():
            if t == 'O': continue
            cls = t.split('-', 1)[-1]
            w += inv.get(cls, 0.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.double)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    print(f"Selected model: {MODEL}")

    previous_best_f1 = 0.0
    best_seed = None
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, 'r') as f:
                data = json.load(f)
            previous_best_f1 = data.get('best_validation_f1', 0.0)
            best_seed = data.get('seed')
            print(f"Best F1 is {previous_best_f1:.4f} with seed: {best_seed}")
        except:
            print("Could not read score file.")

    if SET_SEED == 1 and best_seed is not None:
        current_seed = best_seed
    else:
        current_seed = random.randint(1, 1_000_000)
    print(f"Using seed: {current_seed}")
    set_deterministic(current_seed)

    train_df = pd.read_csv(NER_TRAIN).dropna()
    val_df = pd.read_csv(NER_VAL).dropna()
    print(f"Training sentences: {len(train_df)}")
    print(f"Validation sentences: {len(val_df)}")

    tag2id, id2tag = load_or_build_tags(train_df, val_df)
    tokenizer_args = {'add_prefix_space': True} if MODEL == 'roberta' else {}
    tokenizer = CONFIGS[MODEL]['tokenizer'].from_pretrained(CONFIGS[MODEL]['model_name'], **tokenizer_args)

    model = CONFIGS[MODEL]['model'].from_pretrained(
        CONFIGS[MODEL]['model_name'],
        num_labels=len(tag2id),
        id2label={i: k for k, i in tag2id.items()},
        label2id=tag2id
    )
    model.to(device)

    train_dataset = NERDataset(train_df, tokenizer, tag2id, MAX_LEN)
    val_dataset = NERDataset(val_df, tokenizer, tag2id, MAX_LEN)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=PAD_TO)
    class_weights = compute_class_weights(train_df, tag2id).to(device)

    sent_weights = token_balanced_sentence_weights(train_df)
    sampler = WeightedRandomSampler(weights=sent_weights, num_samples=len(sent_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight', 'LayerNorm.bias'])], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight', 'LayerNorm.bias'])], 'weight_decay': 0.0},
    ], lr=LEARNING_RATE)

    total_steps = (len(train_loader) * MAX_EPOCHS) // max(1, ACCUM_STEPS)
    warmup_steps = int(total_steps * WARM_UP)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("Starting training")
    best_f1, best_epoch, best_state = 0.0, -1, None
    patience_counter, best_f1_for_stopping = 0, 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    epoch = 0
    while True:
        epoch += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = 0

        # training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    weight=class_weights,
                    ignore_index=-100
                ) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1

        # validation
        metrics, _, _ = evaluate_model(model, val_loader, device, {i: k for k, i in tag2id.items()}, thresholds=None)
        f1 = metrics.get('macro avg', {}).get('f1-score', 0.0)
        print(f"  Val F1: {f1:.4f}")

        # track best model
        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best at epoch {best_epoch}: {best_f1:.4f}")

        if f1 > best_f1_for_stopping + MIN_DELTA:
            best_f1_for_stopping = f1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{PATIENCE}")

        if (epoch >= MIN_EPOCHS and patience_counter >= PATIENCE) or epoch >= MAX_EPOCHS:
            print("Stopping criteria met.")
            break

    # save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    os.makedirs(PARAMS_PATH, exist_ok=True)
    score = {
        'model': MODEL,
        'best_validation_f1': best_f1,
        'seed': current_seed,
        'epoch': best_epoch,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'accum_steps': ACCUM_STEPS,
        'sampler': 'token_balanced'
    }
    with open(SCORE_FILE, 'w') as f:
        json.dump(to_py(score), f, indent=2)
    print(f"Model saved to '{OUTPUT_DIR}'. Best Val F1: {best_f1:.4f}")

    # per-class threshold tuning
    id2tag = {i: k for k, i in tag2id.items()}
    classes = sorted({t.split('-', 1)[-1] for t in id2tag.values() if t != 'O'})
    grid = [0.5, 0.6, 0.7, 0.8]
    thr = {c: 0.6 for c in classes}
    for c in classes:
        best_tc, best_macro = thr[c], -1.0
        for t in grid:
            tmp = dict(thr); tmp[c] = t
            m, _, _ = evaluate_model(model, val_loader, device, id2tag, thresholds=tmp)
            macro = m.get('macro avg', {}).get('f1-score', 0.0)
            if macro > best_macro:
                best_macro, best_tc = macro, t
        thr[c] = best_tc
    with open(THR_FILE, 'w') as f:
        json.dump(to_py(thr), f, indent=2)
    print(f"Tuned per-class thresholds saved to '{THR_FILE}'")

    
if __name__ == "__main__":
    main()