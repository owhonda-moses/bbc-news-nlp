import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    DistilBertForTokenClassification, RobertaForTokenClassification,
    DistilBertTokenizerFast, RobertaTokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import random
import json
import warnings
import logging
from src.ner.utils import NERDataset, evaluate_model

# suppress warnings
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# model config
MODEL = 'roberta'
# MODEL = 'distilbert'

CONFIGS = {
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'tokenizer': DistilBertTokenizerFast, 'model': DistilBertForTokenClassification,
        'batch_size': 4, 'learning_rate': 3e-5
    },
    'roberta': {
        'model_name': 'roberta-base',
        'tokenizer': RobertaTokenizerFast, 'model': RobertaForTokenClassification,
        'batch_size': 4, 'learning_rate': 1e-5
    }
}
config = CONFIGS[MODEL]

# paths and hyperparameters
VERSION = 'v2'
DATA_PATH = '././data/output/ner'
MODELS_PATH = '././models'
PARAMS_PATH = '././params'
NER_TRAIN = f"{DATA_PATH}/train_{MODEL}_{VERSION}.csv"
NER_VAL = f"{DATA_PATH}/val_{MODEL}_{VERSION}.csv"
OUTPUT_DIR = f"{MODELS_PATH}/ner-{MODEL}-model-{VERSION}"
SCORE_FILE = f"{PARAMS_PATH}/ner_{MODEL}_score_{VERSION}.json"

MAX_LEN = 512
BATCH_SIZE = config['batch_size']
EPOCHS = 10
LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VAL = 1.0
WARM_UP = 0.1
PATIENCE = 4
SET_SEED = 0 #0:use random seed, 1:use best seed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    print(f"Selected model: {MODEL}")

    previous_best_f1 = 0.0
    best_seed = None
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, 'r') as f:
            try:
                data = json.load(f)
                previous_best_f1 = data['best_validation_f1']
                best_seed = data['seed']
                print(f"Best F1 is {previous_best_f1:.4f} with seed: {best_seed}")
            except (json.JSONDecodeError, KeyError):
                print("Could not read score file.")

    current_seed = best_seed if SET_SEED == 1 and best_seed is not None else random.randint(1, 1_000_000)
    print(f"Using seed: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)

    train_df = pd.read_csv(NER_TRAIN).dropna()
    val_df = pd.read_csv(NER_VAL).dropna()
    
    print(f"Training sentences: {len(train_df)}")
    print(f"Validation sentences: {len(val_df)}")

    all_tags = set(tag for tags in pd.concat([train_df['ner_tags'], val_df['ner_tags']]).values for tag in tags.split())
    tag2id = {tag: id for id, tag in enumerate(sorted(list(all_tags)))}
    id2tag = {id: tag for tag, id in tag2id.items()}

    tokenizer_class, model_class = config['tokenizer'], config['model']
    
    tokenizer_args = {'add_prefix_space': True} if MODEL == 'roberta' else {}
    tokenizer = tokenizer_class.from_pretrained(config['model_name'], **tokenizer_args)
        
    model = model_class.from_pretrained(config['model_name'], num_labels=len(tag2id), id2label=id2tag, label2id=tag2id)
    model.to(device)

    train_dataset = NERDataset(train_df, tokenizer, tag2id, MAX_LEN)
    val_dataset = NERDataset(val_df, tokenizer, tag2id, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARM_UP)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    print("Starting training")
    best_f1, best_epoch, best_state = 0.0, -1, None
    patience_counter, best_f1_for_stopping = 0, 0.0

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to(device),
                              attention_mask=batch['attention_mask'].to(device),
                              labels=batch['labels'].to(device))
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
            optimizer.step()
            scheduler.step()
        
        metrics, _, _ = evaluate_model(model, val_loader, device, id2tag)
        f1 = metrics['macro avg']['f1-score']
        print(f"  Val F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch + 1, model.state_dict().copy()
            print(f"  Best F1 so far is {best_f1:.4f} at epoch {best_epoch}")

        if f1 > best_f1_for_stopping:
            best_f1_for_stopping = f1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"\nBest validation F1 was {best_f1:.4f}.")
    if best_f1 >= previous_best_f1:
        print(f"F1 ({best_f1:.4f}) is an improvement over previous best ({previous_best_f1:.4f}).")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.load_state_dict(best_state)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        score = {'model': MODEL, 'best_validation_f1': best_f1, 'seed': current_seed, 
                 'epoch': best_epoch, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
        with open(SCORE_FILE, 'w') as f:
            json.dump(score, f, indent=4)
        print(f"Model saved to '{OUTPUT_DIR}'")
    else:
        print(f"Existing model with F1 {previous_best_f1:.4f} remains the best.")

if __name__ == "__main__":
    main()