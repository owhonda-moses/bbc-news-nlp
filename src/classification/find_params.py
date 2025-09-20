import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import optuna
from optuna.trial import TrialState
import random
import warnings
import logging
from tqdm import tqdm

# suppress warnings and verbose logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.classification.utils import NewsClassifierDataset, eval_model

DATA_PATH = '././data/output/sub_class'
OUTPUT_DIR= '././params'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 512
EPOCHS = 10
N_TRIALS = 50


# custom callback
class TqdmCallback:
    def __init__(self, n_trials):
        self.tqdm_bar = tqdm(total=n_trials, desc="Hyperparameter Search")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        self.tqdm_bar.update(1)
        params_str = ", ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.params.items())
        self.tqdm_bar.set_description(
            f"Best F1: {study.best_value:.4f} | Last Trial: {params_str}"
        )
        if self.tqdm_bar.n == self.tqdm_bar.total:
            self.tqdm_bar.close()

def objective(trial, train_df, val_dataset, unique_labels, device):

    trial_seed = random.randint(1, 1_000_000_000)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    trial.set_user_attr("seed", trial_seed)

    batch_size = 4
    lrs = np.logspace(np.log10(1e-5), np.log10(5e-4), 20)
    learning_rate = trial.suggest_categorical("learning_rate", lrs.tolist())
    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1, step=0.005)

    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=train_df['target_label']
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fct(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        _, metrics = eval_model(model, val_loader, device)
        current_f1 = metrics['f1']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
        
        trial.report(current_f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    del model
    torch.cuda.empty_cache()

    return best_val_f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_zeroshot.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_zeroshot.csv'))

    unique_labels = sorted(train_df['target_label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    tqdm_callback = TqdmCallback(N_TRIALS)

    print(f"Starting bayesian optimization with {N_TRIALS} trials")

    study.optimize(
        lambda trial: objective(trial, train_df, val_dataset, unique_labels, device),
        n_trials=N_TRIALS,
        callbacks=[tqdm_callback]
    )

    print("\nOptimization complete.")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best parameters:")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        if key == 'learning_rate':
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    print(f"  seed: {best_trial.user_attrs['seed']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_params = best_trial.params
    best_params['f1_score'] = best_trial.value
    best_params['seed'] = best_trial.user_attrs['seed']

    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    all_trials = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trial_data = trial.params.copy()
            trial_data['f1_score'] = trial.value
            trial_data['seed'] = trial.user_attrs.get('seed')
            all_trials.append(trial_data)

    with open(os.path.join(OUTPUT_DIR, 'all_trials.json'), 'w') as f:
        json.dump(all_trials, f, indent=2)

    return study.best_params

if __name__ == "__main__":
    best_params = main()