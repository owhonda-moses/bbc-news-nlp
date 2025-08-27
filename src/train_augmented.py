import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

from src.train_baseline import NewsClassifierDataset, compute_metrics, eval_model
from src.train_weighted import train_epoch_weighted


DATA_PATH = './data'
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_MODEL_DIR = './models/augmented-classifier'

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 10

# default values
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

PARAMS_PATH = './params/best_params.json' # tuned parameters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'zeroshot_balanced.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_zeroshot.csv'))

    unique_labels_list = sorted(train_df['target_label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels_list),
        y=train_df['target_label']
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_labels_list), id2label=id2label, label2id=label2id
    )
    model.to(device)

    # load hyperparameters
    try:
        with open(PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        learning_rate = best_params['learning_rate']
        weight_decay = best_params['weight_decay']
        print(f"Loaded tuned parameters: LR={learning_rate:.2e}, WD={weight_decay:.2e}")
    except FileNotFoundError:
        print("best_params.json not found. Using default values.")
        learning_rate = LEARNING_RATE
        weight_decay = WEIGHT_DECAY

    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    print("Starting training.")

    best_f1_score = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss = train_epoch_weighted(model, train_loader, optimizer, device, loss_fct)
        val_loss, metrics = eval_model(model, val_loader, device)

        current_f1 = metrics['f1']
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            patience_counter = 0
            os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_MODEL_DIR)
            tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\nTraining complete.")
    print(f"Best model saved to '{OUTPUT_MODEL_DIR}' with F1-score: {best_f1_score:.4f}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()