import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
import numpy as np
import random

from src.classification.utils import NewsClassifierDataset, eval_model


DATA_PATH = '././data/output/sub_class'
MODELS_PATH = '../../models'
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_MODEL_DIR = os.path.join(MODELS_PATH, 'baseline-classifier')

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 14
LEARNING_RATE = 2e-5
SEED = 42

def train_epoch(model, data_loader, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    print(f"  Training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_zeroshot.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_zeroshot.csv'))

    unique_labels = sorted(train_df['target_label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(unique_labels)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training baseline")
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_epoch(model, train_loader, optimizer, device)
        _, metrics = eval_model(model, val_loader, device)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_MODEL_DIR)
            tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
            print(f"  Best model saved with F1: {best_f1:.4f}")
    
    print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()