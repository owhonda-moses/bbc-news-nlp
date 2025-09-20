import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random

from src.classification.utils import NewsClassifierDataset, eval_model, train_epoch_weighted


DATA_PATH = '././data/output/sub_class'
OUTPUT_MODEL_DIR = '././models/weighted-classifier'
MODEL_NAME = 'distilbert-base-uncased'

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 20

# LEARNING_RATE = 2.25e-05
# WEIGHT_DECAY = 0.01
# SEED = None

LEARNING_RATE = 1.2286253559934765e-05
WEIGHT_DECAY = 0.075
SEED = 367688847


def train_epoch_weighted(model, data_loader, optimizer, device, loss_function):
    """Training loop that uses a custom weighted loss function."""
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    print(f"  Training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if SEED is not None:
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
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=train_df['target_label']
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    model.to(device)

    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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