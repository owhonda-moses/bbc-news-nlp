import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.train_baseline import NewsClassifierDataset, compute_metrics, eval_model


DATA_PATH = './data'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 2e-5
OUTPUT_MODEL_DIR = './models/weighted-classifier'

# torch.cuda.empty_cache()

def train_epoch_weighted(model, data_loader, optimizer, device, loss_function):
    """Modified training loop that uses a custom weighted loss function."""
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # get logits from the model, don't use its internal loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)  # calculate loss with our weighted function
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    print(f"  Training loss: {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_final.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_final.csv'))

    # create label mapping
    unique_labels = sorted(train_df['target_label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # calculate weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=train_df['target_label']
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # initialize tokenizer, model, and dataLoaders
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    model.to(device)

    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # initialize optimizer and the weighted loss function
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    
    print("Starting training.")
    # training loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_epoch_weighted(model, train_loader, optimizer, device, loss_fct)
        eval_model(model, val_loader, device)
    
    print("\nTraining complete.")

    # Save model
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Model saved to '{OUTPUT_MODEL_DIR}'")

    # clear GPU cache
    if device.type == 'cuda':
        print("Clearing GPU cache.")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()