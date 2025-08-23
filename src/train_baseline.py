import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


DATA_PATH = './data'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_MODEL_DIR = './models/baseline-classifier'

# custom dataset class
class NewsClassifierDataset(Dataset):
    """Custom PyTorch Dataset for text classification."""
    def __init__(self, dataframe, tokenizer, label_map, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe.text.tolist()
        self.labels = [label_map[label] for label in dataframe.target_label]
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# helper functions
def compute_metrics(preds, labels):
    """Computes performance metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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

def eval_model(model, data_loader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
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

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    print(f"  Validation loss: {avg_loss:.4f}")
    print(f"  Validation Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    return avg_loss, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_data.csv'))
    print("Train and test data loaded.")

    # create label mapping
    unique_labels = train_df['target_label'].unique()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(unique_labels)
    print(f"Found {num_labels} unique labels.")

    # initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    print("Model and tokenizer initialized.")

    # create datasets and dataLoaders
    train_dataset = NewsClassifierDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print("DataLoaders created.")

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = eval_model(model, val_loader, device)
    
    print("\nTraining complete.")

    # save fine-tuned model and tokenizer
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Model saved to '{OUTPUT_MODEL_DIR}'")

if __name__ == "__main__":
    main()