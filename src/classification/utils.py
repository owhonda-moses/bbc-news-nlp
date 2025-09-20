import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


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


def compute_metrics(preds, labels):
    """Computes performance metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


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
    print(f"  Loss: {avg_loss:.4f} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    return avg_loss, metrics


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