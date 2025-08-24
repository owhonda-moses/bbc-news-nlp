import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.train_baseline import NewsClassifierDataset


DATA_PATH = './data'
# MODEL_PATH = './models/baseline-classifier'
MODEL_PATH = './models/weighted-classifier'
# MODEL_PATH = './models/augmented-classifier'
TEXT_PATH = os.path.join(DATA_PATH, 'train_data.csv') 
MAX_LEN = 512
BATCH_SIZE = 16


# evaluation function
def get_predictions(model, data_loader, device):
    """Gets model predictions for a given dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    print("Model and tokenizer loaded.")

    # load data and get label mappings from the model config
    # val_df = pd.read_csv(os.path.join(DATA_PATH, 'val_data.csv'))
    
    annotations_df = pd.read_csv(os.path.join(DATA_PATH, 'annotated_test_set.csv'))
    source_df = pd.read_csv(TEXT_PATH)[['filename', 'text']].drop_duplicates(subset=['filename'])
    val_df = pd.merge(annotations_df, source_df, on='filename', how='left')
    if val_df['text'].isnull().any():
        raise ValueError("Not all texts for gold-standard filenames were found.")    
    val_df['gold_label'] = val_df['gold_label'].replace({'Mergers & Acq': 'Mergers & Acquisitions'})
    val_df['target_label'] = val_df['gold_label']
    print(f"Loaded {len(val_df)} samples from the gold-standard test set.")

    label2id = model.config.label2id
    id2label = model.config.id2label
    label_names = list(label2id.keys())
    print("Loaded validation data.")

    # create dataset and dataloader
    val_dataset = NewsClassifierDataset(val_df, tokenizer, label2id, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print("DataLoader created.")

    y_pred, y_true = get_predictions(model, val_loader, device) # get predictions
    
    # convert numerical predictions back to string labels for report
    y_pred_labels = [id2label[pred] for pred in y_pred]
    y_true_labels = [id2label[true] for true in y_true]

    # print classification report
    report = classification_report(y_true_labels, y_pred_labels, labels=label_names, zero_division=0)
    print(report)

    # save confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # save plot to file
    output_path = './img/confusion_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Confusion matrix saved to '{output_path}'")

if __name__ == "__main__":
    main()