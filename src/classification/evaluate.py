import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.classification.utils import NewsClassifierDataset

# define filepaths
DATA_PATH = '././data/output'
# MODEL_PATH = '././models/baseline-classifier'
# MODEL_PATH = '././models/weighted-classifier'
MODEL_PATH = '././models/augmented-classifier'

TEST_SET_PATH = os.path.join(DATA_PATH, 'test_set.csv')
MAX_LEN = 512
BATCH_SIZE = 4

def get_predictions(model, data_loader, device):
    """Gets model predictions for a given dataset."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    return all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    print(f"Model loaded from: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_SET_PATH)
    test_df['target_label'] = test_df['gold_label'] # use annotated labels
    
    label2id = model.config.label2id
    known_labels = set(label2id.keys())
    
    # filter for labels the model was trained on
    test_df = test_df[test_df['target_label'].isin(known_labels)].copy()
    print(f"Loaded {len(test_df)} samples from the test set.")
    
    id2label = model.config.id2label
    report_labels = sorted([label for label in test_df['target_label'].unique() if label in known_labels])
    
    # create dataset and dataloader
    test_dataset = NewsClassifierDataset(test_df, tokenizer, label2id, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("DataLoader created.")

    y_pred_ids, y_true_ids = get_predictions(model, test_loader, device) # get predictions
    
    # convert numerical predictions back to string labels for report
    y_pred_labels = [id2label.get(pred_id) for pred_id in y_pred_ids]
    y_true_labels = [id2label.get(true_id) for true_id in y_true_ids]

    report = classification_report(y_true_labels, y_pred_labels, labels=report_labels, zero_division=0)
    print(report)
    
    # save confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=report_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_labels, yticklabels=report_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # save plot to file
    output_path = '././data/output/img/confusion_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Confusion matrix saved to '{output_path}'")

if __name__ == "__main__":
    main()