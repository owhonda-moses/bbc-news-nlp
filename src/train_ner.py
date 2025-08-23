import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AdamW
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm


DATA_PATH = './data'
NER_DATA_FILE = os.path.join(DATA_PATH, 'ner_training_data.csv')
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 256 # shorter max length is fine for single sentences
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
OUTPUT_MODEL_DIR = './models/ner-model'

def load_and_prepare_data(filepath):
    """Loads NER data and creates tag mappings."""
    df = pd.read_csv(filepath)
    
    # create a comprehensive list of all unique tags
    unique_tags = set()
    for tags in df['ner_tags'].values:
        unique_tags.update(tags.split())
    
    tag2id = {tag: id for id, tag in enumerate(sorted(list(unique_tags)))}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    return df, tag2id, id2tag

class NERDataset(Dataset):
    """Custom PyTorch Dataset for NER."""
    def __init__(self, dataframe, tokenizer, tag2id, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['tokens'].apply(lambda x: x.split()).tolist()
        self.tags = dataframe['ner_tags'].apply(lambda x: x.split()).tolist()
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.texts[index]
        tags = self.tags[index]
        
        # tokenize and align labels
        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True, 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_len)
        
        labels = [self.tag2id[tag] for tag in tags]
        
        # align labels with word-piece tokens
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # this is the start of a new token
                if i < len(labels):
                    encoded_labels[idx] = labels[i]
                    i += 1
        
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

def train_model(model, train_loader, optimizer, device):
    """Main training loop for the NER model."""
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    print(f"  Training Loss: {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data and create tag mappings
    df, tag2id, id2tag = load_and_prepare_data(NER_DATA_FILE)
    print(f"Found {len(tag2id)} unique NER tags.")
    
    # split data
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

    # initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
    )
    model.to(device)
    print("Model and tokenizer initialized.")

    # create datasets and dataLoaders
    train_dataset = NERDataset(train_df, tokenizer, tag2id, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Setup complete. Starting training.")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_model(model, train_loader, optimizer, device)

    print("\nTraining complete.")

    # save final model
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"NER model saved to '{OUTPUT_MODEL_DIR}'")
    
    if device.type == 'cuda':
        print("Clearing GPU cache.")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()