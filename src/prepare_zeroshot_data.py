import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DATA_PATH = './data'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'train_data.csv')
OUTPUT_TRAIN_PATH = os.path.join(DATA_PATH, 'train_zeroshot.csv')
OUTPUT_VAL_PATH = os.path.join(DATA_PATH, 'val_zeroshot.csv')
ZERO_SHOT_MODEL = 'facebook/bart-large-mnli'
BATCH_SIZE = 32
CHUNK_SIZE = 128

CANDIDATE_LABELS = [
    'Stock Market', 'Mergers & Acquisitions', 'Company News', 'Economy',
    'Cinema', 'Music', 'Theatre', 'Literature', 'TV & Radio', 'Celebrity News',
    'Football', 'Cricket', 'Rugby', 'Tennis', 'Athletics', 'Formula 1',
    'Politics', 'Tech'
]

def smart_split(df, test_size=0.2, random_state=42):
    train_indices, val_indices = [], []
    grouped = df.groupby('target_label')
    for label, group in grouped:
        if len(group) < 2:
            train_indices.extend(group.index)
            continue
        group_train, group_val = train_test_split(group.index, test_size=test_size, random_state=random_state, stratify=group['target_label'])
        if len(group_train) == 0 and len(group_val) > 0:
            train_indices.append(group_val.pop(0))
        train_indices.extend(group_train)
        val_indices.extend(group_val)
    return df.loc[train_indices], df.loc[val_indices]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=0 if device.type == 'cuda' else -1)
    
    df = pd.read_csv(SOURCE_DATA_PATH)
    
    print("Getting zero-shot predictions for all articles")
    texts_to_classify = [" ".join(text.split()[:400]) for text in df['text']]
    results = []
    with tqdm(total=len(texts_to_classify), desc="Labeling Articles") as pbar:
        for i in range(0, len(texts_to_classify), CHUNK_SIZE):
            chunk = texts_to_classify[i : i + CHUNK_SIZE]
            chunk_results = classifier(chunk, candidate_labels=CANDIDATE_LABELS, batch_size=BATCH_SIZE, multi_label=False)
            results.extend(chunk_results)
            pbar.update(len(chunk))

    df['predicted_label'] = [res['labels'][0] for res in results]
    df['confidence_score'] = [res['scores'][0] for res in results]
    
    final_threshold = df['confidence_score'].median()
    labeled_df = df[df['confidence_score'] >= final_threshold].copy()
    labeled_df['target_label'] = labeled_df['predicted_label']
    
    final_df = labeled_df[['unique_id', 'filename', 'text', 'main_category', 'target_label']]
    print(f"\nLabeled {len(final_df)} articles.")

    train_df, val_df = smart_split(final_df)

    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    val_df.to_csv(OUTPUT_VAL_PATH, index=False)

    print(f"  Training set saved to '{OUTPUT_TRAIN_PATH}'")
    print(f"  Validation set saved to '{OUTPUT_VAL_PATH}'")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()