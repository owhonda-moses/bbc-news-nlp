# FILE: src/4_augment_data.py

import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


DATA_PATH = './data'
MODEL_NAME = 't5-base'

# list of classes with an F1-score < 0.75 in the weighted report
TARGET_CLASSES = [
    'Formula 1', 'Literature', 'Theatre', 'General Business', 
    'General Sport', 'Tennis', 'Cricket', 'General Entertainment'
]
NUM_AUGMENTATIONS = 5
BATCH_SIZE = 4 # batching for faster generation
OUTPUT_CSV_PATH = os.path.join(DATA_PATH, 'train_data_augmented.csv')

def batch_augment_text(texts, model, tokenizer, device, num_versions=5):
    """Generates paraphrased versions for a batch of texts."""
    paraphrased_texts = []
    # prepend the task prefix to each text in the batch
    input_prompts = [f"paraphrase: {text}" for text in texts]
    
    # tokenize the batch
    inputs = tokenizer(input_prompts, return_tensors='pt', padding=True, max_length=1024, truncation=True).to(device)
    
    # generate multiple versions for the whole batch
    outputs = model.generate(
        **inputs,
        max_length=1200,
        num_return_sequences=num_versions,
        num_beams=5,
        early_stopping=True
    )
    
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # the output will be [v1_text1, v2_text1..., v1_text2, v2_text2...]
    # we need to group them by the original text
    for i in range(len(texts)):
        start_index = i * num_versions
        end_index = start_index + num_versions
        paraphrased_texts.append(decoded_outputs[start_index:end_index])
        
    return paraphrased_texts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {MODEL_NAME} model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded.")

    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_data.csv'))
    print(f"Original training data has {len(train_df)} samples.")

    target_df = train_df[train_df['target_label'].isin(TARGET_CLASSES)].copy()
    print(f"Found {len(target_df)} articles to augment from {len(TARGET_CLASSES)} classes.")

    augmented_rows = []
    
    # process in batches
    target_texts = target_df['text'].tolist()
    for i in range(0, len(target_texts), BATCH_SIZE):
        batch_texts = target_texts[i:i+BATCH_SIZE]
        batch_rows = target_df.iloc[i:i+BATCH_SIZE]
        print(f"\nAugmenting batch {i//BATCH_SIZE + 1}...")

        generated_versions_for_batch = batch_augment_text(batch_texts, model, tokenizer, device, NUM_AUGMENTATIONS)
        
        for j, (index, original_row) in enumerate(batch_rows.iterrows()):
            new_texts = generated_versions_for_batch[j]
            for k, new_text in enumerate(new_texts):
                new_row = original_row.copy()
                new_row['text'] = new_text
                new_row['filename'] = f"aug_{k+1}_{original_row['filename']}"
                augmented_rows.append(new_row)

    if not augmented_rows:
        print("No articles were augmented.")
        return

    augmented_df = pd.DataFrame(augmented_rows)
    final_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
    
    print(f"Original training set size: {len(train_df)}")
    print(f"Number of new samples generated: {len(augmented_df)}")
    print(f"New augmented training set size: {len(final_train_df)}")
    
    final_train_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nAugmented training data saved to '{OUTPUT_CSV_PATH}'")
    
    if device.type == 'cuda':
        print("Clearing GPU cache.")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()