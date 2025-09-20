import os
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


DATA_PATH = '././data/output'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'test_set.csv')
OUTPUT_PATH = os.path.join(DATA_PATH, 'april_events', 'summaries.csv')
MODEL_NAME = 'facebook/bart-large-cnn'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading summarization model: {MODEL_NAME}...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    df = pd.read_csv(SOURCE_DATA_PATH)
    print(f"Loaded {len(df)} documents from the unified test set.")

    # use pre-calculated 'mentions_april' column
    april_df = df[df['mentions_april'] == True].copy()
    print(f"Found {len(april_df)} articles that mention April.")

    if april_df.empty:
        print("No articles mentioning April found in the test set.")
        return
        
    print("\nGenerating summaries")
    summaries = []
    for index, row in april_df.iterrows():
        article_text = row['text']
        print(f"\nSummarizing: {row['unique_id']}...")

        inputs = tokenizer([article_text], max_length=1024, return_tensors='pt', truncation=True).to(device)
        
        summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=4, 
            min_length=30,
            max_length=100,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        
        print(f" Summary: {summary}")
        
    april_df['summary'] = summaries
    april_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSummaries saved to '{OUTPUT_PATH}'")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()