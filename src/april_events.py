import os
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


DIR = './data'
DATA_PATH = os.path.join(DIR, 'train_data_augmented.csv') 
MODEL_NAME = 'facebook/bart-large-cnn'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the pre-trained summarization model and tokenizer
    print(f"Loading summarization model: {MODEL_NAME}")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded.")

    df = pd.read_csv(DATA_PATH)
    # remove duplicate articles from augmentation
    df.drop_duplicates(subset=['filename'], inplace=True)
    print(f"Loaded {len(df)} unique documents.")

    # filter for articles mentioning April and run case-insensitive search
    april_df = df[df['text'].str.contains(r'\bApril\b', case=False, na=False)].copy()
    print(f"Found {len(april_df)} articles that mention 'April'.")

    if april_df.empty:
        print("No articles found mentioning April")
        return
        
    print("\n Generating summaries for April events")

    # generate a summary for each article
    summaries = []
    for index, row in april_df.iterrows():
        article_text = row['text']
        print(f"\nSummarizing article: {row['filename']}...")

        # prepare the article text for the model
        inputs = tokenizer([article_text], max_length=1024, return_tensors='pt', truncation=True).to(device)
        
        # generate the summary
        summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=4, 
            min_length=30, # set a minimum length for the summary
            max_length=100, # set a maximum length
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        summaries.append(summary)
        
        print(f" Summary: {summary}")
        
    # save results to new CSV
    april_df['summary'] = summaries
    output_path = os.path.join(DIR, 'april_events_summaries.csv')
    april_df.to_csv(output_path, index=False)
    print(f"\nSummaries saved to '{output_path}'")

    if device.type == 'cuda':
        print("Clearing GPU cache.")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()