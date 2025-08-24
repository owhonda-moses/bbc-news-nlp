import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = './data'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'train_data.csv') 
OUTPUT_SAMPLE_PATH = os.path.join(DATA_PATH, 'annotation_sample.csv')
SAMPLE_SIZE = 0.05 # sample 5% of the training data

def main():
    print(f"Loading data from: {SOURCE_DATA_PATH}")
    df = pd.read_csv(SOURCE_DATA_PATH)

    print(f"Original dataset size: {len(df)}")
    
    # ensure every class has at least 2 samples to be stratified
    label_counts = df['target_label'].value_counts()
    labels_to_remove = label_counts[label_counts < 2].index
    if not labels_to_remove.empty:
        print(f"Removing rare classes with < 2 samples for stratification: {list(labels_to_remove)}")
        df = df[~df['target_label'].isin(labels_to_remove)]
        print(f"Filtered dataset size: {len(df)}")

    # perform stratified sampling and use test data as sample.
    _, sample_df = train_test_split(
        df,
        test_size=SAMPLE_SIZE,
        random_state=42,
        stratify=df['target_label']
    )
    
    # copy heuristic label as a starting point
    sample_df['gold_label'] = sample_df['target_label']
    
    # select and reorder columns
    annotation_df = sample_df[['filename', 'text', 'main_category', 'target_label', 'gold_label']].copy()
    annotation_df.rename(columns={'target_label': 'heuristic_label'}, inplace=True)
    
    # save the sample
    annotation_df.to_csv(OUTPUT_SAMPLE_PATH, index=False)
    
    print(f"\nSuccessfully created a stratified sample of {len(annotation_df)} articles.")
    print(f"The sample is ready for manual annotation at: '{OUTPUT_SAMPLE_PATH}'")

if __name__ == "__main__":
    main()