import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = '././data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'bbc_raw')
OUTPUT_DIR = os.path.join(DATA_PATH, 'output')
OUTPUT_TRAINING_PATH = os.path.join(OUTPUT_DIR, 'train_data.csv')
OUTPUT_TEST_PATH = os.path.join(OUTPUT_DIR, 'test_samples.csv')
TEST_SET_SIZE = 0.10

def load_raw_data(path):
    data = []
    main_categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for category in main_categories:
        folder_path = os.path.join(path, category)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    unique_id = f"{category}_{filename}"
                    data.append({
                        'unique_id': unique_id, 
                        'filename': filename, 
                        'text': f.read(), 
                        'main_category': category
                    })
    return pd.DataFrame(data)

def main():
    print("Performing master split")
    
    full_df = load_raw_data(RAW_DATA_PATH)
    
    # assign temporary label to stratify on
    full_df['target_label'] = full_df['main_category']

    training_pool_df, gold_standard_df = train_test_split(
        full_df,
        test_size=TEST_SET_SIZE,
        random_state=42,
        stratify=full_df['target_label']
    )
    
    # drop temporary label
    training_pool_df = training_pool_df.drop(columns=['target_label'])
    gold_standard_df = gold_standard_df.drop(columns=['target_label'])
    
    gold_standard_df.to_csv(OUTPUT_TEST_PATH, index=False)
    training_pool_df.to_csv(OUTPUT_TRAINING_PATH, index=False)
    
    print(f"  Test pool articles saved: {len(gold_standard_df)} samples")
    print(f"  Training pool articles saved: {len(training_pool_df)} samples")

if __name__ == "__main__":
    main()