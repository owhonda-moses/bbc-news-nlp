import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

dir = './data'
data_path = os.path.join(dir, "bbc_raw")
main_categories_to_process = ['business', 'entertainment', 'sport', 'tech', 'politics']

# keyword definitions for bootstrapping sub-category labels
# these are used to generate the initial, heuristic labels.
subcategory_keywords = {
    'business': {
        'Stock Market': ['stock', 'market', 'shares', 'dow jones', 'nasdaq', 'ftse', 'trading'],
        'Mergers & Acquisitions': ['acquisition', 'merger', 'takeover', 'buyout', 'deal'],
        'Company News': ['company', 'firm', 'corp', 'inc', 'plc', 'results', 'profits', 'sales'],
        'Economy': ['economic', 'economy', 'growth', 'inflation', 'interest rates', 'gdp'],
    },
    'entertainment': {
        'Cinema': ['film', 'movie', 'cinema', 'box office', 'actor', 'actress', 'director', 'hollywood'],
        'Music': ['music', 'singer', 'band', 'album', 'song', 'chart', 'grammy', 'concert'],
        'Theatre': ['theatre', 'stage', 'play', 'musical', 'broadway'],
        'Literature': ['book', 'novel', 'author', 'writer', 'prize', 'literary'],
        'TV & Radio': ['tv', 'television', 'radio', 'bbc', 'programme', 'series'],
        'Celebrity News': ['star', 'celebrity', 'awards', 'oscar', 'bafta'],
    },
    'sport': {
        'Football': ['football', 'club', 'league', 'cup', 'manchester united', 'arsenal', 'chelsea', 'player'],
        'Cricket': ['cricket', 'england', 'test match', 'ashes', 'batsman', 'bowler'],
        'Rugby': ['rugby', 'six nations', 'world cup', 'england team'],
        'Tennis': ['tennis', 'wimbledon', 'grand slam', 'atp', 'wta', 'nadal', 'federer'],
        'Athletics': ['athletics', 'olympics', 'marathon', 'champion', 'track'],
        'Formula 1': ['formula one', 'f1', 'grand prix', 'driver', 'ferrari', 'mclaren'],
    }
}

# data loading and labeling functions
def classify_subcategory(text, main_cat):
    """Assigns a sub-category based on keyword matching."""
    if main_cat not in subcategory_keywords:
        return main_cat # return the main category for tech/politics

    lower_text = text.lower()
    for sub_cat, keywords in subcategory_keywords[main_cat].items():
        if any(re.search(r'\b' + re.escape(key) + r'\b', lower_text) for key in keywords):
            return sub_cat
    return f"General {main_cat.capitalize()}"

def load_data(path, categories):
    """Loads text files into a pandas DataFrame."""
    data = []
    print("Starting data load process.")
    for category in categories:
        folder_path = os.path.join(path, category)
        if not os.path.isdir(folder_path):
            print(f"Directory not found for category '{category}'. Skipping.")
            continue
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        data.append({'filename': filename, 'text': text, 'main_category': category})
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    
    df = pd.DataFrame(data)
    print(f"Data load complete. Loaded {len(df)} documents.")
    return df


# load raw data and generate sub-category labels
full_df = load_data(data_path, main_categories_to_process)
print("Generating sub-category labels.")
full_df['sub_category'] = full_df.apply(
    lambda row: classify_subcategory(row['text'], row['main_category']),
    axis=1
)

# create a single target label for our first model
# we will use the sub-category for b/e/s and the main category for p/t
full_df['target_label'] = full_df['sub_category']

# display info about the prepared data
print("\nDataset Info")
full_df.info()

print("\nLabel Distribution")
print(full_df['target_label'].value_counts())
label_counts = full_df['target_label'].value_counts()

# Exclude labels that appear fewer than 2 times
labels_to_remove = label_counts[label_counts < 2].index
print(f"\nRemoving rare classes with < 2 samples: {list(labels_to_remove)}")
filtered_df = full_df[~full_df['target_label'].isin(labels_to_remove)]
print(f"Original dataset size: {len(full_df)}")
print(f"Filtered dataset size: {len(filtered_df)}")

# split the data for training and validation using stratify
train_df, val_df = train_test_split(
    filtered_df,
    test_size=0.2,
    random_state=42,
    stratify=filtered_df['target_label']
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Save the datasets
train_df.to_csv(os.path.join(dir, 'train_data.csv'), index=False)
val_df.to_csv(os.path.join(dir, 'val_data.csv'), index=False)
print(f"\nTraining and validation data saved to '{dir}'.")