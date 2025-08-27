import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = './data'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'train_data.csv')
OUTPUT_TRAIN_PATH = os.path.join(DATA_PATH, 'train_heuristic.csv')
OUTPUT_VAL_PATH = os.path.join(DATA_PATH, 'val_heuristic.csv')

# keyword definitions
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


def classify_subcategory(text, main_cat):
    if main_cat not in subcategory_keywords: return main_cat
    lower_text = text.lower()
    for sub_cat, keywords in subcategory_keywords[main_cat].items():
        if any(re.search(r'\b' + re.escape(key) + r'\b', lower_text) for key in keywords):
            return sub_cat
    return f"General {main_cat.capitalize()}"

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
    print("Preparing heuristic data")
    df = pd.read_csv(SOURCE_DATA_PATH)
    
    df['sub_category'] = df.apply(lambda row: classify_subcategory(row['text'], row['main_category']), axis=1)
    df['target_label'] = df['sub_category']
    
    label_counts = df['target_label'].value_counts()
    labels_to_remove = label_counts[label_counts < 2].index
    if not labels_to_remove.empty:
        df = df[~df['target_label'].isin(labels_to_remove)]

    train_df, val_df = smart_split(df)
    
    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    val_df.to_csv(OUTPUT_VAL_PATH, index=False)
    
    print(f"  Training set saved to '{OUTPUT_TRAIN_PATH}'")
    print(f"  Validation set saved to '{OUTPUT_VAL_PATH}'")

if __name__ == "__main__":
    main()