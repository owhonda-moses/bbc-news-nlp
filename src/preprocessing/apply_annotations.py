import os
import pandas as pd
import json

# define filepaths and files
DATA_PATH = '././data/output'
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(DATA_PATH, 'test_samples.csv')
ANNOTATIONS_PATH = os.path.join(SRC_PATH, 'annotations.json')
TEST_SET_PATH = os.path.join(DATA_PATH, 'test_set.csv') # output


def main():
    """
    Consolidates all manual annotations from a JSON file
    into a single, unified test set.
    """
    print("Creating test set from annotations.json...")

    # load annotations file
    try:
        with open(ANNOTATIONS_PATH, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Annotation file not found at '{ANNOTATIONS_PATH}'.")
        return
    
    # load base sample file
    try:
        sample_df = pd.read_csv(SAMPLE_PATH)
    except FileNotFoundError:
        print(f"Sample file not found at '{SAMPLE_PATH}'.")
        return

    # apply annotations-
    sample_df['gold_label'] = sample_df['unique_id'].map(lambda x: annotations.get(x, {}).get('sub_classification'))
    sample_df['ner_entities'] = sample_df['unique_id'].apply(lambda x: json.dumps(annotations.get(x, {}).get('ner_entities', [])))
    
    # generate mentions_april flag
    sample_df['mentions_april'] = sample_df['text'].str.contains(r'\bApril\b', case=False, na=False)

    # drop missing rows if any
    unannotated_mask = sample_df['gold_label'].isnull()
    unannotated_count = unannotated_mask.sum()
    if unannotated_count > 0:
        ids_to_drop = sample_df[unannotated_mask]['unique_id'].tolist()
        print(f"{unannotated_count} sample(s) not found in annotations and will be dropped.")
        print(f"Dropped IDs: {ids_to_drop}")
        sample_df.dropna(subset=['gold_label'], inplace=True)
    else:
        print(f"All samples annotated.")

    # reorder columns and create final dataframe
    final_df = sample_df[[
        'unique_id', 'filename', 'text', 'main_category',
        'gold_label', 'ner_entities', 'mentions_april'
    ]].copy()

    # save the test set
    final_df.to_csv(TEST_SET_PATH, index=False)

    print(f"\nSaved the unified test set to '{TEST_SET_PATH}'")

if __name__ == "__main__":
    main()