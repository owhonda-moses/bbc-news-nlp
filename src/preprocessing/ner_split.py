import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '././data/output'
INPUT_FILE = os.path.join(DATA_PATH, 'test_set.csv')
VAL_OUTPUT_FILE = os.path.join(DATA_PATH, 'ner', 'ner_val.csv')
TEST_OUTPUT_FILE = os.path.join(DATA_PATH, 'ner', 'ner_test.csv')
VAL_SIZE = 0.20 # 20% for val

def main():
    """
    Splits the unified, manually-annotated test set into a
    validation set and a test set for NER
    """
    df = pd.read_csv(INPUT_FILE)

    # select rows with NER annotations
    annotated_df = df[df['ner_entities'] != '[]'].copy()
    print(f"Found {len(annotated_df)} articles with NER annotations")

    # remove classes with only one sample
    class_counts = annotated_df['gold_label'].value_counts()
    single_instance_classes = class_counts[class_counts < 2].index.tolist()

    if single_instance_classes:
        print(f"\nRemoving {len(single_instance_classes)} classes with only 1 sample: {single_instance_classes}")
        annotated_df = annotated_df[~annotated_df['gold_label'].isin(single_instance_classes)]
        print(f"Proceeding with {len(annotated_df)} articles.")

    # stratify by the main classification label to ensure both sets are representative
    val_df, test_df = train_test_split(
        annotated_df,
        test_size=1-VAL_SIZE,
        random_state=42,
        stratify=annotated_df['gold_label']
    )

    val_df.to_csv(VAL_OUTPUT_FILE, index=False)
    test_df.to_csv(TEST_OUTPUT_FILE, index=False)

    print(f"\nNER validation set contains {len(val_df)} articles: '{VAL_OUTPUT_FILE}'")
    print(f"NER test set contains {len(test_df)} articles: '{TEST_OUTPUT_FILE}'")

if __name__ == "__main__":
    main()