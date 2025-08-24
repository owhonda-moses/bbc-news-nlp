import os
import pandas as pd


DATA_PATH = './data'
ANNOTATION_SAMPLE_PATH = os.path.join(DATA_PATH, 'gold_standard.csv')
GOLD_STANDARD_TEST_SET_PATH = os.path.join(DATA_PATH, 'annotated_test_set.csv')


# map filename to the corrected gold label
gemini_corrections = {
    '458.txt': 'Company News', '312.txt': 'Rugby',
    '283.txt': 'Music', '411.txt': 'Rugby', '452.txt': 'Tennis',
    '361.txt': 'Economy', '383.txt': 'Rugby', '044.txt': 'Economy',
    '207.txt': 'Company News', '188.txt': 'Mergers & Acquisitions',
    '264.txt': 'Music', '140.txt': 'Music', '105.txt': 'Music',
    '506.txt': 'Company News', '438.txt': 'Tennis', '060.txt': 'Company News',
    '382.txt': 'Economy' if 'Ban on forced retirement' in 'text' else 'tech'
}


def apply_gold_labels(df, corrections):
    """Applies the high-quality corrected labels to the dataframe."""
    print("Applying gold-standard annotations")
    
    # start with the heuristic label as the base and apply corrections
    df['gold_label'] = df['heuristic_label']
    df['gold_label'] = df.apply(
        lambda row: corrections.get(row['filename'], row['gold_label']),
        axis=1
    )
    
    # count how many labels were changed
    num_changes = (df['heuristic_label'] != df['gold_label']).sum()
    print(f"Corrected {num_changes} labels based on contextual analysis.")
    return df

def main():
    print(f"Loading annotation sample from: {ANNOTATION_SAMPLE_PATH}")
    try:
        annotation_df = pd.read_csv(ANNOTATION_SAMPLE_PATH.replace('annotation_sample.csv', 'gold_standard.csv'))
    except FileNotFoundError:
        print(f"Annotation file not found.")
        return

    # apply corrections
    gold_standard_df = apply_gold_labels(annotation_df, gemini_corrections)
    
    # save the corrected test set
    gold_standard_df.to_csv(GOLD_STANDARD_TEST_SET_PATH, index=False)
    
    print(f"\nSuccessfully created the gold standard test set with {len(gold_standard_df)} samples.")
    print(f"File saved to: '{GOLD_STANDARD_TEST_SET_PATH}'")

if __name__ == "__main__":
    main()