import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = './data'
SOURCE_TEXT_PATH = os.path.join(DATA_PATH, 'train_data.csv')
OUTPUT_TRAIN_PATH = os.path.join(DATA_PATH, 'train_final.csv')
OUTPUT_VAL_PATH = os.path.join(DATA_PATH, 'val_final.csv')
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    print("Loading full dataset to create a smart split")
    df = pd.read_csv(SOURCE_TEXT_PATH)

    # ensure we don't have single-sample classes
    label_counts = df['target_label'].value_counts()
    labels_to_remove = label_counts[label_counts < 2].index
    if not labels_to_remove.empty:
        df = df[~df['target_label'].isin(labels_to_remove)]

    print(f"Working with {len(df)} total samples across {df['target_label'].nunique()} categories.")
    
    train_indices = []
    val_indices = []
    
    # group by the target label to handle each category separately
    grouped = df.groupby('target_label')
    
    for label, group in grouped:
        # if a group is too small to split put all of it in training
        if len(group) < 2:
            train_indices.extend(group.index)
            continue
        
        # sklearn's train_test_split handles small groups well
        # as long as n_splits > 1, which it is for test_size=0.2
        group_train_indices, group_val_indices = train_test_split(
            group.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=group['target_label'] # stratify within the group
        )
        
        # handle edge cases where a small group might not get split properly
        if len(group_train_indices) == 0:
            # if split results in zero training samples, move one from validation
            train_indices.append(group_val_indices.pop(0))
        
        train_indices.extend(group_train_indices)
        val_indices.extend(group_val_indices)

    # create final dataframes
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]

    print("\n Split Summary")
    print(f"Total training samples: {len(train_df)}")
    print(f"Total validation samples: {len(val_df)}")
    
    # verify all original labels are in the new training set
    original_labels = set(df['target_label'].unique())
    train_labels = set(train_df['target_label'].unique())
    missing_labels = original_labels - train_labels
    
    if not missing_labels:
        print("All categories are represented in the training set.")
    else:
        print(f"These labels are still missing from the training set: {missing_labels}")
        
    # save the new splits
    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    val_df.to_csv(OUTPUT_VAL_PATH, index=False)
    
    print(f"\nNew data splits saved to '{OUTPUT_TRAIN_PATH}' and '{OUTPUT_VAL_PATH}'")

if __name__ == "__main__":
    main()