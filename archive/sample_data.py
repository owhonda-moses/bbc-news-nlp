import pandas as pd
import os


INPUT_FILE = '././data/output/ner/labeled_data.csv'
OUTPUT_FILE = '././data/output/ner/bootstrap_review.csv'
STATE_FILE = '././data/output/ner/sampled_indices.txt'
SAMPLE_SIZE = 1000 # number of sentences to correct

def main():
    """
    Creates a smaller, random sample from the bootstrap file,
    excluding any indices that have been sampled before.
    """
    print(f"Creating a random sample of {SAMPLE_SIZE} sentences.")

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found at '{INPUT_FILE}'.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # load indices already used
    used_indices = set()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            used_indices = set(map(int, f.read().splitlines()))
            
    # create a pool of available data to sample from
    available_df = df.drop(index=used_indices, errors='ignore')
    
    if len(available_df) == 0:
        print("No new sentences available to sample.")
        return

    if len(available_df) < SAMPLE_SIZE:
        print(f"Only {len(available_df)} new sentences available, which is less than the requested {SAMPLE_SIZE}.")
        sample_size = len(available_df)
    else:
        sample_size = SAMPLE_SIZE

    sample_df = available_df.sample(n=sample_size, random_state=42)
    sample_df.to_csv(OUTPUT_FILE, index=False)
    
    # update the state file
    new_indices = set(sample_df.index)
    all_used_indices = used_indices.union(new_indices)
    
    with open(STATE_FILE, 'w') as f:
        for index in sorted(list(all_used_indices)):
            f.write(f"{index}\n")

    print(f"\nSaved {len(sample_df)} new sentences to '{OUTPUT_FILE}'.")
    print(f"Updated state file '{STATE_FILE}' with {len(new_indices)} new indices.")

if __name__ == "__main__":
    main()