import os
import pandas as pd


DATA_PATH = './data'
SAMPLE_PATH = os.path.join(DATA_PATH, 'test_samples.csv')
TEST_SET_PATH = os.path.join(DATA_PATH, 'test_set.csv')

# annotations
annotations = {
    'politics_377.txt': 'Politics', 'sport_235.txt': 'Football', 'entertainment_090.txt': 'Cinema',
    'politics_394.txt': 'Politics', 'politics_207.txt': 'Politics', 'sport_041.txt': 'Athletics',
    'business_026.txt': 'Company News', 'business_104.txt': 'Company News', 'tech_357.txt': 'Tech',
    'business_025.txt': 'Company News', 'business_037.txt': 'Company News', 'tech_023.txt': 'Tech',
    'business_294.txt': 'Economy', 'sport_333.txt': 'Rugby', 'politics_375.txt': 'Politics',
    'sport_119.txt': 'Football', 'business_300.txt': 'Mergers & Acquisitions', 'politics_381.txt': 'Politics',
    'sport_154.txt': 'Football', 'tech_381.txt': 'Tech', 'business_424.txt': 'Company News',
    'sport_433.txt': 'Tennis', 'tech_161.txt': 'Tech', 'sport_133.txt': 'Football',
    'sport_313.txt': 'Rugby', 'entertainment_052.txt': 'Cinema', 'entertainment_238.txt': 'Music',
    'sport_439.txt': 'Tennis', 'politics_277.txt': 'Politics', 'business_415.txt': 'Economy',
    'sport_498.txt': 'Tennis', 'entertainment_048.txt': 'Cinema', 'business_100.txt': 'Company News',
    'business_304.txt': 'Company News', 'politics_199.txt': 'Politics', 'sport_204.txt': 'Football',
    'entertainment_264.txt': 'Music', 'politics_317.txt': 'Politics', 'tech_239.txt': 'Tech',
    'sport_084.txt': 'Athletics', 'tech_352.txt': 'Tech', 'tech_079.txt': 'Tech',
    'business_508.txt': 'Company News', 'sport_219.txt': 'Football', 'sport_374.txt': 'Rugby',
    'sport_157.txt': 'Football', 'business_248.txt': 'Economy', 'tech_171.txt': 'Tech',
    'sport_449.txt': 'Tennis', 'business_447.txt': 'Mergers & Acquisitions', 'sport_118.txt': 'Football',
    'politics_319.txt': 'Politics', 'business_252.txt': 'Tech', 'tech_132.txt': 'Tech',
    'business_241.txt': 'Economy', 'tech_213.txt': 'Tech', 'sport_429.txt': 'Tennis',
    'tech_067.txt': 'Tech', 'politics_273.txt': 'Politics', 'business_060.txt': 'Company News',
    'politics_037.txt': 'Politics', 'tech_250.txt': 'Tech', 'entertainment_357.txt': 'Cinema',
    'business_012.txt': 'Economy', 'business_174.txt': 'Company News', 'entertainment_209.txt': 'TV & Radio',
    'politics_341.txt': 'Politics', 'politics_314.txt': 'Politics', 'sport_424.txt': 'Tennis',
    'sport_132.txt': 'Football', 'politics_079.txt': 'Politics', 'sport_479.txt': 'Tennis',
    'entertainment_314.txt': 'Cinema', 'tech_059.txt': 'Tech', 'sport_146.txt': 'Football',
    'sport_197.txt': 'Football', 'politics_216.txt': 'Politics', 'politics_308.txt': 'Politics',
    'politics_412.txt': 'Politics', 'business_179.txt': 'Mergers & Acquisitions', 'entertainment_318.txt': 'Music',
    'entertainment_142.txt': 'Music', 'tech_265.txt': 'Tech', 'entertainment_214.txt': 'TV & Radio',
    'entertainment_255.txt': 'Music', 'entertainment_207.txt': 'TV & Radio', 'tech_145.txt': 'Music',
    'business_460.txt': 'Mergers & Acquisitions', 'business_039.txt': 'Mergers & Acquisitions', 'tech_105.txt': 'Tech',
    'tech_324.txt': 'Company News', 'business_053.txt': 'Company News', 'business_215.txt': 'Economy',
    'entertainment_375.txt': 'Cinema', 'business_421.txt': 'Mergers & Acquisitions', 'sport_293.txt': 'Rugby',
    'entertainment_184.txt': 'TV & Radio', 'politics_238.txt': 'Politics', 'politics_075.txt': 'Politics',
    'entertainment_347.txt': 'Cinema', 'sport_471.txt': 'Tennis', 'business_333.txt': 'Company News',
    'entertainment_247.txt': 'Music', 'politics_237.txt': 'Politics', 'business_317.txt': 'Economy',
    'tech_309.txt': 'Tech', 'entertainment_188.txt': 'TV & Radio', 'tech_198.txt': 'Tech',
    'politics_219.txt': 'Politics', 'tech_225.txt': 'Tech', 'sport_383.txt': 'Rugby'
}

def main():
    print(f"Loading annotation sample from: {SAMPLE_PATH}")
    try:
        sample_df = pd.read_csv(SAMPLE_PATH)
    except FileNotFoundError:
        print(f"Sample file not found at '{SAMPLE_PATH}'.")
        return

    # Create gold_label column and check for missing articles
    sample_df['gold_label'] = sample_df['unique_id'].map(annotations)
    unannotated_count = sample_df['gold_label'].isnull().sum()
    if unannotated_count > 0:
        print(f"{unannotated_count} samples could not be found in the annotation map and will be dropped.")
        sample_df.dropna(subset=['gold_label'], inplace=True)

    # reorder columns and save
    gold_standard_df = sample_df[['unique_id', 'filename', 'text', 'main_category', 'gold_label']].copy()
    gold_standard_df.to_csv(TEST_SET_PATH, index=False)
    
    print(f"\nCreated the gold standard test set with {len(gold_standard_df)} samples.")
    print(f"File saved to: '{TEST_SET_PATH}'")

if __name__ == "__main__":
    main()