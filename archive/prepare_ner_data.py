import os
import pandas as pd
import spacy
import re
from tqdm import tqdm


DIR = './data'
DATA_PATH = os.path.join(DIR, 'train_data.csv') 
OUTPUT_NER_FILE = os.path.join(DIR, 'ner_training_data.csv')
SPACY_MODEL = 'en_core_web_trf' 


JOB_KEYWORDS = {
    'POLITICIAN': ['government', 'minister', 'mp', 'election', 'party', 'prime minister', 'chancellor', 'political'],
    'MUSICIAN': ['singer', 'musician', 'band', 'song', 'album', 'concert', 'grammy', 'artist', 'rock'],
    'TV-FILM-PERSONALITY': ['actor', 'actress', 'film', 'movie', 'director', 'tv', 'host', 'oscar', 'hollywood', 'bbc']
}

def get_job_label(sentence_text, keywords_map):
    """Checks for keywords in a sentence and returns the corresponding job label."""
    lower_text = sentence_text.lower()
    for job, keywords in keywords_map.items():
        if any(re.search(r'\b' + re.escape(key) + r'\b', lower_text) for key in keywords):
            return job
    return None


def main():
    print(f"Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Spacy model '{SPACY_MODEL}' not found.")
        return
    
    print("Loading source data")
    df = pd.read_csv(DATA_PATH)
    
    print("Starting NER annotation")
    ner_data = []
    
    # process each document
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing documents"):
        doc = nlp(row['text'])
        for sent in doc.sents:
            if len(sent) < 3:
                continue
            person_entities = [ent for ent in sent.ents if ent.label_ == 'PERSON']
            if not person_entities:
                continue
            job_label = get_job_label(sent.text, JOB_KEYWORDS)
            if not job_label:
                continue

            tokens = [token.text for token in sent]
            tags = ['O'] * len(sent)
            
            for ent in person_entities:
                tags[ent.start - sent.start] = f"B-{job_label}"
                for i in range(ent.start + 1, ent.end):
                    tags[i - sent.start] = f"I-{job_label}"
            
            ner_data.append({
                'sentence_id': f"{row['filename']}_{sent.start}",
                'tokens': " ".join(tokens),
                'ner_tags': " ".join(tags)
            })

    # create and save final DataFrame
    ner_df = pd.DataFrame(ner_data)
    ner_df.to_csv(OUTPUT_NER_FILE, index=False)
    
    print("\nNER data preparation complete.")
    print(f"Found {len(ner_df)} sentences with labeled entities.")
    print(f"Data saved to '{OUTPUT_NER_FILE}'")
    

if __name__ == "__main__":
    main()