import os
import pandas as pd
import spacy
import torch
from transformers import pipeline
from tqdm import tqdm


DATA_PATH = './data'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'train_final.csv') 
OUTPUT_NER_FILE = os.path.join(DATA_PATH, 'ner_training_zeroshot.csv')
SPACY_MODEL = 'en_core_web_trf' 
ZERO_SHOT_MODEL = 'facebook/bart-large-mnli'

# candidate labels
CANDIDATE_LABELS = [
    'Politician', 
    'Musician', 
    'Actor or Director',
    'Athlete', 
    'Author or Writer', 
    'Business Executive or Manager'
]
CONFIDENCE_THRESHOLD = 0.70

def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    print(f"Loading spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)
    
    print(f"Loading zero-shot classification pipeline with model: {ZERO_SHOT_MODEL}...")
    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=device)
    
    print("Loading source data...")
    df = pd.read_csv(SOURCE_DATA_PATH)
    
    print("Starting zero-shot NER annotation process")
    ner_data = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Annotating Documents"):
        doc = nlp(row['text'])
        for sent in doc.sents:
            person_entities = [ent for ent in sent.ents if ent.label_ == 'PERSON']
            if not person_entities:
                continue

            # classify the sentence context
            result = classifier(sent.text, CANDIDATE_LABELS, multi_label=False)
            
            if result['scores'][0] > CONFIDENCE_THRESHOLD:
                # sanitize label for IOB format
                job_label = result['labels'][0].upper().replace(' OR ', '-').replace(' ', '-')
                
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

    ner_df = pd.DataFrame(ner_data)
    ner_df.to_csv(OUTPUT_NER_FILE, index=False)
    
    print("\nZero-shot NER data prepared.")
    print(f"Found {len(ner_df)} high-confidence sentences.")
    print(f"New data saved to '{OUTPUT_NER_FILE}'")

if __name__ == "__main__":
    main()