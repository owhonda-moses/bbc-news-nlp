import os
import pandas as pd
import spacy
import torch
from transformers import pipeline
from tqdm import tqdm
import json

# paths
DATA_PATH = '././data/output/ner'
SOURCE_FILE = f"{DATA_PATH}/train_data.csv"
OUTPUT_FILE = f"{DATA_PATH}/ner_train.csv" 
SPACY_MODEL = 'en_core_web_trf'
ZS_MODEL = 'facebook/bart-large-mnli'

CANDIDATE_LABELS = [
    'Politician', 'Musician', 'Actor or Director', 'Athlete',
    'Author or Writer', 'Business Executive or Manager', 'Public Figure'
]
CONFIDENCE_THRESHOLD = 0.70
BATCH_SIZE = 32

def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    nlp = spacy.load(SPACY_MODEL)
    classifier = pipeline("zero-shot-classification", model=ZS_MODEL, device=device)
    df = pd.read_csv(SOURCE_FILE)
    
    print("Extracting sentences.")
    sentences_to_process = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scanning"):
        doc = nlp(row['text'])
        for sent in doc.sents:
            if any(ent.label_ == 'PERSON' for ent in sent.ents):
                sentences_to_process.append({'id': row['unique_id'], 'sent': sent})

    print(f"Found {len(sentences_to_process)} sentences to classify.")
    
    print("Classifying sentences in batch.")
    text_generator = (item['sent'].text for item in sentences_to_process)
    all_results = [res for res in tqdm(classifier(text_generator, CANDIDATE_LABELS, batch_size=BATCH_SIZE), total=len(sentences_to_process), desc="Classifying")]

    print("Generating NER data with JSON entities.")
    ner_data = []
    for item, result in zip(sentences_to_process, all_results):
        if result['scores'][0] > CONFIDENCE_THRESHOLD:
            sent = item['sent']
            entities = []
            job_label = result['labels'][0].upper().replace(' OR ', '-').replace(' ', '-')
            
            person_entities = [ent for ent in sent.ents if ent.label_ == 'PERSON']
            for ent in person_entities:
                entities.append({'text': ent.text, 'label': job_label})
            
            ner_data.append({
                'sentence_id': f"{item['id']}_{sent.start}",
                'sentence': sent.text,
                'ner_entities': json.dumps(entities)
            })

    ner_df = pd.DataFrame(ner_data)
    ner_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nFound {len(ner_df)} high-confidence sentences.")
    print(f"`data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()