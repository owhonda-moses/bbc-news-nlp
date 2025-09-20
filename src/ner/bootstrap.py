import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import json
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


NER_MODEL = "Jean-Baptiste/roberta-large-ner-english"
SUBCLASS_MODEL = "././models/augmented-classifier"
RAW_DATA = '././data/output/train_data.csv'
OUTPUT_FILE = '././data/output/ner/bootstrap_data.csv'
BATCH_SIZE = 32

nlp = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
nlp.add_pipe("sentencizer")

def main():
    """
    Uses a sub-classification model and a general NER model to create a 
    pre-labeled seed set for manual correction.
    """
    print(f"Starting bootstrap using {NER_MODEL}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    if not all(os.path.exists(p) for p in [RAW_DATA, SUBCLASS_MODEL]):
        print(f"Ensure raw data and subclass model exist.")
        return
        
    raw_df = pd.read_csv(RAW_DATA).dropna(subset=['text'])
    print(f"Processing all {len(raw_df)} articles.")

    # predict sub-categories
    print("Loading subclass model.")
    sub_class_tokenizer = AutoTokenizer.from_pretrained(SUBCLASS_MODEL)
    sub_class_model = AutoModelForSequenceClassification.from_pretrained(SUBCLASS_MODEL).to(device)
    
    predictions = []
    texts_to_classify = raw_df['text'].tolist()
    for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE), desc="Classifying articles"):
        batch = texts_to_classify[i:i+BATCH_SIZE]
        inputs = sub_class_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = sub_class_model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend([sub_class_model.config.id2label[id] for id in predicted_class_ids])
        
    raw_df['sub_category'] = predictions
    
    # segment into sentences
    all_sentences_with_context = []
    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Segmenting articles"):
        doc = nlp(row['text'])
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                all_sentences_with_context.append({'text': sent_text, 'sub_category': row['sub_category']})

    print(f"Found {len(all_sentences_with_context)} sentences to process.")

    # pre-label entities
    print("Loading NER model.")
    ner_pipeline = pipeline("ner", model=NER_MODEL, tokenizer=NER_MODEL, aggregation_strategy="simple", device=device)

    pre_labeled_data = []
    sentence_texts = [item['text'] for item in all_sentences_with_context]

    for i, ner_results in enumerate(tqdm(ner_pipeline(sentence_texts, batch_size=BATCH_SIZE), total=len(sentence_texts), desc="Pre-labeling sentences")):
        person_entities = []
        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                entity_word = entity['word']
                if entity_word.startswith('##') or entity_word.startswith('Ġ'):
                    continue
                if len(entity_word.strip()) <= 1:
                    continue
                
                person_entities.append({
                    "text": entity_word,
                    "label": "PERSON" 
                })
        
        if person_entities:
            pre_labeled_data.append({
                "sub_category": all_sentences_with_context[i]['sub_category'],
                "sentence": all_sentences_with_context[i]['text'],
                "ner_entities": json.dumps(person_entities)
            })

    # save for review
    if pre_labeled_data:
        review_df = pd.DataFrame(pre_labeled_data)
        review_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(review_df)} pre-labeled sentences to '{OUTPUT_FILE}'")
    else:
        print("\nNo sentences with PERSON entities were found.")

if __name__ == "__main__":
    main()