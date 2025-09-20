import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
import json
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

MODEL = 'distilbert' 

# define weak classes and keywords
TARGET_CLASSES = {
    'AUTHOR-WRITER': [
        'author', 'writer', 'novelist', 'journalist', 'correspondent', 
        'professor', 'editor', 'columnist', 'presenter', 'critic'
    ],
    'PUBLIC-FIGURE': [
        'analyst', 'expert', 'activist', 
        'pundit', 'lawyer', 'prosecutor', 'attorney', 'rabbi', 'bishop'
    ],
    'BUSINESS-EXECUTIVE-MANAGER': [
        'ceo', 'chief executive', 'manager', 'chairman', 
        'vice president', 'founder', 'publisher', 'tycoon'
    ]
}

# sentences to find for each class
NUM_PER_CLASS = 100

# paths
DATA_PATH = '././data/output/ner'
MODELS_PATH = '././models'
MODEL_PATH = f"{MODELS_PATH}/ner-{MODEL}-model"
RAW_DATA = '././data/output/train_data.csv' 
TRAIN_DATA = f"{DATA_PATH}/ner_train.csv"
OUTPUT_FILE = f"{DATA_PATH}/added_data.csv"

nlp = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
nlp.add_pipe("sentencizer")

def iob_to_json(tokens, tags):
    """Converts a list of tokens and IOB tags back to the JSON entity format."""
    entities = []
    current_entity_text, current_entity_label = "", ""
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_entity_text:
                entities.append({"text": current_entity_text.strip(), "label": current_entity_label})
            current_entity_text = token
            current_entity_label = tag.split('-')[1]
        elif tag.startswith('I-'):
            current_entity_text += " " + token
        else:
            if current_entity_text:
                entities.append({"text": current_entity_text.strip(), "label": current_entity_label})
                current_entity_text, current_entity_label = "", ""
    if current_entity_text:
        entities.append({"text": current_entity_text.strip(), "label": current_entity_label})
    
    cleaned_entities = []
    for ent in entities:
        cleaned_text = re.sub(r'\s+([#])\1\s*', '', ent['text']).replace('Ġ', '')
        cleaned_entities.append({'text': cleaned_text, 'label': ent['label']})
    return json.dumps(cleaned_entities)

def main():
    print("Starting active learning to find targeted data.")

    if not os.path.exists(TRAIN_DATA) or not os.path.exists(RAW_DATA) or not os.path.exists(MODEL_PATH):
        print("Ensure ner_train.csv, train_data.csv, and a trained model exist.")
        return

    current_train_df = pd.read_csv(TRAIN_DATA)
    seen_sentences = set(current_train_df['sentence'].tolist())
    raw_df = pd.read_csv(RAW_DATA).dropna(subset=['text'])

    print(f"Loading model: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    id2tag = model.config.id2label

    all_keywords = [kw for sublist in TARGET_CLASSES.values() for kw in sublist]
    regex_pattern = r'\b(' + '|'.join(all_keywords) + r')\b'
    
    relevant_df = raw_df[raw_df['text'].str.contains(regex_pattern, case=False, na=False)]
    print(f"Found {len(relevant_df)} potentially relevant articles out of {len(raw_df)}.")

    pre_labeled_data = []
    with torch.no_grad():
        pbar = tqdm(total=len(TARGET_CLASSES) * NUM_PER_CLASS, desc="Finding new sentences")
        
        # iterate through articles
        for _, row in relevant_df.iterrows():
            if all(sum(1 for d in pre_labeled_data if d['target_class'] == label) >= NUM_PER_CLASS for label in TARGET_CLASSES):
                break

            doc = nlp(row['text'])
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text in seen_sentences: continue

                for target_label, keywords in TARGET_CLASSES.items():
                    # Check if this class is already full
                    if sum(1 for d in pre_labeled_data if d['target_class'] == target_label) >= NUM_PER_CLASS:
                        continue

                    if any(re.search(r'\b' + kw + r'\b', sent_text, re.IGNORECASE) for kw in keywords):
                        inputs = tokenizer(sent_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                        outputs = model(**inputs)
                        predictions = torch.argmax(outputs.logits, dim=2)
                        
                        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0], skip_special_tokens=True)
                        tags = [id2tag.get(p.item(), 'O') for p in predictions[0]][1:-1]

                        generated_json = iob_to_json(tokens, tags)
                        
                        pre_labeled_data.append({
                            'target_class': target_label, # add for context
                            'sentence': sent_text,
                            'ner_entities': generated_json
                        })
                        seen_sentences.add(sent_text)
                        pbar.update(1)
                        break
    
    pbar.close()
    
    if pre_labeled_data:
        review_df = pd.DataFrame(pre_labeled_data)
        final_df = review_df.drop(columns=['target_class']) # remove helper column
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nSaved {len(final_df)} new, pre-labeled sentences to '{OUTPUT_FILE}'")
        for label in TARGET_CLASSES:
            count = len([d for d in pre_labeled_data if d['target_class'] == label])
            print(f"  - Found {count} sentences for {label}")
    else:
        print("\nNo new sentences found matching the criteria.")

if __name__ == "__main__":
    main()