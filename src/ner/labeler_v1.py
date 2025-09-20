import os
import pandas as pd
import spacy
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


VERSION = 'v1'
SUBCLASS_MODEL = "././models/augmented-classifier"
CONFIG_FILE = Path(__file__).resolve().parent / "entity_roles.json"
REVIEWS_FILE = Path(__file__).resolve().parent / "reviews.json"
RAW_DATA = '././data/output/train_data.csv'
OUTPUT_FILE = f"././data/output/ner/labeled_{VERSION}.csv"

BATCH_SIZE = 16

ROLE_PRIORITY = [
    "BUSINESS-EXECUTIVE-MANAGER", "POLITICIAN", "ATHLETE", "ACTOR-DIRECTOR",
    "MUSICIAN", "AUTHOR-WRITER", "PUBLIC-FIGURE",
]

SUBCAT_TO_ROLE = {
    'Football': 'ATHLETE', 'Rugby': 'ATHLETE', 'Tennis': 'ATHLETE', 'Athletics': 'ATHLETE',
    'Cinema': 'ACTOR-DIRECTOR', 'Music': 'MUSICIAN', 'TV & Radio': 'ACTOR-DIRECTOR',
    'Politics': 'POLITICIAN', 'Economy': 'BUSINESS-EXECUTIVE-MANAGER',
    'Company News': 'BUSINESS-EXECUTIVE-MANAGER', 'Mergers & Acquisitions': 'BUSINESS-EXECUTIVE-MANAGER',
    'Theatre': 'ACTOR-DIRECTOR', 'Stock Market': 'BUSINESS-EXECUTIVE-MANAGER'
}

nlp = spacy.load("en_core_web_trf")

def to_role_list(value):
    if isinstance(value, list): return sorted(list(set(r.strip() for r in value if r and isinstance(r, str))))
    if isinstance(value, str): return sorted(list(set(p.strip() for p in value.split(",") if p.strip())))
    return []

def pick_role(roles):
    order = {r: i for i, r in enumerate(ROLE_PRIORITY)}
    return sorted(roles, key=lambda r: order.get(r, len(order)))[0] if roles else "PUBLIC-FIGURE"

def get_context_role(sentences, role_keywords):
    full_text = " ".join(sentences).lower()
    for role, keywords in role_keywords.items():
        for kw in keywords:
            if kw and re.search(r'\b' + re.escape(kw.lower()) + r'\b', full_text):
                return role
    return None

def label_article(article_text, sub_category, config, reviews):
    known_people = {k.lower(): to_role_list(v) for k, v in config.get('known_people', {}).items()}
    non_person_terms = set(map(str.lower, config.get('non_person_terms', [])))
    role_keywords = config.get('role_keywords', {})

    doc = nlp(article_text)
    
    linked_entities = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            key = ent.text.lower().strip()
            if key not in non_person_terms and len(key) > 2:
                linked_entities[key].append(ent)
    
    final_roles = {}
    for name_key, mentions in linked_entities.items():
        kb_roles = known_people.get(name_key)
        subcat_role = SUBCAT_TO_ROLE.get(sub_category)
        
        mention_sentences = [ent.sent.text for ent in mentions]
        keyword_role = get_context_role(mention_sentences, role_keywords)

        if kb_roles:
            strong_context = subcat_role or keyword_role
            if strong_context and strong_context not in kb_roles:
                final_roles[name_key] = strong_context
                reviews["mismatches"][name_key] = {"kb_roles": kb_roles, "context_role": strong_context, "sentence": mention_sentences[0]}
            elif subcat_role and subcat_role in kb_roles:
                final_roles[name_key] = subcat_role
            elif keyword_role and keyword_role in kb_roles:
                final_roles[name_key] = keyword_role
            else:
                final_roles[name_key] = pick_role(kb_roles)
        else:
            new_name_role = subcat_role or keyword_role or "PUBLIC-FIGURE"
            source = "sub_category" if subcat_role else ("keyword" if keyword_role else "fallback")
            final_roles[name_key] = new_name_role
            reviews["learned_names"][name_key] = {"role": new_name_role, "source": source, "sentence": mention_sentences[0]}

    sentences_with_entities = defaultdict(list)
    for name_key, mentions in linked_entities.items():
        final_role = final_roles.get(name_key)
        if final_role:
            for ent in mentions:
                sentences_with_entities[ent.sent.text.strip()].append({"text": ent.text, "label": final_role})

    return sentences_with_entities

def main():
    print("Starting document-level labeling.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    with open(CONFIG_FILE, encoding='utf-8') as f: config = json.load(f)
    raw_df = pd.read_csv(RAW_DATA).dropna(subset=['text'])

    # predict sub-cats
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
    
    # label entities using document-level context
    reviews = {"learned_names": {}, "mismatches": {}}
    all_labeled_sentences = []

    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Labeling articles"):
        article_text = row['text']
        sub_category = row['sub_category']
        
        labeled_sents_map = label_article(article_text, sub_category, config, reviews)
        
        doc = nlp(article_text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                entities_json = json.dumps(labeled_sents_map.get(sent_text, []))
                all_labeled_sentences.append({
                    "sub_category": sub_category,
                    "sentence": sent_text,
                    "ner_entities": entities_json
                })

    output_df = pd.DataFrame(all_labeled_sentences)
    output_df = output_df[output_df['ner_entities'] != '[]'].reset_index(drop=True)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    if reviews["learned_names"] or reviews["mismatches"]:
        with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(reviews['learned_names'])} learned names and {len(reviews['mismatches'])} mismatches to {REVIEWS_FILE} for review.")

    print(f"Processed {len(raw_df)} articles. Saved {len(output_df)} labeled sentences to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()