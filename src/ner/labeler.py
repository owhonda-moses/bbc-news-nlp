import csv
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "entity_roles.json" 
REVIEWS_FILE = BASE_DIR / "reviews.json"
DATA_PATH = Path("././data/output/ner")

ROLE_PRIORITY = [
    "BUSINESS-EXECUTIVE-MANAGER", "POLITICIAN", "ATHLETE", "ACTOR-DIRECTOR",
    "MUSICIAN", "AUTHOR-WRITER", "PUBLIC-FIGURE",
]

SUBCAT_TO_ROLE = {
    'Football': 'ATHLETE', 'Rugby': 'ATHLETE', 'Tennis': 'ATHLETE', 'Athletics': 'ATHLETE',
    'Cinema': 'ACTOR-DIRECTOR', 'Music': 'MUSICIAN', 'TV & Radio': 'ACTOR-DIRECTOR',
    'Politics': 'POLITICIAN',
    'Economy': 'BUSINESS-EXECUTIVE-MANAGER',
    'Company News': 'BUSINESS-EXECUTIVE-MANAGER',
    'Mergers & Acquisitions': 'BUSINESS-EXECUTIVE-MANAGER',
    'Theatre': 'ACTOR-DIRECTOR',
    'Stock Market': 'BUSINESS-EXECUTIVE-MANAGER'
}

def reconstruct_name(tokens):
    """Merge a list of tokens into a clean name string."""
    name = ' '.join(tokens).replace(' ##', '').replace('Ġ', '').strip()
    return name

def detect_context_role(sentence_lower, role_keywords):
    """Check if sentence contains keywords for any role."""
    for role, keywords in role_keywords.items():
        for kw in keywords:
            if kw and re.search(r'\b' + re.escape(kw.lower()) + r'\b', sentence_lower):
                return role
    return None

def to_role_list(value):
    """Normalize roles to a list."""
    if isinstance(value, list): return sorted(list(set(r.strip() for r in value if r and isinstance(r, str))))
    if isinstance(value, str): return sorted(list(set(p.strip() for p in value.split(",") if p.strip())))
    return []

def pick_role(roles):
    """Choose a deterministic role using priority."""
    order = {r: i for i, r in enumerate(ROLE_PRIORITY)}
    return sorted(roles, key=lambda r: order.get(r, len(order)))[0] if roles else "PUBLIC-FIGURE"

def classify_entity(entity_text, sentence, sub_category, role_keywords, known_people, non_person_terms, reviews):
    """Classify a PERSON entity into a specific role with the corrected hierarchy."""
    name_clean = reconstruct_name([entity_text]) if not isinstance(entity_text, list) else reconstruct_name(entity_text)
    name_lower = name_clean.lower()
    sentence_lower = sentence.lower()

    if not name_clean or name_lower in non_person_terms: 
        return None, "non_person"
    
    # gather all available signals
    subcat_role = SUBCAT_TO_ROLE.get(sub_category)
    keyword_role = detect_context_role(sentence_lower, role_keywords)
    kb_roles = known_people.get(name_lower)

    # decide based on signals
    if kb_roles:
        # knowledge base entry exists
        strong_context_role = subcat_role or keyword_role

        # context conflicts with KB, trust context and log for review
        if strong_context_role and strong_context_role not in kb_roles:
            reviews["mismatches"][name_lower] = {
                "kb_roles": kb_roles, "context_role": strong_context_role, "sentence": sentence
            }
            return strong_context_role, "context_override"

        # strong context disambiguates a multi-role person in KB
        if subcat_role and subcat_role in kb_roles:
            return subcat_role, "known_people_disambiguated_by_subcat"
        if keyword_role and keyword_role in kb_roles:
            return keyword_role, "known_people_disambiguated_by_keyword"
        
        # no useful context. use highest priority role from KB
        return pick_role(kb_roles), "known_people_priority"
    else:
        # use best available context for new name
        new_name_role = subcat_role or keyword_role
        if new_name_role:
            source = "sub_category" if subcat_role else "keyword"
            reviews["learned_names"][name_lower] = {"role": new_name_role, "source": source, "sentence": sentence}
            return new_name_role, source
            
        # fallback if no signals for a new name
        reviews["learned_names"][name_lower] = {"role": "PUBLIC-FIGURE", "source": "fallback", "sentence": sentence}
        return 'PUBLIC-FIGURE', "fallback"

def process_csv(input_file, output_file):
    with open(CONFIG_FILE, encoding='utf-8') as f: config = json.load(f)
    role_keywords = config.get('role_keywords', {})
    known_people = {k.lower(): to_role_list(v) for k, v in config.get('known_people', {}).items()}
    non_person_terms = set(map(str.lower, config.get('non_person_terms', [])))
    
    reviews = {"learned_names": {}, "mismatches": {}}
    input_df = pd.read_csv(input_file)
    output_data = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Labeling entities"):
        sentence = row['sentence']
        sub_category = row['sub_category']
        try:
            entities = json.loads(row['ner_entities'])
        except (json.JSONDecodeError, TypeError):
            entities = []

        filtered_entities = []
        for ent in entities:
            if ent.get('label') == 'PERSON':
                new_label, source = classify_entity(ent.get('text', ""), sentence, sub_category, role_keywords, known_people, non_person_terms, reviews)
                
                if new_label:
                    # reconstruct name from original tokens
                    if isinstance(ent.get('text'), list):
                        ent['text'] = reconstruct_name(ent.get('text'))
                    ent['label'] = new_label
                    ent['source'] = source
                    filtered_entities.append(ent)
        
        output_data.append({'sub_category': sub_category, 'sentence': sentence, 'ner_entities': json.dumps(filtered_entities)})

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)

    if reviews["learned_names"] or reviews["mismatches"]:
        with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(reviews['learned_names'])} learned names and {len(reviews['mismatches'])} mismatches to {REVIEWS_FILE} for review.")
    
    print(f"Processed {len(input_df)} rows. Output saved to {output_file}")

if __name__ == "__main__":
    input_csv = DATA_PATH / "bootstrap_data.csv"
    output_csv = DATA_PATH / "labeled_data.csv"
    process_csv(input_csv, output_csv)