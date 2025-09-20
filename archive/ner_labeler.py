import csv
import json
import re
from pathlib import Path
import pandas as pd

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
    """Classify a PERSON entity into a specific role with upgraded logic."""
    name_clean = reconstruct_name([entity_text]) if not isinstance(entity_text, list) else reconstruct_name(entity_text)
    name_lower = name_clean.lower()
    sentence_lower = sentence.lower()

    if name_lower in non_person_terms: return None, "non_person"

    if name_lower in known_people:
        kb_roles = known_people[name_lower]
        context_role = detect_context_role(sentence_lower, role_keywords)
        
        # if context clarifies a known multi-role person, use it
        if context_role and context_role in kb_roles:
            return context_role, "known_people_disambiguated"
        
        # use highest-priority known role
        return pick_role(kb_roles), "known_people"

    # context for unknown names
    subcat_role = SUBCAT_TO_ROLE.get(sub_category)
    if subcat_role:
        reviews["learned_names"][name_lower] = {"role": subcat_role, "source": "sub_category", "sentence": sentence}
        return subcat_role, "sub_category"

    # keyword search for unknown names
    keyword_role = detect_context_role(sentence_lower, role_keywords)
    if keyword_role:
        reviews["learned_names"][name_lower] = {"role": keyword_role, "source": "keyword", "sentence": sentence}
        return keyword_role, "keyword"
        
    # fallback
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
        entities = json.loads(row['ner_entities'])

        for ent in entities:
            if ent.get('label') == 'PERSON':
                new_label, source = classify_entity(ent.get('text', ""), sentence, sub_category, role_keywords, known_people, non_person_terms, reviews)
                
                if new_label:
                    ent['label'] = new_label
                    ent['source'] = source
                else:
                    ent['label'] = 'O'
                    ent['source'] = "filtered_non_person"
        
        output_data.append({'sentence': sentence, 'ner_entities': json.dumps(entities)})

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