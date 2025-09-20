import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

CONFIG_FILES = [
    Path("ner_config.json"),
    Path("ner_config_wiki.json"),
]
KEYWORDS_FILE = Path("keywords_nonperson.json")

OUTPUT_FILE = Path("entity_roles.json")

def normalize_text(text):
    """Normalize Unicode and strip accents."""
    text = unicodedata.normalize("NFKC", text)
    text = ''.join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != 'Mn')
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': '-', '–': '-', '−': '-', '‐': '-',
        '…': '...', '•': '*', '·': '*',
        '\u00A0': ' ', '\u200B': '', '\u202F': ' '
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.strip()

def to_role_list(value):
    """Ensure roles are stored as a list."""
    if isinstance(value, list):
        return [r.strip() for r in value if isinstance(r, str) and r.strip()]
    if isinstance(value, str):
        return [r.strip() for r in value.split(",") if r.strip()]
    return []

def is_valid_name(name):
    """Filter out qIDs, emoji/symbol names, and non-Latin scripts."""
    name = name.lower()
    if re.fullmatch(r"q\d+", name):
        return False
    if re.search(r"[^\w\s\-']", name):  # symbols, emojis, punctuation
        return False
    if not re.search(r"[a-z]", name):  # must contain Latin letters
        return False
    return True

def main():
    # load keywords and non-person terms
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        kd = json.load(f)
    role_keywords = kd.get("role_keywords", {})
    non_person_terms = [normalize_text(term) for term in kd.get("non_person_terms", [])]

    # merge known_people
    merged = defaultdict(set)

    for file in CONFIG_FILES:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        for raw_name, raw_roles in data.get("known_people", {}).items():
            name = normalize_text(raw_name.lower())
            if not is_valid_name(name):
                continue
            for role in to_role_list(raw_roles):
                merged[name].add(normalize_text(role))

    # cleaned dict
    known_people = {name: sorted(list(roles)) for name, roles in merged.items()}

    final_config = {
        "role_keywords": role_keywords,
        "non_person_terms": non_person_terms,
        "known_people": known_people
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_config, f, ensure_ascii=False, indent=2)

    print(f"Cleaned and saved {len(known_people)} entities to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()
