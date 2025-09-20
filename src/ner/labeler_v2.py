import os
import pandas as pd
import spacy
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv


VERSION = 'v2'
SUBCLASS_MODEL = "././models/augmented-classifier"
CONFIG_FILE = Path(__file__).resolve().parent / "entity_roles.json"
PROGRESS_FILE = Path(__file__).resolve().parent / "progress_llm.log"
RAW_DATA = '././data/output/train_data.csv'
OUTPUT_FILE = f"././data/output/ner/labeled_{VERSION}.csv"
BATCH_SIZE = 16

ROLE_PRIORITY = [
    "BUSINESS-EXECUTIVE-MANAGER", "POLITICIAN", "ATHLETE", "ACTOR-DIRECTOR",
    "MUSICIAN", "AUTHOR-WRITER", "PUBLIC-FIGURE",
]

nlp = spacy.load("en_core_web_trf")

def to_role_list(value):
    if isinstance(value, list): return sorted(list(set(r.strip() for r in value if r and isinstance(r, str))))
    if isinstance(value, str): return sorted(list(set(p.strip() for p in value.split(",") if p.strip())))
    return []

def create_batch_prompt(sub_category, linked_entities, known_people):
    valid_labels = ROLE_PRIORITY + ["NOT_A_PERSON"]

    one_shot_example = (
        "--- EXAMPLE ---\n"
        "Sub-Category: Football\n"
        "--- PEOPLE TO CLASSIFY ---\n\n"
        "ENTITY NAME: David Beckham\n"
        "KNOWLEDGE BASE INFO: ['ATHLETE']\n"
        "CONTEXT SENTENCE: \"The club manager, David Beckham, announced his retirement.\"\n\n"
        "ENTITY NAME: BBC Sport\n"
        "KNOWLEDGE BASE INFO: Not in Knowledge Base\n"
        "CONTEXT SENTENCE: \"The news was first reported by BBC Sport.\"\n\n"
        "CORRECT JSON OUTPUT:\n"
        "{\n"
        '  "David Beckham": "ATHLETE",\n'
        '  "BBC Sport": "NOT_A_PERSON"\n'
        "}\n"
    )

    prompt_header = (
        "You are an expert data annotator. Your task is to classify each entity's role based on the evidence provided for a single news article.\n"
        f"You must choose one of the following labels for each entity: {json.dumps(valid_labels)}\n\n"
        "Critically evaluate all evidence. Use your own broad world knowledge as a final check. If the provided 'Sub-Category' or 'Context Sentence' seems to contradict the entity's well-known identity, prioritize its true identity.\n"
        "IMPORTANT: If an entity is clearly not a real person (e.g., a company, place, or concept), you MUST label it as 'NOT_A_PERSON'.\n\n"
        "Respond with ONLY a single, valid JSON object, exactly like the example. Do not include any other text or explanations.\n\n"
        f"{one_shot_example}\n"
        "--- ACTUAL TASK ---\n"
        f"Sub-Category: {sub_category}\n"
        "--- ENTITIES TO CLASSIFY ---"
    )
    
    person_prompts = []
    for name, mentions in linked_entities.items():
        kb_info = known_people.get(name.lower(), "Not in Knowledge Base")
        context_sentence = mentions[0].sent.text
        person_prompt = (
            f"\n\nENTITY NAME: {name}\n"
            f"KNOWLEDGE BASE INFO: {kb_info}\n"
            f"CONTEXT SENTENCE: \"{context_sentence}\""
        )
        person_prompts.append(person_prompt)
        
    return prompt_header + "".join(person_prompts)

def parse_llm_response(response_text):
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from LLM response: {response_text}")
        return {}
    return {}

def load_progress():
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def save_progress(unique_id):
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{unique_id}\n")

def main():
    print("Starting document-level labeling with Gemini.")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env file.")
        return
    genai.configure(api_key=api_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    with open(CONFIG_FILE, encoding='utf-8') as f: config = json.load(f)
    known_people = {k.lower(): to_role_list(v) for k, v in config.get('known_people', {}).items()}
    non_person_terms = set(map(str.lower, config.get('non_person_terms', [])))

    raw_df = pd.read_csv(RAW_DATA).dropna(subset=['text', 'unique_id'])
    
    completed_ids = load_progress()
    df_to_process = raw_df[~raw_df['unique_id'].isin(completed_ids)].copy()
    
    if df_to_process.empty:
        print("All articles have already been processed.")
        return
        
    print(f"Found {len(raw_df)} total articles. {len(completed_ids)} processed. {len(df_to_process)} remaining.")
    
    # predict sub-categories
    print("Loading sub-classification model.")
    sub_class_tokenizer = AutoTokenizer.from_pretrained(SUBCLASS_MODEL)
    sub_class_model = AutoModelForSequenceClassification.from_pretrained(SUBCLASS_MODEL).to(device)
    
    predictions = []
    texts_to_classify = df_to_process['text'].tolist()
    for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE), desc="Classifying articles"):
        batch = texts_to_classify[i:i+BATCH_SIZE]
        inputs = sub_class_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = sub_class_model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend([sub_class_model.config.id2label[id] for id in predicted_class_ids])
        
    df_to_process['sub_category'] = predictions
    
    # label entities using LLM
    llm = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    if os.path.exists(OUTPUT_FILE):
        all_labeled_sentences = pd.read_csv(OUTPUT_FILE).to_dict('records')
    else:
        all_labeled_sentences = []

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Labeling articles with LLM"):
        article_text, sub_category, unique_id = row['text'], row['sub_category'], row['unique_id']
        
        doc = nlp(article_text)
        linked_entities = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                key = ent.text.strip()
                if key.lower() not in non_person_terms and len(key) > 2 and not key.isdigit():
                    linked_entities[key].append(ent)
        
        if not linked_entities:
            save_progress(unique_id)
            continue
            
        prompt = create_batch_prompt(sub_category, linked_entities, known_people)
        
        final_roles = {}
        try:
            response = llm.generate_content(prompt)
            final_roles = parse_llm_response(response.text)
            time.sleep(4.1) # for 15 RPM limit
        except Exception as e:
            print(f"\nAPI error for article {unique_id}: {e}")
            print("Stopping. Re-run to resume.")
            break

        save_progress(unique_id)

        sentences_with_entities = defaultdict(list)
        for name, mentions in linked_entities.items():
            final_role = final_roles.get(name)
            if final_role in ROLE_PRIORITY:
                for ent in mentions:
                    sentences_with_entities[ent.sent.text.strip()].append({"text": ent.text, "label": final_role})

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text and sent_text in sentences_with_entities:
                entities_json = json.dumps(sentences_with_entities[sent_text])
                all_labeled_sentences.append({
                    "sub_category": sub_category,
                    "sentence": sent_text,
                    "ner_entities": entities_json
                })

    output_df = pd.DataFrame(all_labeled_sentences)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nProcessing complete. Total labeled sentences saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()