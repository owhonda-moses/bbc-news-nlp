import os
import pandas as pd
from transformers import pipeline


MODEL_PATH = './models/ner-model'
DATA_PATH = './data'
SOURCE_DATA_PATH = os.path.join(DATA_PATH, 'val_final.csv')
NUM_TEST_SAMPLES = 10 # no. of articles to test

def aggregate_entities(entities):
    """A robust function to correctly group sub-word tokens."""
    if not entities:
        return []

    aggregated = []
    current_entity_tokens = []
    current_entity_label = None
    current_entity_score = 0.0
    
    for entity in entities:
        label = entity['entity'].replace('B-', '').replace('I-', '')
        
        if entity['entity'].startswith('B-'):
            if current_entity_tokens:
                word = build_word(current_entity_tokens)
                aggregated.append({
                    'word': word, 'entity': current_entity_label,
                    'score': current_entity_score / len(current_entity_tokens)
                })
            
            current_entity_tokens = [entity['word']]
            current_entity_label = label
            current_entity_score = entity['score']
        elif entity['entity'].startswith('I-') and label == current_entity_label:
            current_entity_tokens.append(entity['word'])
            current_entity_score += entity['score']
    
    if current_entity_tokens:
        word = build_word(current_entity_tokens)
        aggregated.append({
            'word': word, 'entity': current_entity_label,
            'score': current_entity_score / len(current_entity_tokens)
        })
        
    return aggregated

def build_word(tokens):
    """Joins sub-word tokens into a clean, single word or phrase."""
    word = ""
    for token in tokens:
        if token.startswith('##'):
            word += token[2:]
        else:
            word += ' ' + token
    return word.strip()

def main():
    print(f"Loading custom NER model from: {MODEL_PATH}")
    ner_pipeline = pipeline("ner", model=MODEL_PATH, tokenizer=MODEL_PATH)
    
    val_df = pd.read_csv(SOURCE_DATA_PATH)
    
    # select a few random samples to test on
    test_samples = val_df.sample(n=NUM_TEST_SAMPLES, random_state=42)
    
    print(f"\n Running inference on {NUM_TEST_SAMPLES} random sentences from test data")
    
    for index, row in test_samples.iterrows():
        # test first ~100 words for simplicity
        text_snippet = " ".join(row['text'].split()[:100])
        print(f"\nTesting Article: \"{row['filename']}\"")
        print(f"  Snippet: \"{text_snippet}...\"")
        print("  Entities Found:")
        
        raw_entities = ner_pipeline(text_snippet)
        aggregated = aggregate_entities(raw_entities)
        
        if not aggregated:
            print("    - None")
            continue
            
        for entity in aggregated:
            if entity['score'] > 0.6: # set threshold
                 print(f"    - Name: \"{entity['word']}\", Role: {entity['entity']}, Score: {entity['score']:.4f}")

if __name__ == "__main__":
    main()