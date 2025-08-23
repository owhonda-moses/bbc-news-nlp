import os
from transformers import pipeline


MODEL_PATH = './models/ner-model'

# test sentences
test_sentences = [
    "The bill was passed after Prime Minister Gordon Brown gave a compelling speech.",
    "The film was directed by Quentin Tarantino, who is known for his unique style.",
    "Led Zeppelin's new album features the legendary singer Robert Plant on vocals.",
    "Chancellor of the Exchequer, Rishi Sunak, announced the new budget.",
    "The new series starring actress Nicole Kidman is set to be a huge hit.",
    "Musician and songwriter Bob Dylan has won the Nobel Prize in Literature."
]

def aggregate_entities(entities):
    """A final, robust function to correctly group sub-word tokens."""
    if not entities:
        return []

    aggregated = []
    current_entity_tokens = []
    current_entity_label = None
    current_entity_score = 0.0
    
    for entity in entities:
        label = entity['entity'].replace('B-', '').replace('I-', '')
        
        # if a new entity begins
        if entity['entity'].startswith('B-'):
            # save the previous entity if it exists
            if current_entity_tokens:
                word = build_word(current_entity_tokens)
                aggregated.append({
                    'word': word, 'entity': current_entity_label,
                    'score': current_entity_score / len(current_entity_tokens)
                })
            
            # start a new one
            current_entity_tokens = [entity['word']]
            current_entity_label = label
            current_entity_score = entity['score']
        # if the entity continues
        elif entity['entity'].startswith('I-') and label == current_entity_label:
            current_entity_tokens.append(entity['word'])
            current_entity_score += entity['score']
    
    # add the last entity
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
    
    print("\n Running Inference on Test Sentences")
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nSentence {i+1}: \"{sentence}\"")
        print("  Entities Found:")
        
        raw_entities = ner_pipeline(sentence)
        aggregated = aggregate_entities(raw_entities)
        
        if not aggregated:
            print("    - None")
            continue
            
        for entity in aggregated:
            # filter out low-confidence results
            if entity['score'] > 0.6 and len(entity['word']) > 2:
                 print(f"    - Name: \"{entity['word']}\", Role: {entity['entity']}, Score: {entity['score']:.4f}")

if __name__ == "__main__":
    main()