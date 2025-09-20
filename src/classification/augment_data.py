import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

DATA_PATH = '././data/output/sub_class'
MODEL_NAME = 't5-base'
BATCH_SIZE = 4
MIN_AUGMENTATION = 1
MAX_AUGMENTATION = 5


DEFAULT_PROMPT = "paraphrase: {text}" # default prompt for general topics

# prompts for topics that need careful handling
CONTEXT_PROMPTS = {
    'Company News': "paraphrase this business report about a company's performance: {text}",
    'Economy': "paraphrase this financial news article about the economy: {text}",
    'Stock Market': "paraphrase this stock market report: {text}",
    'Mergers & Acquisitions': "paraphrase this article about a corporate merger or acquisition: {text}"
}

# default generation settings for creative topics
DEFAULT_GEN_PARAMS = {
    "max_length": 1200,
    "num_return_sequences": 1,
    "num_beams": 5,
    "early_stopping": True,
    "temperature": 0.9,
    "top_p": 0.95,
    "do_sample": True
}

# technical topics to preserve meaning
STRICT_GEN_PARAMS = {
    "max_length": 1200,
    "num_return_sequences": 1,
    "num_beams": 5,
    "early_stopping": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}


# define targets
AUGMENTATION_TARGETS = {
    "zeroshot": {
        "source_path": "././data/output/sub_class/train_zeroshot.csv",
        "output_path": "././data/output/sub_class/zeroshot_augmented.csv",
        "target_classes": {
            'Company News': {'test_samples': 13, 'min_target': 180},
            'Economy': {'test_samples': 7, 'min_target': 100},
            'TV & Radio': {'test_samples': 5, 'min_target': 80},
            'Music': {'test_samples': 7, 'min_target': 100},
            'Cinema': {'test_samples': 7, 'min_target': 100},
            'Politics': {'test_samples': 21, 'min_target': 250},
            'Tech': {'test_samples': 19, 'min_target': 230}
        }
    }
}

def batch_augment_text(texts, model, tokenizer, device, prompt_template, gen_params, num_versions=3):
    """Generates paraphrased versions for a batch of texts using specific settings."""
    input_prompts = [prompt_template.format(text=text) for text in texts]
    inputs = tokenizer(input_prompts, return_tensors='pt', padding=True, max_length=1024, truncation=True).to(device)
    
    # update generation params
    gen_params["num_return_sequences"] = num_versions

    outputs = model.generate(**inputs, **gen_params)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    paraphrased_texts = []
    for i in range(len(texts)):
        start_index = i * num_versions
        end_index = start_index + num_versions
        paraphrased_texts.append(decoded_outputs[start_index:end_index])
    return paraphrased_texts

def calculate_smart_augmentation(df, target_info):
    """Calculate augmentation based on current count and target."""
    augmentation_plan = {}
    class_counts = df['target_label'].value_counts()
    
    for label, info in target_info.items():
        if label not in class_counts.index:
            print(f"{label} not found in training data")
            continue
            
        current_count = class_counts[label]
        min_target = info['min_target']
        test_samples = info['test_samples']
        
        weight = test_samples / 111
        adjusted_target = int(min_target * (1 + weight))
        
        if current_count < adjusted_target:
            samples_needed = adjusted_target - current_count
            base_factor = samples_needed // current_count + 1
            augmentation_factor = min(max(base_factor, MIN_AUGMENTATION), MAX_AUGMENTATION)
        else:
            augmentation_factor = 0
        
        augmentation_plan[label] = {
            'factor': augmentation_factor,
            'current': current_count,
            'target': adjusted_target
        }
    
    return augmentation_plan

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {MODEL_NAME} model")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    for mode, config in AUGMENTATION_TARGETS.items():
        print(f"Processing {mode} data")      
        train_df = pd.read_csv(config['source_path'])
        
        augmentation_plan = calculate_smart_augmentation(train_df, config['target_classes'])
        
        print("\nAugmentation plan:")
        for label, details in augmentation_plan.items():
            if details['factor'] > 0:
                print(f"  {label}: {details['current']} samples -> "
                      f"augment {details['factor']}x -> target ~ {details['target']} samples")
            else:
                print(f"  {label}: {details['current']} samples (sufficient, no augmentation)")
        
        augmented_rows = []
        
        for label, details in augmentation_plan.items():
            if details['factor'] == 0:
                continue
                
            target_df = train_df[train_df['target_label'] == label].copy()
            
            if len(target_df) == 0:
                continue
            
            target_texts = target_df['text'].tolist()
            aug_factor = details['factor']
            
            # select the appropriate prompt and generation parameters
            prompt = CONTEXT_PROMPTS.get(label, DEFAULT_PROMPT)
            gen_params = STRICT_GEN_PARAMS if label in CONTEXT_PROMPTS else DEFAULT_GEN_PARAMS
            
            print(f"  Using {'strict' if label in CONTEXT_PROMPTS else 'default'} settings for {label}")
            
            for i in tqdm(range(0, len(target_texts), BATCH_SIZE), 
                          desc=f"Augmenting {label} ({aug_factor}x)"):
                batch_texts = target_texts[i:i+BATCH_SIZE]
                batch_rows = target_df.iloc[i:i+BATCH_SIZE]
                
                generated_versions = batch_augment_text(
                    batch_texts, model, tokenizer, device, prompt, gen_params, aug_factor
                )
                
                for j, (index, original_row) in enumerate(batch_rows.iterrows()):
                    new_texts = generated_versions[j]
                    for k, new_text in enumerate(new_texts):
                        new_row = original_row.copy()
                        new_row['text'] = new_text
                        new_row['unique_id'] = f"aug_{k+1}_{original_row['unique_id']}"
                        augmented_rows.append(new_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            final_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
        else:
            final_train_df = train_df
            print("No augmentation performed")
        
        final_train_df.to_csv(config['output_path'], index=False)
        print(f"\nSaved to '{config['output_path']}'")
        
        print("\nFinal class distribution:")
        final_counts = final_train_df['target_label'].value_counts()
        for label in config['target_classes'].keys():
            if label in final_counts.index:
                print(f"  {label}: {final_counts[label]} samples")

    print("Augmentation complete.")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()