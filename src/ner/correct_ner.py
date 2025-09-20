import os
import pandas as pd
import sys
import json
import argparse
import time

# paths
DATA_PATH = '././data/output/ner'
TARGET_FILE = f"{DATA_PATH}/labeled_llm.csv"

# mapping for quick-fix
KEYWORD_TO_LABEL = {
    'coach': 'ATHLETE',
    'manager': 'ATHLETE',
    'player': 'ATHLETE',
    'boss': 'ATHLETE',
    'striker': 'ATHLETE',
    'author': 'AUTHOR-WRITER',
    'novelist': 'AUTHOR-WRITER',
    'journalist': 'AUTHOR-WRITER',
    'editor': 'AUTHOR-WRITER',
    'correspondent': 'AUTHOR-WRITER',
    'director': 'ACTOR-DIRECTOR',
    'actor': 'ACTOR-DIRECTOR',
    'actress': 'ACTOR-DIRECTOR',
    'singer': 'MUSICIAN',
    'musician': 'MUSICIAN',
    'ceo': 'BUSINESS-EXECUTIVE-MANAGER',
    'chairman': 'BUSINESS-EXECUTIVE-MANAGER',
    'executive': 'BUSINESS-EXECUTIVE-MANAGER',
    'minister': 'POLITICIAN',
    'chancellor': 'POLITICIAN',
    'lawyer': 'PUBLIC-FIGURE',
    'analyst': 'PUBLIC-FIGURE'
}

# colors
class colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_sentence(idx, total, sentence, entities):
    print(f"{colors.HEADER}Reviewing sentence {idx + 1} of {total}{colors.ENDC}")
    print("-" * 30)
    print(f"{colors.BOLD}Sentence:{colors.ENDC}\n{sentence}\n")
    print(f"{colors.BOLD}Entities:{colors.ENDC}")
    if not entities:
        print("  - None")
    else:
        for i, entity in enumerate(entities):
            print(f"  [{i}] '{entity['text']}' -> {colors.OKGREEN}{entity['label']}{colors.ENDC}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Interactively correct labels.")
    parser.add_argument(
        '--filter', 
        type=str, 
        default=None, 
        help="Keyword to filter sentences for review."
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(TARGET_FILE)
        df['ner_entities'] = df['ner_entities'].astype(str)
    except FileNotFoundError:
        print(f"File not found.")
        return

    df_to_process = df.copy()
    target_label = None

    if args.filter:
        keyword = args.filter.lower()
        target_label = KEYWORD_TO_LABEL.get(keyword)
        print(f"Filtering sentences with keyword: '{keyword}'")
        df_to_process = df[df['sentence'].str.contains(keyword, case=False, na=False)].copy()
        if df_to_process.empty:
            print("No sentences found with that keyword.")
            return

    print(f"Loaded {len(df_to_process)} sentences to correct.")
    
    corrected_entities_list = [json.loads(ents) for ents in df_to_process['ner_entities']]
    
    i = 0
    while i < len(df_to_process):
        row_index = df_to_process.index[i]
        sentence = df_to_process.iloc[i]['sentence']
        entities = corrected_entities_list[i]
        
        clear_screen()
        display_sentence(i, len(df_to_process), sentence, entities)
        
        # dynamic prompt for quick-fix mode
        if target_label:
            prompt_text = f"Enter index to correct to '{colors.BOLD}{target_label}{colors.ENDC}', or other action: "
        else:
            prompt_text = f"Action: {colors.BOLD}[Enter] accept{colors.ENDC}, (e)dit, (a)dd, (d)elete, (b)ack, (q)uit: "
        
        action = input(prompt_text).lower()

        if target_label and action.isdigit():
            try:
                idx = int(action)
                if 0 <= idx < len(entities):
                    entities[idx]['label'] = target_label
                    print(f"Corrected entity {idx} to '{target_label}'.")
                    time.sleep(0.5) # pause to show confirmation
                    # i += 1 # auto-accept and next
                    continue
                else:
                    print("Invalid index.")
                    time.sleep(1)
                    continue
            except ValueError:
                pass # fall through to other actions

        if action in ['', 'y', 's']:
            i += 1
            continue
        elif action == 'd':
            df_to_process.drop(row_index, inplace=True)
            corrected_entities_list.pop(i)
            continue
        elif action == 'b':
            i = max(0, i - 1)
            continue
        elif action == 'q':
            break
        elif action == 'a':
            text = input("  Entity text to add: ").strip()
            label = input(f"  Label for '{text}': ").strip().upper()
            if text and label:
                entities.append({'text': text, 'label': label})
        elif action == 'e':
            while True:
                clear_screen()
                display_sentence(i, len(df_to_process), sentence, entities)
                print(f"{colors.WARNING}EDIT MODE{colors.ENDC} | Press Enter to finish.")
                try:
                    cmd = input("Command ('index action [value]'): ").strip()
                    if not cmd: break
                    parts = cmd.split()
                    idx = int(parts[0])
                    if not (0 <= idx < len(entities)):
                        print("Invalid index."); continue
                    if parts[1].lower() == 'delete':
                        entities.pop(idx)
                    elif parts[1].lower() == 'label' and len(parts) == 3:
                        entities[idx]['label'] = parts[2].upper()
                    else:
                        print("Invalid command.")
                except (ValueError, IndexError):
                    print("Invalid command format."); input("Press Enter...")
        
        if i > 0 and i % 10 == 0:
            df.loc[df_to_process.index, 'ner_entities'] = [json.dumps(e) for e in corrected_entities_list]
            df.to_csv(TARGET_FILE, index=False)
            print(f"{colors.OKBLUE}Progress saved!{colors.ENDC}")

    print("\nSaving corrections...")
    df.loc[df_to_process.index, 'ner_entities'] = [json.dumps(e) for e in corrected_entities_list]
    df.to_csv(TARGET_FILE, index=False)
    print(f"Corrected file saved to '{TARGET_FILE}'.")

if __name__ == "__main__":
    main()