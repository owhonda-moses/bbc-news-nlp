import os
import pandas as pd
from transformers import DistilBertTokenizerFast, RobertaTokenizerFast
import json
import spacy
import re
from tqdm import tqdm
import warnings
import random
from collections import defaultdict, Counter
import numpy as np

warnings.filterwarnings("ignore")

MODEL = 'roberta'
# MODEL = 'distilbert'

# augmentation settings
MIN_TARGET = 2000
MAX_TARGET = 2000
RATIO_CAP = 2.0
MAX_CLONES = 4

# context augmentation
MAX_AUG_PER_SENT = 2
AUG_POS_RATE = 0.5
POS_FRAMES = [["was", "interviewed"], ["spoke", "at"], ["appeared", "on"], ["joined"], ["announced"]]
NEG_FRAMES = [["street"], ["valley"], ["station"], ["school"], ["stadium"], ["boulevard"], ["pub"]]

SEED = 42

VERSION = 'v2'
DATA_PATH = '././data/output/ner'
INPUTS = {
    'train': f"{DATA_PATH}/labeled_{VERSION}.csv",
    'val': f"{DATA_PATH}/ner_val.csv",
    'test': f"{DATA_PATH}/ner_test.csv"
}
OUTPUTS = {
    'train': f"{DATA_PATH}/train_{MODEL}_{VERSION}.csv",
    'val': f"{DATA_PATH}/val_{MODEL}.csv",
    'test': f"{DATA_PATH}/test_{MODEL}.csv"
}

AUGMENT_DATA = True if VERSION == 'v2' else False

CONFIGS = {
    'distilbert': {'tokenizer': DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')},
    'roberta': {'tokenizer': RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)}
}
config = CONFIGS[MODEL]
tokenizer = config['tokenizer']

nlp = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
nlp.add_pipe("sentencizer")

random.seed(SEED)
np.random.seed(SEED)

# utilities 
def parse_entities(ner_json_str):
    try:
        ents = json.loads(ner_json_str)
        if isinstance(ents, list):
            return [e for e in ents if isinstance(e, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
    return []

def compute_entity_and_row_counts(df):
    entity_counts = defaultdict(int)
    row_counts = defaultdict(int)
    for _, row in df.iterrows():
        ents = parse_entities(row.get('ner_entities'))
        labels_in_row = set()
        for e in ents:
            lbl = e.get('label')
            if not lbl:
                continue
            entity_counts[lbl] += 1
            labels_in_row.add(lbl)
        for lbl in labels_in_row:
            row_counts[lbl] += 1
    return dict(entity_counts), dict(row_counts)

def pretty_print_counts(title, entity_counts, row_counts):
    print(title)
    labels = sorted(set(list(entity_counts.keys()) + list(row_counts.keys())))
    print(f"{'Label':35} {'Entities':>10} {'Rows':>10}")
    for lbl in labels:
        print(f"{lbl:35} {entity_counts.get(lbl,0):10} {row_counts.get(lbl,0):10}")

def safe_single_swap(sentence, original_text, replacement_text):
    pattern = re.compile(rf'\b{re.escape(original_text)}\b')
    matches = list(pattern.finditer(sentence))
    if len(matches) != 1:
        return None
    start, end = matches[0].span()
    return sentence[:start] + replacement_text + sentence[end:]

# augmentation
def context_augment_row(tokens_text, ents):
    aug = []
    toks = tokens_text.split()
    if not ents or MAX_AUG_PER_SENT <= 0:
        return aug
    # pick first entity span text if available
    ent_texts = [e.get("text") for e in ents if e.get("text")]
    if not ent_texts:
        return aug
    surface = ent_texts[0]
    for _ in range(MAX_AUG_PER_SENT):
        if random.random() < AUG_POS_RATE:
            frame = random.choice(POS_FRAMES)
            # insert near start to preserve tokenization stability
            new_toks = frame + toks
            aug.append((" ".join(new_toks), ents))
        else:
            if " " not in surface:  # negative only for single-token names
                frame = random.choice(NEG_FRAMES)
                new_toks = [surface] + frame + toks
                # negative: mark all O
                aug.append((" ".join(new_toks), []))
    return aug

def augment_data(df):
    print("\nAugmenting data.")
    if df.empty:
        return df

    print("\nDe-duplicating original data.")
    orig_size = len(df)
    df = df.drop_duplicates(subset=['sentence'], keep='first').reset_index(drop=True)
    print(f"Removed {orig_size - len(df)} duplicate sentences from originals. Originals now: {len(df)}")

    base_entity_counts, base_row_counts = compute_entity_and_row_counts(df)
    pretty_print_counts("\nOriginal entity and row counts:", base_entity_counts, base_row_counts)
    if not base_row_counts:
        print("No row counts found; skipping augmentation.")
        return df

    label_to_texts = defaultdict(set)
    for _, row in df.iterrows():
        for ent in parse_entities(row.get('ner_entities')):
            lbl = ent.get('label'); txt = ent.get('text')
            if lbl and txt:
                label_to_texts[lbl].add(txt)
    label_to_texts = {lbl: sorted(list(txts)) for lbl, txts in label_to_texts.items()}

    plans = {}
    for lbl, R in base_row_counts.items():
        gaz = label_to_texts.get(lbl, [])
        if len(gaz) < 2:
            continue
        if R >= MAX_TARGET:
            T = R
        else:
            max_total_by_ratio = R + int(np.floor(R * RATIO_CAP))
            band_target = max(MIN_TARGET, min(MAX_TARGET, max(R, MIN_TARGET)))
            T = min(max_total_by_ratio, band_target)
        synth_needed = max(0, T - R)
        if synth_needed > 0:
            plans[lbl] = {'original_rows': R, 'target_total_rows': T, 'synth_needed': synth_needed}

    if not plans:
        print("\nNo classes require augmentation under current settings. Skipping augmentation.")
        return df

    print("\nAugmentation plan:")
    for lbl, p in sorted(plans.items()):
        print(f"  - {lbl}: originals={p['original_rows']}, target={p['target_total_rows']}, synth_needed={p['synth_needed']} (gaz={len(label_to_texts.get(lbl, []))})")

    label_to_rows = defaultdict(list)
    for idx, row in df.iterrows():
        ents = parse_entities(row.get('ner_entities'))
        row_labels = {e.get('label') for e in ents if e.get('label')}
        for lbl in row_labels:
            label_to_rows[lbl].append((idx, row))

    source_clone_counter = defaultdict(lambda: defaultdict(int))
    augmented_rows = []
    seen_sentences = set(df['sentence'].tolist())

    rng = np.random.RandomState(SEED)

    for lbl, plan in plans.items():
        synth_needed = plan['synth_needed']
        gazetteer = label_to_texts.get(lbl, [])
        if len(gazetteer) < 2:
            continue
        candidates = label_to_rows.get(lbl, [])
        if not candidates:
            continue
        pbar = tqdm(total=synth_needed, desc=f"Augmenting {lbl}")
        generated = 0
        order = list(range(len(candidates)))
        rng.shuffle(order)
        while generated < synth_needed and order:
            idx_in_list = order.pop()
            src_idx, src_row = candidates[idx_in_list]
            if source_clone_counter[lbl][src_idx] >= MAX_CLONES:
                continue
            sentence = src_row['sentence']
            ents = parse_entities(src_row['ner_entities'])
            indices = [i for i, e in enumerate(ents) if e.get('label') == lbl and e.get('text')]
            rng.shuffle(indices)
            chosen_idx = None
            chosen_ent = None
            for i_ent in indices:
                ent_txt = ents[i_ent]['text']
                if safe_single_swap(sentence, ent_txt, ent_txt) is not None:
                    chosen_idx = i_ent; chosen_ent = ents[i_ent]; break
            if chosen_idx is None:
                continue
            original_text = chosen_ent['text']
            pool = [t for t in gazetteer if t != original_text]
            if not pool:
                continue
            replacement_text = rng.choice(pool)
            new_sentence = safe_single_swap(sentence, original_text, replacement_text)
            if new_sentence is None or new_sentence in seen_sentences:
                continue
            new_entities = [dict(e) for e in ents]
            new_entities[chosen_idx]['text'] = replacement_text

            augmented_rows.append({'sentence': new_sentence, 'ner_entities': json.dumps(new_entities)})
            seen_sentences.add(new_sentence)
            source_clone_counter[lbl][src_idx] += 1
            generated += 1
            pbar.update(1)
        pbar.close()

    # context augmentation
    context_rows = []
    for _, row in df.iterrows():
        ents = parse_entities(row.get('ner_entities'))
        for toks_text, new_ents in context_augment_row(row['sentence'], ents):
            if toks_text not in seen_sentences:
                context_rows.append({'sentence': toks_text, 'ner_entities': json.dumps(new_ents)})
                seen_sentences.add(toks_text)

    if not augmented_rows and not context_rows:
        print("\nNo augmented rows were generated.")
        final_df = df
    else:
        oversampled_df = pd.concat([df, pd.DataFrame(augmented_rows + context_rows)], ignore_index=True)
        post_entity_counts, post_row_counts = compute_entity_and_row_counts(oversampled_df)
        pretty_print_counts("\nEntity and row counts after augmentation:", post_entity_counts, post_row_counts)
        final_df = oversampled_df

    print("\nFinal de-duplication after augmentation.")
    size_before = len(final_df)
    final_df = final_df.drop_duplicates(subset=['sentence'], keep='first').reset_index(drop=True)
    print(f"Removed {size_before - len(final_df)} duplicates. Final dataset size: {len(final_df)}")
    return final_df

# tagging
def _collect_non_overlapping_spans(text, ents):
    # find candidate matches with word boundaries & resolve overlaps by longest-span wins.
    cand = []
    occupied = [False] * (len(text) + 1)  # char occupancy for construction
    for ent in ents:
        ent_text = ent.get('text', '')
        label = ent.get('label', '')
        if not ent_text or not label:
            continue
        for m in re.finditer(rf'\b{re.escape(ent_text)}\b', text):
            s, e = m.start(), m.end()
            cand.append((s, e, label))
    # sort by length desc, then by start asc
    cand.sort(key=lambda x: (-(x[1]-x[0]), x[0]))
    chosen = []
    used = [False] * len(text)
    for s, e, lbl in cand:
        if any(used[i] for i in range(s, e)):
            continue
        for i in range(s, e):
            used[i] = True
        chosen.append((s, e, lbl))
    chosen.sort(key=lambda x: x[0])
    return chosen

def process_data(df, out_path, is_article_data=False):
    processed = []
    iterator = df.iterrows()
    total = len(df)

    for _, row in tqdm(iterator, total=total, desc=f"Processing {os.path.basename(out_path)}"):
        text_source = row.get('text') if is_article_data else row.get('sentence')
        if pd.isna(text_source) or not isinstance(text_source, str) or not text_source.strip():
            continue

        ents = parse_entities(row.get('ner_entities'))
        sentences_to_process = (
            [sent.text for sent in nlp(text_source).sents if sent.text.strip()]
            if is_article_data else [text_source]
        )

        for text in sentences_to_process:
            enc = tokenizer(text, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=512)
            tags = ['O'] * len(enc['input_ids'])

            spans = _collect_non_overlapping_spans(text, ents)
            for s, e, lbl in spans:
                token_indices = []
                for i, (o_s, o_e) in enumerate(enc.offset_mapping):
                    if (o_s, o_e) != (0, 0) and o_s >= s and o_e <= e:
                        token_indices.append(i)
                if token_indices:
                    tags[token_indices[0]] = f'B-{lbl}'
                    for i in token_indices[1:]:
                        tags[i] = f'I-{lbl}'

            tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
            final_tokens = [tok for tok in tokens if tok not in tokenizer.all_special_tokens]
            final_tags = [tags[i] for i, tok in enumerate(tokens) if tok not in tokenizer.all_special_tokens]

            if final_tokens:
                processed.append({"tokens": " ".join(final_tokens), "ner_tags": " ".join(final_tags)})

    out_df = pd.DataFrame(processed).drop_duplicates()
    out_df.to_csv(out_path, index=False)
    print(f"Saved file: '{out_path}' ({len(out_df)} sentences)")

def main():
    print(f"Processing data for {MODEL} model")
    train_df = pd.read_csv(INPUTS['train'])
    if AUGMENT_DATA:
        train_df = augment_data(train_df)
    else:
        pre = len(train_df)
        train_df = train_df.drop_duplicates(subset=['sentence'], keep='first').reset_index(drop=True)
        print(f"Deduplicated training data: removed {pre - len(train_df)} duplicates. Current size: {len(train_df)}")

    process_data(train_df, OUTPUTS['train'], is_article_data=False)

    for key in ['val', 'test']:
        df = pd.read_csv(INPUTS[key])
        process_data(df, OUTPUTS[key], is_article_data=True)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()