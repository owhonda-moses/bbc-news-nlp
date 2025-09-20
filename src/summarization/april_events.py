import os
import re
import json
import calendar
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast


DATA_PATH = "./data/output"
TRAIN = "train_data.csv"
TEST = "test_set.csv"

# SOURCE_DATA_PATH = os.path.join(DATA_PATH, TEST)
SOURCE_DATA_PATH = os.path.join(DATA_PATH, TRAIN)

TARGET_MONTH = "April"

OUTPUT_DIR = os.path.join(DATA_PATH, f"{TARGET_MONTH.lower()}_events")
OUTPUT_FILE = f"summaries_{os.path.basename(SOURCE_DATA_PATH).split('_')[0]}.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

MODEL_NAME = "facebook/bart-large-cnn"

MAX_INPUT_TOKENS = 1024
BATCH_SIZE = 4
SEED = 42

TOP_SNIPPETS = 8
CONTEXT_WINDOW = 1
MIN_ABS_SUM_LEN = 8
MAX_ABS_SUM_LEN = 120


# month patterns
MONTH_NUM = list(calendar.month_name).index(TARGET_MONTH.capitalize())
MONTH_ABBR_L = calendar.month_abbr[MONTH_NUM].lower()
MONTH_NUM_STR = f"{MONTH_NUM:02d}"
MONTH_FULL = TARGET_MONTH.lower()
MONTH_TOKEN = rf"(?:{MONTH_ABBR_L}|{MONTH_FULL})"

MONTH_PATTERNS_RAW = [
    rf"\b{MONTH_TOKEN}\.?\b",
    rf"\b{MONTH_TOKEN}\.?\s+\d{{1,2}}(?:,\s*\d{{4}})?",
    rf"\b\d{{1,2}}\s+{MONTH_TOKEN}\.?\s*(?:,\s*\d{{4}})?",
    rf"\b(?:20\d{{2}})-{MONTH_NUM_STR}-\d{{2}}\b",
    rf"\b0?{MONTH_NUM}[/.-]\d{{1,2}}[/.-](20\d{{2}})\b",
    rf"\b{MONTH_FULL}\s+(20\d{{2}})\b",
    rf"\b(in|by|from|until|through|thru)\s+{MONTH_FULL}\b",
    rf"\b(end|late|early|mid)\s+of\s+{MONTH_FULL}\b",
    rf"\b(early|mid|late)\s+{MONTH_FULL}\b",
]
MONTH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MONTH_PATTERNS_RAW]

EVENT_VERBS = [
    "announced","scheduled","rescheduled","postponed","canceled","cancelled",
    "began","begin","starts","started","start","launch","launched","host","hosted","hold","held",
    "meet","met","meeting","vote","voted","election","fixture","match","play","played","appear","appeared",
    "attend","attended","unveil","unveiled","opened","opens","opening","kicked off","kickoff","debuts","debut",
    "expects","expected","plan","plans","planned","will","due","set to"
]


# utils
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)

def detect_mentions_month(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    lt = text.lower()
    return any(p.search(lt) for p in MONTH_PATTERNS)

def normalize_dates(text: str) -> List[str]:
    lt = (text or "").lower()
    dates = []
    for m in re.finditer(rf"\b(20\d{{2}})-{MONTH_NUM_STR}-(\d{{2}})\b", lt):
        dates.append(f"{m.group(1)}-{MONTH_NUM_STR}-{m.group(2)}")
    for m in re.finditer(rf"\b{MONTH_TOKEN}\.?\s+(\d{{1,2}})(?:,\s*(20\d{{2}}))?", lt):
        d = int(m.group(1)); y = m.group(2) if m.group(2) else "—"
        dates.append(f"{y}-{MONTH_NUM_STR}-{d:02d}" if y != "—" else f"—-{MONTH_NUM_STR}-{d:02d}")
    for m in re.finditer(rf"\b(\d{{1,2}})\s+{MONTH_TOKEN}\.?\s*(?:,\s*)?(20\d{{2}})?", lt):
        d = int(m.group(1)); y = m.group(2) if m.group(2) else "—"
        dates.append(f"{y}-{MONTH_NUM_STR}-{d:02d}" if y != "—" else f"—-{MONTH_NUM_STR}-{d:02d}")
    for m in re.finditer(rf"\b0?{MONTH_NUM}[/.-](\d{{1,2}})[/.-](20\d{{2}})\b", lt):
        d = int(m.group(1)); y = m.group(2)
        dates.append(f"{y}-{MONTH_NUM_STR}-{d:02d}")
    for m in re.finditer(rf"\b{MONTH_FULL}\s+(20\d{{2}})\b", lt):
        dates.append(f"{m.group(1)}-{MONTH_NUM_STR}")
    if any(re.search(p, lt) for p in [
        rf"\b(in|by|from|until|through|thru)\s+{MONTH_FULL}\b",
        rf"\b(end|late|early|mid)\s+of\s+{MONTH_FULL}\b",
        rf"\b(early|mid|late)\s+{MONTH_FULL}\b",
        rf"\b{MONTH_FULL}\b"
    ]) and not dates:
        dates.append(f"—-{MONTH_NUM_STR}")
    seen = set(); out = []
    for d in dates:
        if d not in seen:
            seen.add(d); out.append(d)
    return out

def relevance_score(sent: str) -> int:
    s = (sent or "").lower()
    score = 0
    for p in MONTH_PATTERNS:
        if p.search(s): score += 2
    for v in EVENT_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", s): score += 1
    return score

def collect_month_evidence(text: str, max_snippets: int = TOP_SNIPPETS, context_window: int = CONTEXT_WINDOW) -> Tuple[List[str], List[str]]:
    sents = sentence_split(text)
    scored = []
    for i, s in enumerate(sents):
        if detect_mentions_month(s):
            score = relevance_score(s)
            start = max(0, i - context_window)
            end = min(len(sents), i + context_window + 1)
            snippet = " ".join(sents[start:end]).strip()
            if snippet:
                scored.append((score, i, snippet))
    scored.sort(key=lambda x: (-x[0], x[1]))
    snippets, seen = [], set()
    for sc, idx, sn in scored:
        if sn not in seen:
            seen.add(sn); snippets.append(sn)
        if len(snippets) >= max_snippets:
            break
    dates = normalize_dates(" ".join(snippets))
    return snippets, dates

def build_evidence_article(snippets: List[str]) -> str:
    return " ".join(snippets).strip()

def is_empty_or_boilerplate(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    bad = [
        "summarize", "summarise", "only the events", "ignore information",
        f"no {MONTH_FULL}-specific", f"no {MONTH_FULL}"
    ]
    return any(b in t for b in bad) or len(t.split()) < 5

def extractive_fallback(snippets: List[str], dates: List[str]) -> str:
    if not snippets:
        return f"No {TARGET_MONTH}-specific events could be identified from the article."
    main = snippets[0]
    main = re.sub(r'^["“”]+', '', main).strip()
    main = re.sub(r'\s+["“”]+$', '', main).strip()
    if TARGET_MONTH.lower() not in main.lower() and dates:
        day_dates = [d for d in dates if re.match(r"^(?:—|20\d{2})-" + MONTH_NUM_STR + r"-\d{2}$", d)]
        month_only = [d for d in dates if re.match(r"^(?:—|20\d{2})-" + MONTH_NUM_STR + r"$", d)]
        add = day_dates[0] if day_dates else (month_only[0] if month_only else None)
        if add:
            main = f"{main} ({add})"
    return main

def summarize_with_bart(model, tokenizer, device, articles: List[str]) -> List[str]:
    enc = tokenizer(
        articles,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_TOKENS
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            min_length=MIN_ABS_SUM_LEN,
            max_length=MAX_ABS_SUM_LEN,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True
        )
    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [text.strip() for text in texts]

def confidence_score(ev_sents: List[str], dates: List[str], used_fallback: bool, abstr_len: int) -> float:
    verb_hits = sum(any(re.search(rf"\b{re.escape(v)}\b", s.lower()) for v in EVENT_VERBS) for s in ev_sents)
    explicit_day = any(re.match(r"^(?:—|20\d{2})-" + MONTH_NUM_STR + r"-\d{2}$", d) for d in dates)
    explicit_month = any(re.match(r"^(?:—|20\d{2})-" + MONTH_NUM_STR + r"$", d) for d in dates)
    base = 0.3 + 0.04 * min(len(ev_sents), 10) + 0.03 * min(verb_hits, 6)
    if explicit_day:
        base += 0.2
    elif explicit_month:
        base += 0.1
    if used_fallback:
        base -= 0.1
    if abstr_len >= 10:
        base += 0.05
    return round(max(0.0, min(1.0, base)), 3)

def batch_iter(df: pd.DataFrame, batch_size: int):
    n = len(df)
    for i in range(0, n, batch_size):
        yield df.iloc[i:i+batch_size]


# main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    print(f"Summarizing {os.path.basename(SOURCE_DATA_PATH).split('_')[0]} data")
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    df = pd.read_csv(SOURCE_DATA_PATH)
    print(f"Loaded {len(df)} documents.")

    col_name = f"mentions_{TARGET_MONTH.lower()}"
    if col_name in df.columns:
        work_df = df[df[col_name].astype(bool)].copy()
        print(f"Filtered to {len(work_df)} {TARGET_MONTH}-mentioning docs via existing column.")
    else:
        df[col_name] = df["text"].apply(detect_mentions_month)
        work_df = df[df[col_name]].copy()
        print(f"Filtered to {len(work_df)} {TARGET_MONTH}-mentioning docs via auto-detection.")

    if work_df.empty:
        print(f"No {TARGET_MONTH}-mentioning articles found.")
        pd.DataFrame(columns=["unique_id", "summary", f"{TARGET_MONTH.lower()}_dates", "evidence_sentences", "confidence"]).to_csv(OUTPUT_PATH, index=False)
        return

    outputs = []
    for batch_df in tqdm(batch_iter(work_df, BATCH_SIZE), total=len(work_df)//BATCH_SIZE):
        items = []
        evidence_articles_batch = []
        for _, row in batch_df.iterrows():
            text = str(row["text"])
            ev_sents, dates = collect_month_evidence(text)
            items.append((row.get("unique_id", None), ev_sents, dates))
            evidence_articles_batch.append(build_evidence_article(ev_sents))
        
        # get batch summaries
        abstractive_summaries = summarize_with_bart(model, tokenizer, device, evidence_articles_batch)

        # process results
        for i, (uid, ev_sents, dates) in enumerate(items):
            if not ev_sents:
                summary = f"No {TARGET_MONTH}-specific events could be identified in the article."
                conf = confidence_score(ev_sents, dates, used_fallback=True, abstr_len=0)
            else:
                abstractive = abstractive_summaries[i]
                if is_empty_or_boilerplate(abstractive):
                    summary = extractive_fallback(ev_sents, dates)
                    conf = confidence_score(ev_sents, dates, used_fallback=True, abstr_len=0)
                else:
                    summary = abstractive
                    conf = confidence_score(ev_sents, dates, used_fallback=False, abstr_len=len(summary.split()))

            outputs.append({
                "unique_id": uid,
                "summary": summary,
                f"{TARGET_MONTH.lower()}_dates": json.dumps(dates, ensure_ascii=False),
                "evidence_sentences": json.dumps(ev_sents, ensure_ascii=False),
                "confidence": conf
            })

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out_df)} summaries to {OUTPUT_PATH}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()