import os
import json
import random
import warnings
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils import data as tud
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities
from src.ner.utils import NERDataset, evaluate_model, to_py

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL = "roberta"
# MODEL = "distilbert"

VERSION = "v2"

DATA_PATH = "././data/output/ner"
MODELS_PATH = "././models"
PARAMS_PATH = "././params"

MODEL_PATH = f"{MODELS_PATH}/{MODEL}-model-{VERSION}"
TEST_FILE = f"{DATA_PATH}/test_{MODEL}.csv"
TAGS_FILE = f"{PARAMS_PATH}/tags.json"
THR_FILE = f"{PARAMS_PATH}/{MODEL}_thresholds_{VERSION}.json"

BATCH_SIZE = 4
MAX_LEN = 512
PAD_TO = 8
METRICS_FILE = f"{DATA_PATH}/metrics_{MODEL}_{VERSION}.json"
ERROR_REPORT = f"{DATA_PATH}/errors_{MODEL}_{VERSION}.csv"
PREDICTIONS_FILE = f"{DATA_PATH}/predictions_{MODEL}_{VERSION}.csv"

def set_deterministic(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_tag_mapping(model_path: str, tags_file: str):
    if os.path.exists(tags_file):
        with open(tags_file, "r") as f:
            tag2id = json.load(f)
        tag2id = {str(k): int(v) for k, v in tag2id.items()}
        id2tag = {int(v): str(k) for k, v in tag2id.items()}
        return tag2id, id2tag, "json"
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        cfg_tag2id = getattr(model.config, "label2id", None)
        cfg_id2tag = getattr(model.config, "id2label", None)
        if not cfg_tag2id or not cfg_id2tag:
            raise RuntimeError("Neither tags JSON nor model.config label maps are available.")
        tag2id = {str(k): int(v) for k, v in cfg_tag2id.items()}
        id2tag = {int(k): str(v) for k, v in cfg_id2tag.items()}
        return tag2id, id2tag, "config"

def main():
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating model at: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_FILE):
        print("Model or test file missing.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    try:
        tag2id, id2tag, source = load_tag_mapping(MODEL_PATH, TAGS_FILE)
        print(f"Loaded tag mapping from {source} file")
    except Exception as e:
        print(f"Failed to load tag mapping: {e}")
        return

    thr = None
    if os.path.exists(THR_FILE):
        try:
            thr = json.load(open(THR_FILE))
            print(f"Loaded per-class thresholds from '{THR_FILE}'")
        except Exception as e:
            print(f"Could not load thresholds: {e}")

    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(device); model.eval()

    df = pd.read_csv(TEST_FILE).dropna()
    ds = NERDataset(df, tokenizer, tag2id, MAX_LEN)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=PAD_TO)
    loader = tud.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    metrics_raw, true_raw, pred_raw = evaluate_model(model, loader, device, id2tag, thresholds=None)
    metrics_thr, true_thr, pred_thr = (evaluate_model(model, loader, device, id2tag, thresholds=thr) if thr else (metrics_raw, true_raw, pred_raw))

    with open(METRICS_FILE, "w") as f:
        json.dump(to_py({"raw": metrics_raw, "thresholded": metrics_thr}), f, indent=2)
    print(f"Metrics saved to '{METRICS_FILE}'")

    sents = df["tokens"].tolist()
    errors = []
    n = min(len(true_thr), len(pred_thr), len(sents))
    for i in range(n):
        if true_thr[i] != pred_thr[i]:
            errors.append({
                "index": i,
                "sentence": sents[i],
                "true_tags": " ".join(true_thr[i]),
                "predicted_tags": " ".join(pred_thr[i]),
                "true_entities": str(get_entities(true_thr[i])),
                "predicted_entities": str(get_entities(pred_thr[i])),
            })
    if errors:
        pd.DataFrame(errors).to_csv(ERROR_REPORT, index=False)
        print(f"Error report saved to '{ERROR_REPORT}'")

    pd.DataFrame([{
        "index": i,
        "sentence": sents[i],
        "true_tags": " ".join(true_thr[i]),
        "predicted_tags": " ".join(pred_thr[i]),
    } for i in range(n)]).to_csv(PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to '{PREDICTIONS_FILE}'")

if __name__ == "__main__":
    main()