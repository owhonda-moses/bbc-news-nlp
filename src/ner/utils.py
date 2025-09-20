import torch
from torch.utils.data import Dataset
import numpy as np
from seqeval.metrics import classification_report

IGNORE_INDEX = -100

class NERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, tag2id, max_len):
        self.tokenizer = tokenizer
        self.tokens = [str(x).split() for x in dataframe['tokens'].tolist()]
        self.tags = [str(x).split() for x in dataframe['ner_tags'].tolist()]
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        toks, tags = self.tokens[idx], self.tags[idx]
        if len(toks) != len(tags):
            toks, tags = [], []
        enc = self.tokenizer(
            toks,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len
        )
        word_ids = enc.word_ids()
        labels_map = [self.tag2id.get(t, self.tag2id['O']) for t in tags]
        encoded_labels = np.ones(len(word_ids), dtype=int) * IGNORE_INDEX
        prev_w = None
        for i, w in enumerate(word_ids):
            if w is None or w == prev_w:
                continue
            if w < len(labels_map):
                encoded_labels[i] = labels_map[w]
            prev_w = w
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'labels': torch.tensor(encoded_labels)
        }

def to_py(obj):
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    if isinstance(obj, tuple): return tuple(to_py(v) for v in obj)
    try:
        import numpy as _np
        if isinstance(obj, (_np.generic,)): return obj.item()
    except Exception:
        pass
    return obj

def _iob_fix(tags):
    fixed = []
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            if prev == "O" or (prev.startswith("B-") and prev[2:] != t[2:]) or (prev.startswith("I-") and prev[2:] != t[2:]):
                t = "B-" + t[2:]
        fixed.append(t)
        prev = fixed[-1]
    return fixed

def _apply_thresholds(probs_row, pred_id, id2tag, o_id, thr_map):
    k = int(pred_id)
    tag = id2tag.get(k, "O")
    if tag == "O":
        return k
    cls = tag.split("-", 1)[-1]
    thr = thr_map.get(cls, None)
    if thr is None:
        return k
    if probs_row[k].item() >= float(thr):
        return k
    return o_id

def evaluate_model(model, data_loader, device, id2tag, thresholds=None):
    # thresholds applied per token on head-subtokens only
    model.eval()
    all_true, all_pred = [], []
    # resolve O id
    o_id = None
    for i, t in id2tag.items():
        if t == "O":
            o_id = int(i); break
    if o_id is None:
        o_id = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, C]
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()
            for i in range(labels_cpu.size(0)):
                true_seq, pred_seq = [], []
                for j in range(labels_cpu.size(1)):
                    if labels_cpu[i, j].item() != IGNORE_INDEX:
                        pid = preds[i, j].item()
                        if thresholds:
                            pid = _apply_thresholds(probs[i, j], pid, id2tag, o_id, thresholds)
                        true_seq.append(id2tag[int(labels_cpu[i, j].item())])
                        pred_seq.append(id2tag[int(pid)])
                if true_seq:
                    pred_seq = _iob_fix(pred_seq)
                    all_true.append(true_seq)
                    all_pred.append(pred_seq)
    metrics = classification_report(all_true, all_pred, output_dict=True, digits=4, zero_division=0)
    return to_py(metrics), all_true, all_pred