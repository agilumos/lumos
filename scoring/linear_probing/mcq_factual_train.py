#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train probes on (Question + Choices + Reasoning + AnswerText) with Inside-out style pairing:
- For each base_qid: (q, a_pos, y=1) + (q, a_neg, y=0)
- Here q-context includes choices and raw_samples (reasoning) if available per row.

Then evaluate per layer on dev split, pick BEST layer by AUC (fallback acc),
and SAVE ONLY the best layer probe:
  out_dir/probes/layer_{best_layer}.joblib
Plus:
  out_dir/metadata.json
  out_dir/dev_metrics.json
"""

import argparse, json, os, random, re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

_ID_SUFFIX_RE = re.compile(r"_(\d+)$")


def base_qid(qid: str) -> str:
    return _ID_SUFFIX_RE.sub("", qid)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_text_from_field(val: Any, choices: Optional[List[str]]) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, (int, np.integer)):
        if choices and 0 <= int(val) < len(choices):
            return choices[int(val)]
        return str(int(val))
    return "" if val is None else str(val)


def answer_text(row: Dict[str, Any]) -> str:
    return get_text_from_field(row.get("answer"), row.get("choices"))


def gold_text(row: Dict[str, Any]) -> str:
    return get_text_from_field(row.get("gold_answer"), row.get("choices"))


def compute_label(row: Dict[str, Any]) -> int:
    if "label" in row and row["label"] is not None:
        return int(row["label"])
    return int(answer_text(row).strip() == gold_text(row).strip())


def format_q_choices_reasoning_answer(q: str, choices: Optional[List[str]], reasoning: str, a: str) -> str:
    choices = choices if isinstance(choices, list) else []
    choice_block = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    reasoning = (reasoning or "").strip()
    return (
        f"Question: {q}\n"
        f"Choices:\n{choice_block}\n"
        f"Reasoning: {reasoning}\n"
        f"Answer: {a}"
    )


def build_one_pos_one_neg_per_question_with_reasoning(
    rows: List[Dict[str, Any]],
    seed: int = 42
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Exactly 2 samples per base_id if possible: 1 pos + 1 neg.
    - Uses per-row raw_samples as Reasoning (if available)
    - Uses choices block as context
    - weights are all 1.0
    """
    rng = random.Random(seed)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if "id" not in r:
            continue
        grouped.setdefault(base_qid(r["id"]), []).append(r)

    texts, ys, ws, bids = [], [], [], []

    for bid, grp in grouped.items():
        q = next((r.get("question") for r in grp if r.get("question")), None)
        if not q:
            continue

        choices = grp[0].get("choices")
        gold = grp[0].get("gold_answer")

        gold_txt = get_text_from_field(gold, choices).strip()
        if not gold_txt:
            continue

        # POS row: prefer label==1 and has answer_text
        pos_row = next((r for r in grp if compute_label(r) == 1 and answer_text(r).strip()), None)
        pos_txt = answer_text(pos_row).strip() if pos_row else gold_txt
        pos_reason = (pos_row.get("raw_samples") if pos_row else "") or ""

        # NEG row: prefer label==0 and has answer_text
        neg_row = next((r for r in grp if compute_label(r) == 0 and answer_text(r).strip()), None)
        if neg_row:
            neg_txt = answer_text(neg_row).strip()
            neg_reason = (neg_row.get("raw_samples") or "")
        else:
            # fallback: sample from choices excluding gold
            neg_txt, neg_reason = "", ""
            if isinstance(choices, list) and len(choices) >= 2:
                g_idx = int(gold) if isinstance(gold, (int, np.integer)) else None
                if g_idx is not None:
                    candidates = [c for i, c in enumerate(choices) if i != g_idx and str(c).strip()]
                    if candidates:
                        neg_txt = str(rng.choice(candidates)).strip()
                        neg_reason = ""  # fallback has no row reasoning

        if not pos_txt or not neg_txt or pos_txt == neg_txt:
            continue

        # Build final texts (each uses its own reasoning if present)
        texts.append(format_q_choices_reasoning_answer(q, choices, pos_reason, pos_txt)); ys.append(1); ws.append(1.0); bids.append(bid)
        texts.append(format_q_choices_reasoning_answer(q, choices, neg_reason, neg_txt)); ys.append(0); ws.append(1.0); bids.append(bid)

    return (
        texts,
        np.asarray(ys, dtype=np.int64),
        np.asarray(ws, dtype=np.float32),
        np.asarray(bids, dtype=object),
    )


@torch.no_grad()
def extract_hidden(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    dtype: str,
    use_chat_template: bool
) -> List[np.ndarray]:
    """
    Return list of features per layer (1..L), each [N, H] from last non-pad token.
    """
    model.eval()
    device = next(model.parameters()).device

    amp_dtype = None
    if dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif dtype == "fp16":
        amp_dtype = torch.float16

    layer_chunks: Optional[List[List[np.ndarray]]] = None

    for i in tqdm(range(0, len(texts), batch_size), desc="Extract hidden"):
        bt = texts[i:i + batch_size]

        if use_chat_template and getattr(tokenizer, "chat_template", None):
            # For chat template we put everything as one user message (stable + simple)
            ids_list = []
            for t in bt:
                msgs = [{"role": "user", "content": t}]
                ids = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length
                )
                ids_list.append(torch.tensor(ids, dtype=torch.long))

            input_ids = torch.nn.utils.rnn.pad_sequence(
                ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        else:
            enc = tokenizer(bt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in enc.items()}

        with torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
            out = model(**batch, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states  # (emb, l1..lL)
        if layer_chunks is None:
            layer_chunks = [[] for _ in range(len(hs) - 1)]  # skip emb

        attn = batch["attention_mask"]
        lengths = attn.sum(dim=1)
        last_pos = (lengths - 1).clamp(min=0)

        for li in range(1, len(hs)):
            h = hs[li]  # [B,T,H]
            bsz = h.size(0)
            idx = last_pos.view(bsz, 1, 1).expand(bsz, 1, h.size(-1))
            last_h = h.gather(dim=1, index=idx).squeeze(1)  # [B,H]
            layer_chunks[li - 1].append(last_h.float().cpu().numpy())

    assert layer_chunks is not None
    return [np.concatenate(ch, axis=0) for ch in layer_chunks]


def atomic_joblib_dump(obj, path: str):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)  # atomic rename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--train_ratio", type=float, default=0.8)  # split by base_qid
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    set_seed(args.seed)

    rows = []
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    texts, y, w, bids = build_one_pos_one_neg_per_question_with_reasoning(rows, seed=args.seed)

    if len(texts) == 0:
        raise RuntimeError("No training samples built. Check answer_text/gold_text extraction.")

    uniq_q = sorted(set(bids.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(uniq_q)

    n_train_q = int(len(uniq_q) * args.train_ratio)
    train_set = set(uniq_q[:n_train_q])
    dev_set = set(uniq_q[n_train_q:])

    train_mask = np.asarray([bid in train_set for bid in bids], dtype=bool)
    dev_mask = ~train_mask

    print(f"[DEBUG] samples={len(texts)} (should be 2 * #used_questions), unique_q={len(uniq_q)}", flush=True)
    print(f"[DEBUG] train_q={len(train_set)} dev_q={len(dev_set)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, device_map=args.device_map
    )

    feats_by_layer = extract_hidden(
        model, tokenizer, texts, args.batch_size, args.max_length, args.dtype, args.use_chat_template
    )

    os.makedirs(args.out_dir, exist_ok=True)
    probes_dir = os.path.join(args.out_dir, "probes")
    os.makedirs(probes_dir, exist_ok=True)

    print(f"[DEBUG] will save probe to: {probes_dir}", flush=True)

    dev_metrics = {}
    best_layer, best_score = None, -1.0
    best_pack = None

    for layer_idx, X in enumerate(tqdm(feats_by_layer, desc="Train probes"), start=1):
        Xtr, ytr, wtr = X[train_mask], y[train_mask], w[train_mask]
        Xdv, ydv, wdv = X[dev_mask], y[dev_mask], w[dev_mask]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xdv_s = scaler.transform(Xdv)

        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced", n_jobs=1
        )
        clf.fit(Xtr_s, ytr, sample_weight=wtr)

        p = clf.predict_proba(Xdv_s)[:, 1]
        pred = (p >= 0.5).astype(np.int64)

        acc = float(accuracy_score(ydv, pred, sample_weight=wdv))
        try:
            auc = float(roc_auc_score(ydv, p, sample_weight=wdv))
        except Exception:
            auc = float("nan")

        dev_metrics[str(layer_idx)] = {"acc": acc, "auc": auc}

        score = auc if not np.isnan(auc) else acc
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_pack = {"layer": layer_idx, "scaler": scaler, "clf": clf}

    with open(os.path.join(args.out_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

    if best_pack is None or best_layer is None:
        raise RuntimeError("best_pack is None. Training might have failed before selecting best layer.")

    best_path = os.path.join(probes_dir, f"layer_{best_layer}.joblib")
    atomic_joblib_dump(best_pack, best_path)
    print(f"[SAVED] best probe -> {best_path}", flush=True)

    meta = {
        "model_path": args.model_path,
        "data_jsonl": args.data_jsonl,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "use_chat_template": bool(args.use_chat_template),
        "num_layers": len(feats_by_layer),
        "num_samples": int(len(texts)),
        "num_unique_questions": int(len(uniq_q)),
        "best_layer": int(best_layer),
        "best_score": float(best_score),
        "best_probe_path": best_path,
        "text_format": "Question + Choices + Reasoning(raw_samples) + Answer",
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
