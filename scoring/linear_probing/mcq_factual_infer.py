#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference with BEST layer probe only.

- Load best_layer from {probe_dir}/metadata.json
- Load probe pack from {probe_dir}/probes/layer_{best_layer}.joblib
- For each row in input JSONL:
    text = "Question: ...\nChoices: ...\nReasoning: ...\nAnswer: <answer_text>"
  (answer can be str or int; if int, use choices[answer])
  reasoning uses raw_samples (if missing -> "")
- Extract hidden state of LAST non-pad token at best_layer only
- Predict probe_score = P(y=1 | h_best)
- Write augmented JSONL:
    - base_id
    - answer_text_used
    - probe_best_layer
    - probe_score
"""

import argparse, json, os, random, re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    if val is None:
        return ""
    # numeric string -> choice mapping
    if isinstance(val, str):
        s = val.strip()
        if choices and s.isdigit():
            idx = int(s)
            if 0 <= idx < len(choices):
                return choices[idx]
        return s
    if isinstance(val, (int, np.integer)):
        if choices and 0 <= int(val) < len(choices):
            return choices[int(val)]
        return str(int(val))
    return str(val)

def answer_text(row: Dict[str, Any]) -> str:
    return get_text_from_field(row.get("answer"), row.get("choices"))

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

def load_best_probe(probe_dir: str):
    meta_path = os.path.join(probe_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    best_layer = int(meta["best_layer"])
    probe_path = os.path.join(probe_dir, "probes", f"layer_{best_layer}.joblib")
    pack = joblib.load(probe_path)  # {"layer":..., "scaler":..., "clf":...}
    return pack, meta, best_layer

@torch.no_grad()
def extract_best_layer_hidden(
    model,
    tokenizer,
    texts: List[str],
    best_layer: int,
    batch_size: int,
    max_length: int,
    dtype: str,
    use_chat_template: bool,
) -> np.ndarray:
    """
    Return [N, H] hidden features from best_layer at last non-pad token.
    best_layer is 1-indexed (same as training loop used).
    """
    model.eval()
    device = next(model.parameters()).device

    amp_dtype = None
    if dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif dtype == "fp16":
        amp_dtype = torch.float16

    chunks: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extract best-layer({best_layer}) hidden"):
        bt = texts[i:i+batch_size]

        if use_chat_template and getattr(tokenizer, "chat_template", None):
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

        h = out.hidden_states[best_layer]  # [B, T, H]

        attn = batch["attention_mask"]
        lengths = attn.sum(dim=1)
        last_pos = (lengths - 1).clamp(min=0)

        bsz = h.size(0)
        idx = last_pos.view(bsz, 1, 1).expand(bsz, 1, h.size(-1))
        last_h = h.gather(dim=1, index=idx).squeeze(1)  # [B,H]
        chunks.append(last_h.float().cpu().numpy())

    return np.concatenate(chunks, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--probe_dir", required=True)   # contains metadata.json + probes/
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    pack, meta, best_layer = load_best_probe(args.probe_dir)
    scaler = pack["scaler"]
    clf = pack["clf"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, device_map=args.device_map
    )

    rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    used_answers: List[str] = []

    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            q = (r.get("question") or "").strip()
            ch = r.get("choices")
            reasoning = (r.get("raw_samples") or "").strip()
            a = answer_text(r).strip()

            rows.append(r)

            if not q or not a:
                texts.append("")  # placeholder -> NaN score
                used_answers.append(a if a else "")
                continue

            texts.append(format_q_choices_reasoning_answer(q, ch, reasoning, a))
            used_answers.append(a)

    valid_idx = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_idx]

    scores = [float("nan")] * len(rows)
    if valid_texts:
        X = extract_best_layer_hidden(
            model, tokenizer, valid_texts,
            best_layer=best_layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype=args.dtype,
            use_chat_template=args.use_chat_template,
        )
        Xs = scaler.transform(X)
        p = clf.predict_proba(Xs)[:, 1]
        for j, i in enumerate(valid_idx):
            scores[i] = float(p[j])

    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for i, r in enumerate(rows):
            out = dict(r)
            out["base_id"] = base_qid(r["id"]) if "id" in r else None
            out["answer_text_used"] = used_answers[i]
            out["probe_best_layer"] = int(best_layer)
            out["probe_score"] = scores[i]
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[DONE] best_layer={best_layer} -> wrote {len(rows)} rows to {args.out_jsonl}")

if __name__ == "__main__":
    main()
