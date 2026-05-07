"""
Portions of this implementation are adapted from TruthTorchLM / SAPLMA.

TruthTorchLM is licensed under the MIT License.
Copyright (c) 2024 Yavuz Faruk Bakman

See THIRD_PARTY_LICENSES/TruthTorchLM_LICENSE.txt for the full license text.

[SAPLMA probe - INFER]

Output fields added to each row:
  - saplma_best:
      per_candidate : list of {cand, p_true}  (ordered A/B/C/D)
      pred_letter   : str    choice with highest p_true ("A"/"B"/"C"/"D")
      is_correct    : bool   pred_letter == gold_answer
      gold_letter   : str    correct answer choice
      agg_max       : float  max p_true
      agg_mean      : float  mean p_true
      k_used        : int    number of choices (typically 4)
  - saplma_meta:
      probe_dir, best_layer, feature
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    LETTERS,
    build_sequence_text,
    gold_letter,
    load_jsonl,
    load_model_and_tokenizer,
    set_seed,
    derive_paths,
    derive_prompt_id,
    # module-private helpers: importable in Python
    # functions already used internally in utils.py
    _encode_batch,
    _last_nonpad_hidden,
)

# ---------------------------------------------------------------------------
# Probe I/O
# ---------------------------------------------------------------------------

def load_best_probe(probe_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Load the best probe pack and its metadata from *probe_dir*."""
    meta_path = os.path.join(probe_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    best_layer = int(meta["best_layer"])
    probe_path = os.path.join(probe_dir, "probes", f"layer_{best_layer}.joblib")
    pack = joblib.load(probe_path)
    return pack, meta, best_layer


# ---------------------------------------------------------------------------
# All-choices hidden state extraction (single layer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_all_choices(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    layer_index: int,
    batch_size: int,
    max_length: int,
    dtype: str,
    prompt_id: int,
) -> Tuple[List[Optional[Dict[str, Any]]], np.ndarray]:
    """
    For each row, perform a forward pass on ALL choices (A/B/C/D),
    regardless of what the model actually generated.

    Returns
    -------
    row_info : list aligned with *rows*.
               None  -> row had no valid choices or gold_answer.
               dict  -> {
                   "_flat_indices":    list[int],   # positions in X_flat
                   "_flat_candidates": list[str],   # ["A","B","C","D"]
                   "_gold_letter":     str,
               }
    X_flat   : np.ndarray [N_flat, H]
    """
    model.eval()
    device    = next(model.parameters()).device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)

    # Build flat list: one entry per (row, choice_letter)
    # flat[j] = (row_index, cand_letter, question, choices, reasoning, gold)
    flat: List[Tuple[int, str, str, List[str], str, str]] = []

    for ri, r in enumerate(rows):
        q       = r.get("question")
        choices = r.get("choices")
        if not isinstance(q, str) or not isinstance(choices, list) or not choices:
            continue

        g = gold_letter(r)
        if g is None:
            continue

        valid_letters = LETTERS[: len(choices)]
        for c in valid_letters:
            flat.append((ri, c, q, choices, "", g))

    row_info: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    if not flat:
        return row_info, np.zeros((0, 0), dtype=np.float32)

    feats: List[np.ndarray] = []
    for i in tqdm(
        range(0, len(flat), batch_size),
        desc=f"Extracting hidden states (layer {layer_index})",
    ):
        batch_flat = flat[i : i + batch_size]
        texts = [
            build_sequence_text(prompt_id, q, choices, cand, rsn)
            for _, cand, q, choices, rsn, _ in batch_flat
        ]

        batch = _encode_batch(tokenizer, texts, max_length, device)

        with torch.amp.autocast(
            "cuda", enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32
        ):
            out = model(**batch, output_hidden_states=True, use_cache=False)

        h = out.hidden_states[layer_index]  # [B, T, H]
        del out

        vec = _last_nonpad_hidden(h, batch["attention_mask"])  # [B, H]
        feats.append(vec.float().cpu().numpy())

    X_flat = np.concatenate(feats, axis=0)  # [N_flat, H]

    # Map flat positions back to per-row
    grouped: Dict[int, List[int]] = {}
    for j, (ri, *_) in enumerate(flat):
        grouped.setdefault(ri, []).append(j)

    for ri, idxs in grouped.items():
        g = flat[idxs[0]][5]  # gold_letter stored in flat tuple
        row_info[ri] = {
            "_flat_indices":    idxs,
            "_flat_candidates": [flat[j][1] for j in idxs],
            "_gold_letter":     g,
        }

    return row_info, X_flat


# ---------------------------------------------------------------------------
# Inference command
# ---------------------------------------------------------------------------

def cmd_infer(args: argparse.Namespace) -> None:
    if args.model_path is None or args.probe_dir is None or args.out_jsonl is None:
        derived_model, derived_out = derive_paths(args.data_jsonl)
        if args.model_path is None:
            args.model_path = derived_model
            print(f"[AUTO] model_path = {args.model_path}", flush=True)
        if args.probe_dir is None:
            args.probe_dir = derived_out
            print(f"[AUTO] probe_dir  = {args.probe_dir}", flush=True)
        if args.out_jsonl is None:
            args.out_jsonl = os.path.join(derived_out, "scored.jsonl")
            print(f"[AUTO] out_jsonl  = {args.out_jsonl}", flush=True)
    prompt_id = derive_prompt_id(args.data_jsonl)
    print(f"[AUTO] prompt_id  = {prompt_id}", flush=True)

    set_seed(args.seed)

    pack, meta, best_layer = load_best_probe(args.probe_dir)
    scaler = pack["scaler"]
    clf    = pack["clf"]

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.dtype, args.device_map
    )

    rows = load_jsonl(args.data_jsonl)

    row_info, X_flat = extract_hidden_all_choices(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        layer_index=best_layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        prompt_id=1, ### hard coding
    )

    saplma_meta_out = {
        "probe_dir":  args.probe_dir,
        "best_layer": int(best_layer),
        "feature":    meta.get("feature"),
    }

    if X_flat.size > 0:
        Xs         = scaler.transform(X_flat)
        probs_flat = clf.predict_proba(Xs)[:, 1]  # [N_flat]
    else:
        probs_flat = np.array([], dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            out  = dict(r)
            info = row_info[i]

            if info is None or X_flat.size == 0:
                out["saplma_best"]   = None
                out["saplma_reason"] = (
                    "no_valid_choices" if info is None else "empty_feature_matrix"
                )
            else:
                idxs   = info["_flat_indices"]
                cands  = info["_flat_candidates"]
                g      = info["_gold_letter"]
                p_list = [float(probs_flat[j]) for j in idxs]

                # pred_letter: the choice with the highest p_true
                best_idx    = int(np.argmax(p_list))
                pred_letter = cands[best_idx]
                is_correct  = (pred_letter == g)

                out["saplma_best"] = {
                    "per_candidate": [
                        {"cand": c, "p_true": p}
                        for c, p in zip(cands, p_list)
                    ],
                    "pred_letter":  pred_letter,
                    "gold_letter":  g,
                    "is_correct":   is_correct,
                    "agg_max":      float(max(p_list)),
                    "agg_mean":     float(sum(p_list) / len(p_list)),
                    "k_used":       int(len(p_list)),
                }

            out["saplma_meta"] = saplma_meta_out
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"[DONE] best_layer={best_layer}  wrote {len(rows)} rows -> {args.out_jsonl}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Score all choices (A~D) with a trained SAPLMA probe."
    )

    ap.add_argument("--data_jsonl", required=True, help="Input JSONL file to score.")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--probe_dir",  default=None)
    ap.add_argument("--out_jsonl",  default=None)

    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--max_length", type=int,   default=512)
    ap.add_argument("--dtype",      choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--seed",       type=int,   default=42)

    return ap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd_infer(build_parser().parse_args())
