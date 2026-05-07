#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
---------------
Shared utilities for saplma_train.py and saplma_infer.py.

Contents
--------
Constants
    LETTERS, _ID_SUFFIX_RE

Data utilities
    set_seed
    normalize_letter
    base_qid
    gold_letter
    get_generated_candidates
    extract_reasoning_from_row
    dedup_ordered
    atomic_joblib_dump
    load_jsonl
    load_model_and_tokenizer

Sequence builders
    _choice_block_numbered      (module-private helper)
    _cand_to_answer_text        (module-private helper)
    build_sequence_text         (public entry point)

Hidden-state extraction
    extract_last_nonpad_hidden  (shared batch-level helper)
"""

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_ROOT  = ""
OUT_ROOT   = ""
MODEL_ROOT = ""

SIZE_TO_MODEL = {
    "7B":  "OLMo-2-1124-7B-Instruct",
    "13B": "OLMo-2-1124-13B-Instruct",
    "32B": "OLMo-2-0325-32B-Instruct",
}

def derive_paths(data_jsonl: str) -> tuple[str, str]:
    """
    Automatically derive model_path and out_dir from the data_jsonl path.

    Expected pattern:
        {DATA_ROOT}/{size}/{prompt}/{stage}/{exp}/{exp}_probing.jsonl

    Returns:
        model_path : str
        out_dir    : str
    """
    # Extract relative path under DATA_ROOT
    rel = os.path.relpath(data_jsonl, DATA_ROOT)
    parts = Path(rel).parts

    if len(parts) < 4:
        raise ValueError(
            f"data_jsonl path does not match expected pattern.\n"
            f"  Expected: {{DATA_ROOT}}/{{size}}/{{prompt}}/{{stage}}/{{exp}}/{{exp}}_probing.jsonl\n"
            f"  Actual: {data_jsonl}"
        )

    size = parts[0]

    if size not in SIZE_TO_MODEL:
        raise ValueError(
            f"No model registered for size '{size}'. "
            f"Available sizes: {list(SIZE_TO_MODEL)}"
        )

    model_dir  = SIZE_TO_MODEL[size]
    model_path = os.path.join(MODEL_ROOT, model_dir)
    out_dir    = os.path.join(OUT_ROOT, *parts[:-1])

    return model_path, out_dir

def derive_prompt_id(data_jsonl: str) -> int:
    """
    Extract prompt id from the data_jsonl path.
    Assumes that one of 'prompt1', 'prompt2', 'prompt3' is included in the path.

    Example: .../internal_prompt3/... -> 3
    """
    import re
    m = re.search(r"prompt(\d+)", data_jsonl)
    if m is None:
        raise ValueError(
            f"Cannot find prompt id in data_jsonl path: {data_jsonl}\n"
            f"The path must include one of 'prompt1' / 'prompt2' / 'prompt3'."
        )
    pid = int(m.group(1))
    if pid not in (1, 2, 3, 4):
        raise ValueError(
            f"Unsupported prompt id: {pid} (path: {data_jsonl})\n"
            f"Allowed values: 1, 2, 3, 4"
        )
    return pid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LETTERS: List[str] = ["A", "B", "C", "D"]
_ID_SUFFIX_RE = re.compile(r"_(\d+)$")

# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_letter(s: Any) -> Optional[str]:
    """
    Convert an int index (0-3) or a letter string to 'A'/'B'/'C'/'D'.
    Returns None for anything that cannot be mapped.
    """
    if s is None:
        return None
    if isinstance(s, (int, np.integer)):
        return LETTERS[int(s)] if 0 <= int(s) < 4 else None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().upper()
    if not s:
        return None
    ch = s[0]
    return ch if ch in {"A", "B", "C", "D"} else None


def base_qid(qid: str) -> str:
    """Strip a trailing '_<number>' suffix to obtain the base question id."""
    return _ID_SUFFIX_RE.sub("", qid)


def gold_letter(row: Dict[str, Any]) -> Optional[str]:
    """Return the normalized gold answer letter for a data row."""
    return normalize_letter(row.get("gold_answer"))


def get_generated_candidates(row: Dict[str, Any]) -> List[str]:
    """
    Extract normalized letter candidates from a data row.

    Priority order:
        1. raw_samples  (list[str])
        2. raw_letters  (list[str])
        3. answer       (int | str, single value)
    """
    cands: List[str] = []

    if isinstance(row.get("raw_samples"), list):
        for s in row["raw_samples"]:
            letter = normalize_letter(s)
            if letter:
                cands.append(letter)

    if not cands and isinstance(row.get("raw_letters"), list):
        for s in row["raw_letters"]:
            letter = normalize_letter(s)
            if letter:
                cands.append(letter)

    if not cands and row.get("answer") is not None:
        letter = normalize_letter(row["answer"])
        if letter:
            cands.append(letter)

    return cands


def extract_reasoning_from_row(row: Dict[str, Any]) -> str:
    """
    Return a reasoning string from the row when available.

    Checks (in order):
        'reasoning', 'raw_reasoning'  -> direct string fields
        'raw_samples'                 -> first non-bare-letter string in the list
    Returns '' if nothing is found.
    """
    for key in ("reasoning", "raw_reasoning"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    rs = row.get("raw_samples")
    if isinstance(rs, str) and rs.strip():
        return rs.strip()
    if isinstance(rs, list):
        for s in rs:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            # Skip bare letter tokens (e.g. "A", "B ")
            if normalize_letter(t) is not None and len(t) <= 3:
                continue
            return t
    return ""


def dedup_ordered(seq: List[str]) -> List[str]:
    """Remove duplicates from *seq* while preserving insertion order."""
    seen: set = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def atomic_joblib_dump(obj: Any, path: str) -> None:
    """Write *obj* to *path* atomically (dump to .tmp then os.replace)."""
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dicts (skips blank lines)."""
    rows: List[Dict[str, Any]] = []
    import json
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_and_tokenizer(model_path: str, dtype: str, device_map: str):
    """
    Load an AutoModelForCausalLM and its tokenizer.

    Parameters
    ----------
    model_path : str
        HuggingFace model path or local directory.
    dtype : str
        One of 'bf16', 'fp16', 'fp32'.
    device_map : str
        Passed directly to ``from_pretrained`` (e.g. 'auto').

    Returns
    -------
    model, tokenizer
    """
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device_map
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------

def _choice_block_numbered(choices: List[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(choices))


def _cand_to_answer_text(choices: List[str], cand_letter: str) -> str:
    try:
        idx = LETTERS.index(cand_letter)
    except ValueError:
        idx = -1
    if 0 <= idx < len(choices):
        return str(choices[idx])
    return cand_letter


def _build_sequence_prompt1(question: str, choices: List[str], answer_text: str) -> str:
    """prompt_id 1 / 2 : plain Q + A, no reasoning."""
    choice_block = _choice_block_numbered(choices)
    return (
        "<|user|>\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_block}\n"
        "<|assistant|>\n"
        f"{answer_text}"
    )


def _build_sequence_prompt3(
    question: str, choices: List[str], answer_text: str, reasoning: str
) -> str:
    """prompt_id 3 : Q + reasoning + FINAL ANSWER header."""
    choice_block = _choice_block_numbered(choices)
    reasoning = (reasoning or "").strip()
    return (
        "<|user|>\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_block}\n"
        "<|assistant|>\n"
        f"{reasoning}\n\n"
        "### FINAL ANSWER:\n"
        f"{answer_text}"
    )


def build_sequence_text(
    prompt_id: int,
    question: str,
    choices: List[str],
    cand_letter: str,
    reasoning: str = "",
) -> str:
    """
    Build the full input sequence for a single (question, candidate) pair.

    Parameters
    ----------
    prompt_id : int
        1 or 2  -> plain Q+A format (reasoning ignored)
        3       -> Q + reasoning + FINAL ANSWER format
    question : str
    choices : list[str]
    cand_letter : str
        One of 'A'/'B'/'C'/'D'.
    reasoning : str
        Used only when prompt_id == 3.
    """
    answer_text = _cand_to_answer_text(choices, cand_letter)
    if prompt_id in (1, 2):
        return _build_sequence_prompt1(question, choices, answer_text)
    return _build_sequence_prompt3(question, choices, answer_text, reasoning)


# ---------------------------------------------------------------------------
# Hidden-state extraction  (shared batch-level core)
# ---------------------------------------------------------------------------

def _encode_batch(
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Tokenize *texts*, pad/truncate, and move to *device*."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _last_nonpad_hidden(
    hidden: torch.Tensor,          # [B, T, H]
    attention_mask: torch.Tensor,  # [B, T]
) -> torch.Tensor:                 # [B, H]
    """
    Select the hidden state at the last non-padding position for each sample.

    This is the correct approach when right-padding is used in batch inference,
    as opposed to a fixed index such as -1 (which would point to a pad token).
    """
    lengths  = attention_mask.sum(dim=1)
    last_pos = (lengths - 1).clamp(min=0)
    bsz, _, hdim = hidden.shape
    idx = last_pos.view(bsz, 1, 1).expand(bsz, 1, hdim)
    return hidden.gather(dim=1, index=idx).squeeze(1)


@torch.no_grad()
def extract_hidden_all_layers(
    model,
    tokenizer,
    samples: List[Tuple[str, List[str], str, str]],
    batch_size: int,
    max_length: int,
    dtype: str,
    prompt_id: int,
) -> List[np.ndarray]:
    """
    Extract last non-pad hidden states for **every** transformer layer.

    Parameters
    ----------
    model, tokenizer
        Loaded HuggingFace model and tokenizer.
    samples : list of (question, choices, cand_letter, reasoning)
    batch_size, max_length, dtype, prompt_id
        Forwarded to tokenizer / model.

    Returns
    -------
    list of length L (transformer layers, embedding layer excluded).
    Each element is np.ndarray of shape [N, H].
    """
    model.eval()
    device    = next(model.parameters()).device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)

    layer_chunks: Optional[List[List[np.ndarray]]] = None

    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting hidden states (all layers)"):
        batch_samples = samples[i : i + batch_size]
        texts = [
            build_sequence_text(prompt_id, q, choices, cand, rsn)
            for q, choices, cand, rsn in batch_samples
        ]

        batch = _encode_batch(tokenizer, texts, max_length, device)

        with torch.amp.autocast(
            "cuda", enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32
        ):
            out = model(**batch, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states
        del out

        if layer_chunks is None:
            layer_chunks = [[] for _ in range(len(hs) - 1)]

        for li in range(1, len(hs)):
            vec = _last_nonpad_hidden(hs[li], batch["attention_mask"])
            layer_chunks[li - 1].append(vec.float().cpu().numpy())

    assert layer_chunks is not None, "No samples were processed."
    return [np.concatenate(chunks, axis=0) for chunks in layer_chunks]


@torch.no_grad()
def extract_hidden_single_layer(
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
    Extract last non-pad hidden states at **one** layer for every candidate in
    every row.

    Parameters
    ----------
    rows : list of raw data dicts (each row may contain multiple candidates).
    layer_index : int
        1-indexed transformer layer (HF hidden_states index).

    Returns
    -------
    row_info : list aligned with *rows*.
               None   -> row had no scoreable candidates.
               dict   -> {"_flat_indices": [...], "_flat_candidates": [...]}
    X_flat   : np.ndarray [N_flat, H]
    """
    model.eval()
    device    = next(model.parameters()).device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)

    # Build flat list: (row_index, cand_letter, question, choices, reasoning)
    flat: List[Tuple[int, str, str, List[str], str]] = []
    for ri, r in enumerate(rows):
        q       = r.get("question")
        choices = r.get("choices")
        if not isinstance(q, str) or not isinstance(choices, list) or not choices:
            continue
        cands = dedup_ordered(get_generated_candidates(r))
        if not cands:
            continue
        rsn = extract_reasoning_from_row(r)
        for c in cands:
            flat.append((ri, c, q, choices, rsn))

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
            for _, cand, q, choices, rsn in batch_flat
        ]

        batch = _encode_batch(tokenizer, texts, max_length, device)

        with torch.amp.autocast(
            "cuda", enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32
        ):
            out = model(**batch, output_hidden_states=True, use_cache=False)

        h = out.hidden_states[layer_index]
        del out

        vec = _last_nonpad_hidden(h, batch["attention_mask"])
        feats.append(vec.float().cpu().numpy())

    X_flat = np.concatenate(feats, axis=0)

    # Map flat positions back to per-row indices
    grouped: Dict[int, List[int]] = {}
    for j, (ri, *_) in enumerate(flat):
        grouped.setdefault(ri, []).append(j)

    for ri, idxs in grouped.items():
        row_info[ri] = {
            "_flat_indices":    idxs,
            "_flat_candidates": [flat[j][1] for j in idxs],
        }

    return row_info, X_flat