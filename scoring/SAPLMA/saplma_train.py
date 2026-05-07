"""
Portions of this implementation are adapted from TruthTorchLM / SAPLMA.

TruthTorchLM is licensed under the MIT License.
Copyright (c) 2024 Yavuz Faruk Bakman

See THIRD_PARTY_LICENSES/TruthTorchLM_LICENSE.txt for the full license text.

[SAPLMA probe - TRAIN]

Layer sweep -> pick best layer by dev AUROC -> save best probe only.

Data assumptions (each JSONL row):
  - id          : str
  - question    : str
  - choices     : list[str]   (length 4 assumed; A/B/C/D)
  - gold_answer : int (0-3)  OR  str ("A"/"B"/"C"/"D")

Feature:
  - last non-pad token hidden state of (prompt + candidate_answer_text)
  - extracted per transformer layer (1..L), skipping the embedding layer

Probe:
  - StandardScaler + MLPClassifier
  - class imbalance handled via inverse-frequency sample weights
  - best layer selected by dev AUROC
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import (
    LETTERS,
    atomic_joblib_dump,
    base_qid,
    extract_hidden_all_layers,
    gold_letter,
    load_jsonl,
    load_model_and_tokenizer,
    set_seed,
    derive_paths,
    derive_prompt_id,
)

# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_all_choices_per_question(
    rows: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[
    List[Tuple[str, List[str], str, str]],  # (question, choices, cand_letter, reasoning)
    np.ndarray,   # y    [N] int64
    np.ndarray,   # w    [N] float32  inverse-frequency weights
    np.ndarray,   # bids [N] object   base question ids
]:
    """
    For every base question, enumerate ALL answer choices (A/B/C/D) as
    candidates. gold_answer -> label 1, others -> label 0.

    This is independent of what the model actually generated, so:
      - positive rate is always exactly 1 / num_choices (≈ 25%)
      - no data leakage from model outputs into the probe training signal
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        qid = r.get("id")
        if not isinstance(qid, str):
            continue
        grouped.setdefault(base_qid(qid), []).append(r)

    samples: List[Tuple[str, List[str], str, str]] = []
    ys:   List[int] = []
    bids: List[str] = []

    for bid, grp in grouped.items():
        # question text
        q = next(
            (r["question"] for r in grp
             if isinstance(r.get("question"), str) and r["question"]),
            None,
        )
        if not q:
            continue

        # choices list
        choices = grp[0].get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        # gold answer letter
        g = gold_letter(grp[0])
        if g is None:
            continue

        # Enumerate every valid choice letter
        valid_letters = LETTERS[: len(choices)]  # ["A","B","C","D"] for 4-choice
        for c in valid_letters:
            samples.append((q, choices, c, ""))
            ys.append(1 if c == g else 0)
            bids.append(bid)

    if not samples:
        return (
            [],
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.asarray([], dtype=object),
        )

    y_arr    = np.asarray(ys, dtype=np.int64)
    bids_arr = np.asarray(bids, dtype=object)

    # Inverse-frequency sample weights (balanced pos/neg contribution)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    w_pos = 0.5 / max(1, n_pos)
    w_neg = 0.5 / max(1, n_neg)
    w_arr = np.where(y_arr == 1, w_pos, w_neg).astype(np.float32)

    return samples, y_arr, w_arr, bids_arr


# ---------------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    if args.model_path is None or args.out_dir is None:
        derived_model, derived_out = derive_paths(args.data_jsonl)
        if args.model_path is None:
            args.model_path = derived_model
            print(f"[AUTO] model_path = {args.model_path}", flush=True)
        if args.out_dir is None:
            args.out_dir = derived_out
            print(f"[AUTO] out_dir    = {args.out_dir}", flush=True)
    prompt_id = derive_prompt_id(args.data_jsonl)
    print(f"[AUTO] prompt_id  = {prompt_id}", flush=True)

    set_seed(args.seed)

    rows = load_jsonl(args.data_jsonl)

    samples, y, w, bids = build_all_choices_per_question(
        rows,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError(
            "No training samples built. "
            "Check that rows have id / question / choices / gold_answer fields."
        )

    # Question-level train/dev split (prevents data leakage)
    uniq_q = sorted(set(bids.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(uniq_q)

    n_train_q = int(len(uniq_q) * args.train_ratio)
    train_set = set(uniq_q[:n_train_q])

    train_mask = np.asarray([b in train_set for b in bids], dtype=bool)
    dev_mask   = ~train_mask

    print(
        f"[INFO] samples={len(samples)}  unique_questions={len(uniq_q)}\n"
        f"[INFO] train_q={len(train_set)}  dev_q={len(uniq_q) - len(train_set)}\n"
        f"[INFO] class counts: pos={(y == 1).sum()}  neg={(y == 0).sum()}",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.dtype, args.device_map
    )

    feats_by_layer = extract_hidden_all_layers(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        prompt_id=1, ### hard coding
    )

    del model  # free GPU memory

    os.makedirs(args.out_dir, exist_ok=True)
    probes_dir = os.path.join(args.out_dir, "probes")
    os.makedirs(probes_dir, exist_ok=True)

    hidden_sizes = (
        tuple(int(x) for x in args.mlp_hidden.split(","))
        if args.mlp_hidden
        else (256,)
    )

    dev_metrics: Dict[str, Dict[str, float]] = {}
    best_layer: Optional[int] = None
    best_score = -1.0
    best_pack  = None

    for layer_idx, X in enumerate(tqdm(feats_by_layer, desc="Training probes"), start=1):
        Xtr, ytr, wtr = X[train_mask], y[train_mask], w[train_mask]
        Xdv, ydv      = X[dev_mask],   y[dev_mask]

        scaler  = StandardScaler()
        Xtr_s   = scaler.fit_transform(Xtr)
        Xdv_s   = scaler.transform(Xdv)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=args.mlp_activation,
            solver=args.mlp_solver,
            alpha=args.mlp_alpha,
            batch_size=min(args.mlp_batch_size, Xtr_s.shape[0]),
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            early_stopping=bool(args.mlp_early_stopping),
            validation_fraction=args.mlp_val_fraction,
            n_iter_no_change=args.mlp_patience,
            random_state=args.seed,
            verbose=False,
        )

        try:
            clf.fit(Xtr_s, ytr, sample_weight=wtr)
        except TypeError:
            print(
                "[WARN] This sklearn version does not support sample_weight in "
                "MLPClassifier.fit — training without weights.",
                flush=True,
            )
            clf.fit(Xtr_s, ytr)

        prob_dv = clf.predict_proba(Xdv_s)[:, 1]

        try:
            auc = float(roc_auc_score(ydv, prob_dv))
        except ValueError:
            auc = float("nan")

        dev_metrics[str(layer_idx)] = {"auroc": auc}
        print(f"  layer {layer_idx:3d}  dev_auroc={auc:.4f}", flush=True)

        score = auc if not np.isnan(auc) else -1.0
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_pack  = {"layer": layer_idx, "scaler": scaler, "clf": clf}

    with open(os.path.join(args.out_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

    if best_pack is None or best_layer is None:
        raise RuntimeError("Probe training failed: no valid layer was selected.")

    best_path = os.path.join(probes_dir, f"layer_{best_layer}.joblib")
    atomic_joblib_dump(best_pack, best_path)
    print(
        f"[SAVED] best probe  layer={best_layer}  auroc={best_score:.4f}  -> {best_path}",
        flush=True,
    )

    meta = {
        "model_path":           args.model_path,
        "data_jsonl":           args.data_jsonl,
        "seed":                 args.seed,
        "train_ratio":          args.train_ratio,
        "prompt_id":            prompt_id,
        "batch_size":           args.batch_size,
        "max_length":           args.max_length,
        "dtype":                args.dtype,
        "num_layers":           len(feats_by_layer),
        "num_samples":          len(samples),
        "num_unique_questions": len(uniq_q),
        "best_layer":           int(best_layer),
        "best_auroc":           float(best_score),
        "best_probe_path":      best_path,
        "feature": (
            "last_nonpad_hidden_of(prompt + answer_text) [all choices A~D]"
            + (" with reasoning+FINAL_ANSWER" if prompt_id == 3 else "")
        ),
        "labeling":        "1 if cand_letter == gold_answer else 0",
        "candidate_source": "all_choices_A_to_D",
        "imbalance":       "inverse-frequency sample weights",
        "mlp": {
            "hidden_layer_sizes":  list(hidden_sizes),
            "activation":          args.mlp_activation,
            "solver":              args.mlp_solver,
            "alpha":               args.mlp_alpha,
            "batch_size":          args.mlp_batch_size,
            "learning_rate_init":  args.mlp_lr,
            "max_iter":            args.mlp_max_iter,
            "early_stopping":      bool(args.mlp_early_stopping),
            "validation_fraction": args.mlp_val_fraction,
            "n_iter_no_change":    args.mlp_patience,
        },
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a SAPLMA-style MLP probe on hidden states (all-choices edition)."
    )

    ap.add_argument("--data_jsonl",  required=True, help="Input JSONL file.")
    ap.add_argument("--model_path",  default=None,
                    help="HuggingFace model path. Auto-derived from data_jsonl if omitted.")
    ap.add_argument("--out_dir",     default=None,
                    help="Output directory. Auto-derived from data_jsonl if omitted.")

    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="Fraction of unique questions used for training (rest = dev).")
    ap.add_argument("--seed",        type=int,   default=42)

    ap.add_argument("--batch_size",  type=int,   default=16)
    ap.add_argument("--max_length",  type=int,   default=512)
    ap.add_argument("--dtype",       choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map",  default="auto")

    ap.add_argument("--mlp_hidden",         type=str,   default="256",
                    help='Comma-separated hidden layer sizes, e.g. "512,256".')
    ap.add_argument("--mlp_activation",     type=str,   default="relu",
                    choices=["identity", "logistic", "tanh", "relu"])
    ap.add_argument("--mlp_solver",         type=str,   default="adam",
                    choices=["lbfgs", "sgd", "adam"])
    ap.add_argument("--mlp_alpha",          type=float, default=1e-4)
    ap.add_argument("--mlp_batch_size",     type=int,   default=128)
    ap.add_argument("--mlp_lr",             type=float, default=1e-3)
    ap.add_argument("--mlp_max_iter",       type=int,   default=200)
    ap.add_argument("--mlp_early_stopping", action="store_true")
    ap.add_argument("--mlp_val_fraction",   type=float, default=0.1)
    ap.add_argument("--mlp_patience",       type=int,   default=10)

    return ap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd_train(build_parser().parse_args())
