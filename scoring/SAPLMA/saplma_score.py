"""
Portions of this implementation are adapted from TruthTorchLM / SAPLMA.

TruthTorchLM is licensed under the MIT License.
Copyright (c) 2024 Yavuz Faruk Bakman

See THIRD_PARTY_LICENSES/TruthTorchLM_LICENSE.txt for the full license text.

[SAPLMA probe - SCORE]

Outputs:
  {stem}_final_A.jsonl        per-question results (mode A)
  {stem}_final_A_summary.json accuracy + auroc     (mode A)
  {stem}_final_B.jsonl        per-question results (mode B)
  {stem}_final_B_summary.json accuracy + auroc     (mode B)
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_question_id(full_id: str) -> str:
    return full_id.rsplit("_", 1)[0]


def get_saplma_result(ex: dict) -> Optional[Tuple[bool, float]]:
    """
    Extract (is_correct, gold_p_true) from saplma_best.

    is_correct  : pred_letter == gold_letter
    gold_p_true : p_true of the gold answer choice (for AUROC reference)

    Returns None when saplma_best is absent or malformed.
    """
    sb = ex.get("saplma_best")
    if not isinstance(sb, dict):
        return None

    is_correct = sb.get("is_correct")
    if is_correct is None:
        return None

    # gold_p_true: p_true of the gold letter among per_candidate
    gold_letter = sb.get("gold_letter")
    gold_p_true: Optional[float] = None
    pc = sb.get("per_candidate")
    if isinstance(pc, list) and gold_letter:
        for item in pc:
            if isinstance(item, dict) and item.get("cand") == gold_letter:
                gold_p_true = float(item["p_true"])
                break

    # fallback: use agg_max if gold_p_true is not found
    if gold_p_true is None:
        gold_p_true = sb.get("agg_max")
        if gold_p_true is not None:
            gold_p_true = float(gold_p_true)

    if gold_p_true is None:
        return None

    return bool(is_correct), gold_p_true


def _write_outputs(
    records: List[Tuple[bool, float, str, int]],
    # (is_correct, gold_p_true, question_id, behavioral_label)
    out_jsonl: str,
    out_summary: str,
    mode: str,
) -> None:
    """Write per-question JSONL and summary JSON for one mode."""
    n_total   = len(records)
    n_correct = sum(r[0] for r in records)
    accuracy  = n_correct / n_total if n_total > 0 else None

    # AUROC: gold_p_true vs behavioral_label (label=1 if model got it right)
    # This measures whether probe confidence aligns with behavioral correctness
    b_labels  = [r[3] for r in records]
    p_trues   = [r[1] for r in records]
    n_pos = sum(b_labels)
    n_neg = n_total - n_pos
    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(b_labels, p_trues))
    else:
        auroc = None
        print(
            f"[WARN] Mode {mode}: AUROC not computable — "
            f"pos={n_pos}, neg={n_neg}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for is_correct, gold_p_true, qid, b_label in records:
            f.write(json.dumps(
                {
                    "question_id":   qid,
                    "is_correct":    is_correct,
                    "gold_p_true":   gold_p_true,
                    "behavioral_label": b_label,
                },
                ensure_ascii=False,
            ) + "\n")

    summary = {
        "mode":            mode,
        "num_questions":   n_total,
        "rank_accuracy":   round(accuracy, 6) if accuracy is not None else None,
        "num_correct":     n_correct,
        "auroc_ref":       round(auroc, 6) if auroc is not None else None,
        "auroc_note":      (
            "AUROC of gold_p_true vs behavioral correctness label. "
            "Primary metric is rank_accuracy."
        ),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"  [Mode {mode}] questions={n_total}  "
        f"rank_accuracy={accuracy:.4f}  auroc_ref={auroc}",
        flush=True,
    )
    print(f"           jsonl   -> {out_jsonl}")
    print(f"           summary -> {out_summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_jsonl(input_path: str) -> None:
    stem = input_path.removesuffix(".jsonl")

    out_jsonl_A   = f"{stem}_final_A.jsonl"
    out_summary_A = f"{stem}_final_A_summary.json"
    out_jsonl_B   = f"{stem}_final_B.jsonl"
    out_summary_B = f"{stem}_final_B_summary.json"

    seen_questions: set = set()

    # (is_correct, gold_p_true, question_id, behavioral_label)
    records_A: List[Tuple[bool, float, str, int]] = []
    records_B: List[Tuple[bool, float, str, int]] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)

            # Exclude gold rows: use only greedy / sampling outputs
            if ex.get("source") not in {"greedy", "sampling"}:
                continue

            question_id = extract_question_id(ex["id"])

            # Use only the first row per question_id
            # (since A~D p_true values are identical across rows)
            if question_id in seen_questions:
                continue

            # behavioral_label: whether the model got this row correct
            b_label = ex.get("label")
            if b_label not in (0, 1):
                continue

            result = get_saplma_result(ex)  # None when saplma_best absent

            seen_questions.add(question_id)

            if result is not None:
                is_correct, gold_p_true = result
                records_A.append((is_correct, gold_p_true, question_id, b_label))
                records_B.append((is_correct, gold_p_true, question_id, b_label))
            else:
                # Mode A: if no saplma, treat as incorrect with gold_p_true=0.0
                records_A.append((False, 0.0, question_id, b_label))
                # Mode B: skip

    print(f"\n[SCORE] input: {input_path}", flush=True)
    _write_outputs(records_A, out_jsonl_A, out_summary_A, mode="A")
    _write_outputs(records_B, out_jsonl_B, out_summary_B, mode="B")
    print("Done", flush=True)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Compute SAPLMA rank accuracy in two modes:\n"
            "  A) saplma=None rows treated as incorrect (is_correct=False)\n"
            "  B) saplma=None rows skipped\n"
            "Output paths are derived automatically from --input_jsonl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input_jsonl", required=True,
        help="Path to the scored.jsonl produced by saplma_infer.py.",
    )
    return ap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = build_parser().parse_args()
    process_jsonl(args.input_jsonl)
