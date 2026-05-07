import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

def summarize_subset(rows, name="all"):
    if len(rows) == 0:
        print(f"\n[{name}] no valid rows")
        return

    y_true = np.array([r["label"] for r in rows], dtype=int)
    y_score = np.array([r["mars_score"] for r in rows], dtype=float)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    print(f"\n===== {name} =====")
    print(f"rows        : {len(rows)}")
    print(f"label=1 cnt : {(y_true == 1).sum()}")
    print(f"label=0 cnt : {(y_true == 0).sum()}")

    if len(pos_scores) > 0:
        print(f"mean(score|y=1): {pos_scores.mean():.6f}")
        print(f"med (score|y=1): {np.median(pos_scores):.6f}")
    if len(neg_scores) > 0:
        print(f"mean(score|y=0): {neg_scores.mean():.6f}")
        print(f"med (score|y=0): {np.median(neg_scores):.6f}")

    if len(set(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_score)
        auc_flip = roc_auc_score(y_true, -y_score)
        print(f"AUROC(score) : {auc:.4f} ({auc*100:.2f}%)")
        print(f"AUROC(-score): {auc_flip:.4f} ({auc_flip*100:.2f}%)")
    else:
        print("AUROC unavailable: only one class present.")


def load_rows(input_path):
    rows = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = ex.get("source")
            if source not in {"greedy", "sampling"}:
                continue

            label = ex.get("label")
            answer = ex.get("answer")
            mars_score = ex.get("mars_score")

            if label is None:
                continue

            # A version
            if answer is None:
                mars_score = 0.0
            if mars_score is None:
                mars_score = 0.0

            rows.append({
                "source": source,
                "label": int(label),
                "mars_score": float(mars_score),
                "answer": answer,
                "raw_len": len(ex.get("raw_samples") or ""),
            })

    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python avg_A.py <input_jsonl>")
        raise SystemExit(1)

    input_jsonl = sys.argv[1]

    if not os.path.exists(input_jsonl):
        print(f"Error: Input file '{input_jsonl}' not found.")
        raise SystemExit(1)

    rows = load_rows(input_jsonl)

    summarize_subset(rows, "all")

    raw_lens = np.array([r["raw_len"] for r in rows], dtype=int)
    if len(raw_lens) > 0:
        print("\n===== reasoning length =====")
        print(f"mean raw_samples length: {raw_lens.mean():.2f}")
        print(f"median raw_samples length: {np.median(raw_lens):.2f}")