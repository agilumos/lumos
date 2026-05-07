'''
python SV_confusion_matrix.py \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling_judge.jsonl

'''
# =========================
# Definition of SV_A / SV_B
# =========================
# SV_A: include null answers
#    - all samples are included in the confusion matrix
#    - samples with answer=None are also counted
#    - reflects overall performance including generation failures
#
# SV_B: exclude null answers
#    - samples with answer=None are excluded from the confusion matrix
#    - only samples with valid (non-null) answers are evaluated
#    - reflects classification performance given a valid response
# =========================

import json
import argparse
from collections import defaultdict
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SV_A and SV_B confusion matrices"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input labeling jsonl"
    )
    return parser.parse_args()


def safe_div(num, den):
    return num / den if den != 0 else 0.0


def build_output_paths(input_jsonl: str):
    base = input_jsonl.replace("_labeling_judge.jsonl", "")
    return (
        base + "_labeling_SVexp_A.jsonl",
        base + "_labeling_SVexp_B.jsonl",
    )


def init_stats():
    return defaultdict(lambda: {
        "TP": 0, "FP": 0, "FN": 0, "TN": 0,
        "num_answer_null": 0
    })


def update_confusion(stats, qid, label, judge, count):
    if label == 1 and judge == "A":
        stats[qid]["TP"] += count
    elif label == 0 and judge == "A":
        stats[qid]["FP"] += count
    elif label == 1 and judge == "B":
        stats[qid]["FN"] += count
    elif label == 0 and judge == "B":
        stats[qid]["TN"] += count


def compute_metrics(stats):
    results = []
    accs, precs, recs, f1s = [], [], [], []

    for qid, c in stats.items():
        TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
        num_null = c["num_answer_null"]

        total = TP + FP + FN + TN

        accuracy  = safe_div(TP + TN, total)
        precision = safe_div(TP, TP + FP)
        recall    = safe_div(TP, TP + FN)
        f1        = safe_div(2 * precision * recall, precision + recall)

        accs.append(accuracy)
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)

        results.append({
            "question_id": qid,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "num_answer_null": num_null,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_avg = {
        "question_id": "MACRO_AVERAGE",
        "accuracy": sum(accs) / len(accs) if accs else 0.0,
        "precision": sum(precs) / len(precs) if precs else 0.0,
        "recall": sum(recs) / len(recs) if recs else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }

    return results, macro_avg


def write_jsonl(path, results, macro_avg):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(macro_avg) + "\n")


def main():
    args = parse_args()
    input_jsonl = args.input_jsonl

    output_A, output_B = build_output_paths(input_jsonl)

    print(f"▶ Input : {input_jsonl}")
    print(f"▶ Output SV_A: {output_A}")
    print(f"▶ Output SV_B: {output_B}")

    stats_A = init_stats()
    stats_B = init_stats()

    # -------------------------
    # Read jsonl
    # -------------------------
    with open(input_jsonl, "r") as f:
        for line in f:
            ex = json.loads(line)

            qid = "_".join(ex["id"].split("_")[:-1])
            label = ex["label"]
            judge = ex["judge_answer"]
            answer = ex.get("answer")
            count = ex.get("count", 1)

            # -------------------------
            # SA_A (include null)
            # -------------------------
            if answer is None:
                stats_A[qid]["num_answer_null"] += count
            update_confusion(stats_A, qid, label, judge, count)

            # -------------------------
            # SA_B (except null)
            # -------------------------
            if answer is None:
                stats_B[qid]["num_answer_null"] += count
                continue  # Exclude null values from the confusion matrix

            update_confusion(stats_B, qid, label, judge, count)

    # -------------------------
    # Compute metrics
    # -------------------------
    results_A, macro_A = compute_metrics(stats_A)
    results_B, macro_B = compute_metrics(stats_B)

    # -------------------------
    # Write outputs
    # -------------------------
    write_jsonl(output_A, results_A, macro_A)
    write_jsonl(output_B, results_B, macro_B)

    print(f"✅ Done.")
    print(f"   SV_A questions: {len(results_A)}")
    print(f"   SV_B questions: {len(results_B)}")


if __name__ == "__main__":
    main()