# =========================
# Definition of A / B accuracy
# =========================
# A: accuracy over all samples (paper version)
#    - all samples are included in the denominator (total)
#    - even samples with answer=None are counted
#
# B: accuracy excluding null answers
#    - samples with answer=None are excluded from evaluation
#    - accuracy is computed only on samples with valid answers
# =========================

import json
import argparse
from collections import defaultdict

VALID_SOURCES = {"greedy", "sampling"}


def extract_question_id(full_id: str) -> str:
    return full_id.rsplit("_", 1)[0]


def run_acc(input_jsonl: str, output_jsonl: str):

    # Accumulate per question_id
    question_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "null": 0,
        }
    )

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if obj.get("source") not in VALID_SOURCES:
                continue

            question_id = extract_question_id(obj["id"])
            count = obj.get("count", 0)
            label = obj.get("label", 0)
            answer = obj.get("answer")

            # Total count (for A)
            question_stats[question_id]["total"] += count

            # null count
            if answer is None:
                question_stats[question_id]["null"] += count
            else:
                # correct count
                if label == 1:
                    question_stats[question_id]["correct"] += count

    # =========================
    # Save and compute macro metrics
    # =========================

    macro_A_list = []
    macro_B_list = []

    with open(output_jsonl, "w", encoding="utf-8") as f:

        for qid, stats in question_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            null_count = stats["null"]

            # A
            acc_A = correct / total if total > 0 else 0.0

            # B: excluding null values
            effective_total = total - null_count
            acc_B = correct / effective_total if effective_total > 0 else 0.0

            macro_A_list.append(acc_A)
            macro_B_list.append(acc_B)

            f.write(json.dumps({
                "question_id": qid,
                "total_count": total,
                "correct_count": correct,
                "num_answer_null": null_count,
                "accuracy_A": acc_A,
                "accuracy_B_exclude_null": acc_B
            }, ensure_ascii=False) + "\n")

        # ===== macro overall =====
        macro_A = sum(macro_A_list) / len(macro_A_list) if macro_A_list else 0.0
        macro_B = sum(macro_B_list) / len(macro_B_list) if macro_B_list else 0.0

        f.write(json.dumps({
            "question_id": "__OVERALL_MACRO__",
            "num_questions": len(question_stats),
            "macro_accuracy_A": macro_A,
            "macro_accuracy_B": macro_B
        }, ensure_ascii=False) + "\n")

    print("✅ Macro accuracy computation done")
    print(f"Total questions: {len(question_stats)}")
    print(f"Macro Accuracy A: {macro_A}")
    print(f"Macro Accuracy B: {macro_B}")
    print(f"Saved to: {output_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_acc(args.input_jsonl, args.output_jsonl)
