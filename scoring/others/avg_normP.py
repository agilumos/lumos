# =========================
# Definition of A / B metrics
# =========================
# A: overall average across all questions (paper version)
#    - all questions contribute equally
#    - questions with no correct samples are counted as 0
#
# B: average over correct questions only
#    - only questions with at least one correct sample are included
#    - computed as the mean over correct-question averages
# =========================

import json
from statistics import mean

########################################
# Settings
########################################
VALID_SOURCES = {"greedy", "sampling"}

########################################
# Utils
########################################

def extract_question_id(full_id: str) -> str:
    return full_id.rsplit("_", 1)[0]

########################################
# Main
########################################

def run_normP(input_path, output_path):
    question_dict = {}

    # 1. First, collect correct samples per question
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)

            # filtering
            if ex.get("source") not in VALID_SOURCES:
                continue

            question_id = extract_question_id(ex["id"])
            p_norm = ex.get("P_norm(a|q)")
            label = ex.get("label")  # label = 1 indicates correct sample

            if p_norm is None:
                continue

            if question_id not in question_dict:
                question_dict[question_id] = {
                    "correct_p": [],
                }

            if label == 1:
                question_dict[question_id]["correct_p"].append(p_norm)

    # 2. Compute per-question averages
    total_Q = len(question_dict)
    correct_Q = 0
    sum_p = 0.0

    with open(output_path, "w", encoding="utf-8") as fout:

        for question_id, data in question_dict.items():
            correct_p_list = data["correct_p"]

            if len(correct_p_list) > 0:
                is_correct_Q = True
                avg_p = mean(correct_p_list)
                correct_Q += 1
                sum_p += avg_p
            else:
                is_correct_Q = False
                avg_p = 0.0  # If no correct samples exist, contribute 0

            out = {
                "question_id": question_id,
                "avg_P_norm(a|q)": avg_p,
                "is_correct_Q": is_correct_Q,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

        # 3. Compute summary metrics
        B = sum_p / correct_Q if correct_Q > 0 else 0.0
        A = sum_p / total_Q if total_Q > 0 else 0.0

        summary = {
            "type": "summary",
            "total_questions": total_Q,
            "correct_questions": correct_Q,
            "A_overall_avg": A,          # Based on all questions
            "B_correct_only_avg": B,     # Based only on correct questions
        }

        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("✅ Done")
    print(f"Total questions: {total_Q}")
    print(f"Correct questions: {correct_Q}")
    print(f"A (overall avg): {A}")
    print(f"B (correct-only avg): {B}")

########################################
# run
########################################

if __name__ == "__main__":
    run_normP(input_jsonl, output_jsonl)
