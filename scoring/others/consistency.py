import json
from collections import defaultdict

########################################
# Settings
########################################
VALID_SOURCES = {"greedy", "sampling"}

########################################
# util
########################################

def extract_question_id(full_id: str) -> str:
    return full_id.rsplit("_", 1)[0]

########################################
# main
########################################

def run_cons(input_path, output_path):
    """
    values[qid][answer] = accumulated count
    """
    values = defaultdict(lambda: defaultdict(int))

    # ---------- collect ----------
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            if ex.get("source") not in VALID_SOURCES:
                continue

            qid = extract_question_id(ex["id"])
            answer = ex.get("answer")
            count = ex.get("count", 0)

            if answer is None:
                continue

            values[qid][answer] += count

    # ---------- compute & write ----------
    consistencies = []

    with open(output_path, "w", encoding="utf-8") as fout:
        for qid, answer_counts in values.items():
            total_count = sum(answer_counts.values())
            if total_count == 0:
                continue

            max_answer = max(answer_counts, key=lambda a: answer_counts[a])
            max_count = answer_counts[max_answer]
            consistency = max_count / total_count

            consistencies.append(consistency)

            out = {
                "question_id": qid,
                "total_count": total_count,
                "most_common_answer": max_answer,
                "most_common_count": max_count,
                "consistency": consistency,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

        # ---------- global average ----------
        global_out = {
            "question_id": "__GLOBAL__",
            "avg_consistency": sum(consistencies) / len(consistencies) if consistencies else None,
            "num_questions": len(consistencies),
        }

        fout.write(json.dumps(global_out, ensure_ascii=False) + "\n")

    print("✅ Done")
    print(f"Total questions: {len(consistencies)}")
    print(f"Global avg consistency: {global_out['avg_consistency']}")
    print(f"Saved to: {output_path}")

########################################
# run
########################################

if __name__ == "__main__":
    run_cons(INPUT_JSONL, OUTPUT_JSONL)