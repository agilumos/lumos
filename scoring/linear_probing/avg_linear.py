import json
import os
from statistics import mean

########################################
# Settings
########################################

input_jsonl = os.environ.get(
    "INPUT_JSONL",
    "data/results.jsonl"
)

output_jsonl = os.environ.get(
    "OUTPUT_JSONL",
    "outputs/output_accuracy_probe.jsonl"
)


def extract_question_id(full_id: str) -> str:
    # Extract the base question ID
    # Example: "Robustness (evolution)_1921_0" -> "Robustness (evolution)_1921"
    return full_id.rsplit("_", 1)[0]


def process_accuracy(input_path, output_path):
    seen_questions = set()

    # Store results for aggregation
    results = []

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            # We calculate accuracy using only greedy generations
            if ex.get("source") != "greedy":
                continue

            question_id = extract_question_id(ex["id"])

            # Process each question only once
            if question_id in seen_questions:
                continue

            # Skip rows missing required fields
            label = ex.get("label")
            probe_score = ex.get("probe_score")

            if label is None or probe_score is None:
                continue

            seen_questions.add(question_id)

            data_point = {
                "question_id": question_id,
                "label": label,          # 1 for correct, 0 for incorrect
                "probe_score": probe_score
            }
            results.append(data_point)

            # Write per-question result
            fout.write(json.dumps(data_point, ensure_ascii=False) + "\n")

        # Calculate summary statistics
        if not results:
            print("No valid greedy samples found.")
            return

        total_count = len(results)
        correct_count = sum(1 for r in results if r["label"] == 1)

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_probe_score = mean(r["probe_score"] for r in results)

        # Optional: compare probe scores for correct vs. incorrect predictions
        correct_scores = [r["probe_score"] for r in results if r["label"] == 1]
        wrong_scores = [r["probe_score"] for r in results if r["label"] == 0]

        avg_probe_correct = mean(correct_scores) if correct_scores else 0.0
        avg_probe_wrong = mean(wrong_scores) if wrong_scores else 0.0

        summary = {
            "type": "summary",
            "total_questions": total_count,
            "accuracy": accuracy,
            "avg_probe_score": avg_probe_score,
            "avg_probe_score_correct": avg_probe_correct,
            "avg_probe_score_incorrect": avg_probe_wrong
        }

        # Write summary at the end
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("-" * 30)
    print("Processing done")
    print(f"Total Questions: {total_count}")
    print(f"Model Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Avg Probe Score: {avg_probe_score:.4f}")
    print("-" * 30)
    print(f"Avg Probe (Correct):   {avg_probe_correct:.4f}")
    print(f"Avg Probe (Incorrect): {avg_probe_wrong:.4f}")


if __name__ == "__main__":
    if os.path.exists(input_jsonl):
        process_accuracy(input_jsonl, output_jsonl)
    else:
        print(f"Error: Input file '{input_jsonl}' not found.")