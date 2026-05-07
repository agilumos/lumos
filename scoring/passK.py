import json
import re
import csv
import math

# ==============================
# Settings
# ==============================
INPUT_JSONL = "/mnt/PK/Inside-out/reasoning/32B-SFT/metamath/data/math/metamath-owmfilter/metamath_recovered.jsonl"
OUTPUT_CSV = "/mnt/PK/Inside-out/reasoning/32B-SFT/metamath/data/math/metamath-owmfilter/pass_at_k_results_recovered.csv"

K_LIST = [1, 2, 4, 8]

# ==============================
# Utils
# ==============================

def extract_gold_answer(gold_text: str) -> str:
    gold_text = gold_text.strip()

    # If text contains "####", use the content after it first
    m = re.search(r"####\s*(.+)", gold_text)
    if m:
        return m.group(1).strip()

    # Otherwise, treat the entire text as gold
    return gold_text

def normalize_answer(ans: str) -> str:
    ans = ans.strip()

    # Remove whitespace
    ans = ans.replace(" ", "")

    # Normalize LaTeX-style formatting
    ans = ans.replace(r"\cdot", "")
    ans = ans.replace("{", "").replace("}", "")

    # x^4 vs x^{4}
    ans = re.sub(r"x\^\{(\d+)\}", r"x^\1", ans)

    # Standardize π notation
    ans = ans.replace("π", r"\pi")

    return ans


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

# ==============================
# Main logic
# ==============================

def unbiased_pass_at_k(n, c, k):
    """
    n: total samples
    c: number of correct samples
    k: pass@k
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1 - (math.comb(n - c, k) / math.comb(n, k))


def compute_pass_at_k(jsonl_path: str):
    pass_sum = {k: 0.0 for k in K_LIST}
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            gold = extract_gold_answer(ex["gold_answer"])
            samples = ex["samples"]
            answers = [s["answer"] for s in samples]

            n = len(answers)
            c = sum(is_correct(a, gold) for a in answers)

            total += 1

            for k in K_LIST:
                if n >= k:
                    pass_sum[k] += unbiased_pass_at_k(n, c, k)

    pass_scores = {
        f"pass@{k}": pass_sum[k] / total
        for k in K_LIST
    }

    return pass_scores, total


def save_to_csv(output_path, total, pass_scores):
    header = ["total_questions"] + list(pass_scores.keys())
    row = [total] + [f"{pass_scores[k]:.4f}" for k in pass_scores]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    scores, total = compute_pass_at_k(INPUT_JSONL)

    save_to_csv(OUTPUT_CSV, total, scores)

    print("✅ pass@K evaluation finished")
    print(f"Total questions: {total}")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
    print(f"📁 Saved to: {OUTPUT_CSV}")
