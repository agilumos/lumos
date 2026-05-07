# =========================
# Definition of A / B (accuracy metrics)
# =========================
# A: full accuracy
#    - uses accuracy_A computed over all samples
#    - includes questions regardless of whether answer is None
#
# B: accuracy excluding null answers
#    - uses accuracy_B_exclude_null
#    - considers only questions with valid (non-null) answers
#
# Purpose:
# - to compare how different accuracy definitions
#   affect correlation with consistency
# =========================

import json
from math import sqrt

########################################
# util
########################################

def load_jsonl_to_dict_multi(path, key_field, value_fields):
    """
    jsonl → dict[key] = {field1: v1, field2: v2, ...}
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if key_field in ex:
                key = ex[key_field]
                data[key] = {}
                for vf in value_fields:
                    if vf in ex:
                        data[key][vf] = ex[vf]
    return data


def load_jsonl_to_dict(path, key_field, value_field):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if key_field in ex and value_field in ex:
                data[ex[key_field]] = ex[value_field]
    return data


def pearson_corr(xs, ys):
    n = len(xs)
    if n == 0:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)

    if den_x == 0 or den_y == 0:
        return None

    return num / sqrt(den_x * den_y)

########################################
# main
########################################

def run_corr(acc_jsonl, cons_jsonl, output_jsonl):
    acc_dict = load_jsonl_to_dict_multi(
        acc_jsonl,
        key_field="question_id",
        value_fields=["accuracy_A", "accuracy_B_exclude_null"],
    )

    cons_dict = load_jsonl_to_dict(
        cons_jsonl,
        key_field="question_id",
        value_field="consistency",
    )

    # Matching based on question_id
    common_qids = sorted(set(acc_dict) & set(cons_dict))

    acc_A = []
    acc_B = []
    consistencies = []

    for qid in common_qids:
        if (
            "accuracy_A" in acc_dict[qid]
            and "accuracy_B_exclude_null" in acc_dict[qid]
        ):
            acc_A.append(acc_dict[qid]["accuracy_A"])
            acc_B.append(acc_dict[qid]["accuracy_B_exclude_null"])
            consistencies.append(cons_dict[qid])

    r_A = pearson_corr(acc_A, consistencies)
    r_B = pearson_corr(acc_B, consistencies)

    out = {
        "metric": "pearson_correlation",
        "y": "consistency",
        "num_questions": len(acc_A),
        "accuracy_A_vs_consistency": r_A,
        "accuracy_B_exclude_null_vs_consistency": r_B,
    }

    with open(output_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("✅ Done")
    print(f"Matched questions: {len(acc_A)}")
    print(f"Pearson r (accuracy_A vs consistency): {r_A}")
    print(f"Pearson r (accuracy_B_exclude_null vs consistency): {r_B}")
    print(f"Saved to: {output_jsonl}")
