'''
# Overall metrics
python metrics_runner.py \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl

# Specific metrics only
python metrics_runner.py \
  --metric acc con \
  --input_jsonl data.jsonl \

# Correlation only
python metrics_runner.py \
  --metric corr \
  --acc_jsonl result_acc.jsonl \
  --cons_jsonl result_consistency.jsonl \

'''

import argparse
import os

from acc import run_acc
from avg_normP import run_normP
from consistency import run_cons
from correlation import run_corr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics (accuracy, normP, consistency, correlation)"
    )

    parser.add_argument(
        "--metric",
        nargs="*",
        choices=["acc", "normP", "con", "corr"],
        help="Metrics to run. If omitted, run all."
    )

    parser.add_argument("--input_jsonl", type=str)

    # only for correlation (optional override)
    parser.add_argument("--acc_jsonl", type=str)
    parser.add_argument("--cons_jsonl", type=str)

    return parser.parse_args()


def build_metric_output_path(input_jsonl: str, suffix: str) -> str:
    """
    /path/to/foo.jsonl -> /path/to/foo_{suffix}.jsonl
    """
    dir_path = os.path.dirname(input_jsonl)
    base = os.path.basename(input_jsonl)

    name, ext = os.path.splitext(base)
    if ext != ".jsonl":
        raise ValueError("input_jsonl must be a .jsonl file")

    return os.path.join(dir_path, f"{name}_{suffix}{ext}")

def main():
    args = parse_args()

    metrics = set(args.metric) if args.metric else {"acc", "normP", "con", "corr"}

    # ---------- ACC ----------
    if "acc" in metrics:
        out_path = build_metric_output_path(args.input_jsonl, "acc")
        print(f"▶ Running accuracy → {out_path}")
        run_acc(
            input_jsonl=args.input_jsonl,
            output_jsonl=out_path
        )

    # ---------- normP ----------
    if "normP" in metrics:
        out_path = build_metric_output_path(args.input_jsonl, "normP")
        print("==================================================")
        print(f"▶ Running avg normP → {out_path}")
        run_normP(
            input_path=args.input_jsonl,
            output_path=out_path
        )

    # ---------- consistency ----------
    if "con" in metrics:
        out_path = build_metric_output_path(args.input_jsonl, "consistency")
        print("==================================================")
        print(f"▶ Running consistency → {out_path}")
        run_cons(
            input_path=args.input_jsonl,
            output_path=out_path
        )

    # ---------- correlation ----------
    if "corr" in metrics:
        acc_path = args.acc_jsonl or build_metric_output_path(args.input_jsonl, "acc")
        cons_path = args.cons_jsonl or build_metric_output_path(args.input_jsonl, "consistency")
        out_path = build_metric_output_path(args.input_jsonl, "correl")

        print("==================================================")
        print(f"▶ Running correlation → {out_path}")
        run_corr(
            acc_jsonl=acc_path,
            cons_jsonl=cons_path,
            output_jsonl=out_path
        )

if __name__ == "__main__":
    main()
