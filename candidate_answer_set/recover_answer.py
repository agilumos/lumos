'''
python recover_answer.py \
  --input_jsonl /mnt/PK/Inside-out/reasoning/32B-SFT/metamath/data/math/metamath-owmfilter/metamath.jsonl \
  --output_jsonl /mnt/PK/Inside-out/reasoning/32B-SFT/metamath/data/math/metamath-owmfilter/metamath_recovered.jsonl

'''

import json
import re
import argparse
from typing import Dict


def normalize_answer(ans: str) -> str:
    return re.sub(r"\s+", " ", ans.strip())


def recover_answer(answer: str, reasoning: str) -> str:
    """
    1) If answer is not empty, use it as is
    2) If empty, extract the last number from reasoning
    """
    if answer and answer.strip() != "":
        return normalize_answer(answer)

    nums = re.findall(r"-?\d+", reasoning)
    if nums:
        return nums[-1]

    return ""


def process_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            ex = json.loads(line)

            for sample in ex["samples"]:
                original_answer = sample.get("answer", "")
                reasoning = sample.get("reasoning", "")

                recovered = recover_answer(original_answer, reasoning)

                sample["answer"] = recovered
                sample["answer_source"] = (
                    "explicit" if original_answer.strip() != "" else "reasoning_last_number"
                )

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    process_jsonl(args.input_jsonl, args.output_jsonl)

    print(f"✅ Finished processing")
    print(f"📥 Input : {args.input_jsonl}")
    print(f"📤 Output: {args.output_jsonl}")
