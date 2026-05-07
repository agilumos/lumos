"""
CUDA_VISIBLE_DEVICES=0 python cA_for_SV.py \
  --model_size 7B \
  --prompt_id 1 \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl

"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

########################################
# argparse
########################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_size",
    type=str,
    required=True,
    choices=["7B", "13B", "32B"],
)
parser.add_argument(
    "--input_jsonl",
    type=str,
    required=True,
)
parser.add_argument(
    "--prompt_id",
    type=int,
    required=True,
    choices=[1, 2, 3, 4],
)

args = parser.parse_args()

VALID_SOURCES = {"greedy", "sampling"}

########################################
# MODEL CONFIG
########################################

MODEL_CONFIGS = {
    "7B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct",
        "max_new_tokens": 1,
    },
    "13B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct",
        "max_new_tokens": 1,
    },
    "32B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct",
        "max_new_tokens": 1,
    },
}

cfg = MODEL_CONFIGS[args.model_size]

########################################
# output path
########################################

output_jsonl = args.input_jsonl.replace(".jsonl", "_judge.jsonl")

########################################
# Model loading
########################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_path"],
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()

########################################
# Prompt
########################################

def build_true_false_messages_mc_prompt1(question, choices, answer_index, answer_text):
    choice_block = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return [
        {
            "role": "system",
            "content": (
                "Your job is to judge whether a proposed answer "
                "to a multiple-choice question is correct."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Choices:\n{choice_block}\n\n"
                f"Proposed Answer:\n{answer_text}\n\n"
                "Is the proposed answer:\n"
                "A: CORRECT\n"
                "B: INCORRECT\n\n"
                "Just return the letter A or B, with no text around it."
            ),
        },
    ]

def build_true_false_messages_mc_prompt3(question, choices, answer_index, answer_text, reasoning):
    choice_block = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return [
        {
            "role": "system",
            "content": (
                "Your job is to judge whether a proposed answer "
                "to a multiple-choice question is correct."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Choices:\n{choice_block}\n\n"
                f"Reasoning:\n{reasoning}\n\n"
                f"Proposed Answer:\n{answer_text}\n\n"
                "Is the proposed answer:\n"
                "A: CORRECT\n"
                "B: INCORRECT\n\n"
                "Just return the letter A or B, with no text around it."
            ),
        },
    ]
########################################
# Judge generation
########################################

@torch.no_grad()
def generate_judge_answer(messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=False,
    )

    gen_ids = output_ids[0, input_ids.shape[-1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if answer not in {"A", "B"}:
        return None
    return answer

########################################
# Process JSONL
########################################
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def process_jsonl():
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    total_lines = count_lines(args.input_jsonl)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(
            fin,
            total=total_lines,
            desc="🧠 Judging answers",
            dynamic_ncols=True,
        ):
            ex = json.loads(line)

            if ex.get("source") not in VALID_SOURCES:
                continue

            question = ex["question"]
            choices = ex["choices"]

            answer_index = ex.get("answer")
            if answer_index is None:
                answer_text = None
            else :
                answer_text = choices[answer_index]

            reasoning = ex["raw_samples"]

            if args.prompt_id in (1, 2):
                messages = build_true_false_messages_mc_prompt1(
                    question,
                    choices,
                    answer_index,
                    answer_text,
                )

            elif args.prompt_id in (3,4):
                messages = build_true_false_messages_mc_prompt3(
                    question,
                    choices,
                    answer_index,
                    answer_text,
                    reasoning,
                )

            judge_answer = generate_judge_answer(messages)

            out = {
                "id": ex["id"],
                "gold_answer": ex["gold_answer"],
                "answer": ex["answer"],
                "count": ex["count"],
                "source": ex["source"],
                "label": ex["label"],
                "source": ex["source"],  
                "judge_answer": judge_answer,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
########################################
# main
########################################

if __name__ == "__main__":
    process_jsonl()
    print("✅ Judge results jsonl generation completed")