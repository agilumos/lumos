"""
CUDA_VISIBLE_DEVICES=0 python cA_reasoning.py \
  --model_size 7B \
  --dataset gsm8k-train \
  --input_jsonl /mnt/PK/dataset/gsm8k/main/train/data.jsonl
"""

# dataset : gsm8k-train     gsm8k-test    gsm1k    gsmsym   metamath
# # Be mindful of the NUM_RANDOM_EXAMPLES value!!

import os
import sys
import argparse
import json
import re
import random
import time
from typing import List, Dict
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

########################################
# argparse
########################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_size",
    type=str,
    required=True,
    choices=["7B", "13B", "32B", "7B-SFT", "13B-SFT", "32B-SFT"],
    help="Model size to use"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "gsm8k-train",
        "gsm8k-test",
        "gsm1k",
        "gsmsym",
        "metamath",
    ],
    help="Dataset type"
)
parser.add_argument(
    "--input_jsonl",
    type=str,
    required=True,
    help="Path to input jsonl"
)

args = parser.parse_args()

########################################
# Model-specific settings
########################################
MODEL_CONFIGS = {
    "7B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct",
        "insert_dir": "Inside-out/reasoning/7B",
        "max_new_tokens": 1024,
    },
    "13B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct",
        "insert_dir": "Inside-out/reasoning/13B",
        "max_new_tokens": 1024,
    },
    "32B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct",
        "insert_dir": "Inside-out/reasoning/32B",
        "max_new_tokens": 1024,
    },
    "7B-SFT": {
        "model_path": "/mnt/PK/models/OLMo2-SFT/OLMo-2-1124-7B-SFT",
        "insert_dir": "Inside-out/reasoning/7B-SFT",
        "max_new_tokens": 1024,
    },
    "13B-SFT": {
        "model_path": "/mnt/PK/models/OLMo2-SFT/OLMo-2-1124-13B-SFT",
        "insert_dir": "Inside-out/reasoning/13B-SFT",
        "max_new_tokens": 1024,
    },
    "32B-SFT": {
        "model_path": "/mnt/PK/models/OLMo2-SFT/OLMo-2-0325-32B-SFT",
        "insert_dir": "Inside-out/reasoning/32B-SFT",
        "max_new_tokens": 1024,
    },
}

cfg = MODEL_CONFIGS[args.model_size]

########################################
# Default settings
########################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = cfg["model_path"]
INSERT_DIR = cfg["insert_dir"]
MAX_NEW_TOKENS = cfg["max_new_tokens"]

NUM_SAMPLES = 8 # number of candidate answer (answer 8 times)
TEMPERATURE = 1.0
MAX_TOKENS_ANSWER = 8

NUM_RANDOM_EXAMPLES = 1319
#NUM_RANDOM_EXAMPLES = 1000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_PREFIX = "/mnt/PK"
DATASET_DIR = "dataset"
INPUT_JSONL = args.input_jsonl

########################################
# Output path
########################################

def build_output_path(input_path: str) -> str:
    expected_prefix = os.path.join(BASE_PREFIX, DATASET_DIR)

    if not input_path.startswith(expected_prefix):
        raise ValueError(f"input_jsonl must start with {expected_prefix}")

    relative_path = os.path.relpath(input_path, expected_prefix)
    return os.path.join(BASE_PREFIX, INSERT_DIR, relative_path)


OUTPUT_JSONL = build_output_path(INPUT_JSONL)

########################################
# Model loading
########################################

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

########################################
# Short-answer filtering
########################################

def is_single_answer(answer: str) -> bool:
    answer = answer.strip()

    if len(answer.split()) > MAX_TOKENS_ANSWER:
        return False

    if re.search(r"[.!?]", answer):
        return False

    banned_tokens = [
        " is ", " was ", " are ", " were ",
        " because ", " therefore ", " which ",
        " that ", " who "
    ]
    lowered = f" {answer.lower()} "
    return not any(bt in lowered for bt in banned_tokens)

def normalize_answer(ans: str) -> str:
    return re.sub(r"\s+", " ", ans.strip())

########################################
# Answer generation
########################################

def build_prompt(question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that solves math word problems. "
                "You must reason step by step and produce a correct numerical answer."
            )
        },
        {
            "role": "user",
            "content": (
                "Solve the following math problem.\n\n"
                f"Question: {question}\n\n"
                "Write out your reasoning step by step.\n"
                "Then write the final answer in this format:\n"
                "### FINAL ANSWER:\n"
                "<number>"
            )
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

@torch.no_grad()
def generate_answer(question: str) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

def parse_reasoning_and_answer(text: str):
    reasoning = text
    answer = ""

    if "### FINAL ANSWER:" in text:
        before, after = text.split("### FINAL ANSWER:", 1)
        reasoning = normalize_answer(before)
        answer = normalize_answer(after)

    return reasoning, answer

########################################
# Build candidate answer set
########################################

def build_candidate_answer_set(question: str) -> List[Dict]:
    samples = []

    for sample_id in range(NUM_SAMPLES):
        text = generate_answer(question)
        reasoning, answer = parse_reasoning_and_answer(text)

        samples.append({
            "sample_id": sample_id,
            "answer": answer,
            "reasoning": reasoning,
            "is_single_answer": is_single_answer(answer)
        })

    return samples

########################################
# Process dataset
########################################

def process_dataset(input_path: str, output_path: str, num_samples: int, prefix: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()

    sampled_indices = sorted(random.sample(range(len(all_lines)), num_samples))

    with tqdm(
        total=len(sampled_indices),
        desc="📘 Processing questions",
        file=sys.stderr,
        dynamic_ncols=True
    ) as qbar, open(output_path, "w", encoding="utf-8") as fout:

        for idx in sampled_indices:
            ex = json.loads(all_lines[idx])
            qid = f"{prefix}_{idx}"
            question = ex["question"]
            gold_answer = str(ex["answer"])

            samples = build_candidate_answer_set(question)

            fout.write(json.dumps({
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "samples": samples
            }, ensure_ascii=False) + "\n")

            qbar.update(1)

def process_dataset_gsm1k(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r") as fin:
        lines = list(fin)   

    with tqdm(
        total=len(lines),
        desc="📘 Processing gsm1k",
        file=sys.stderr,
        dynamic_ncols=True
    ) as qbar, open(input_path, "r") as fin, open(output_path, "w") as fout:

        for i, line in enumerate(fin):
            ex = json.loads(line)
            samples = build_candidate_answer_set(ex["question"])

            fout.write(json.dumps({
                "id": f"gsm1k_{i}",
                "question": ex["question"],
                "gold_answer": str(ex["answer"]),
                "samples": samples
            }, ensure_ascii=False) + "\n")

            qbar.update(1)

def process_dataset_gsmsym(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r") as fin:
        lines = list(fin)

    with tqdm(
        total=len(lines),
        desc="📘 Processing gsmsym",
        file=sys.stderr,
        dynamic_ncols=True
    ) as qbar, open(input_path, "r") as fin, open(output_path, "w") as fout:

        for line in fin:
            ex = json.loads(line)
            samples = build_candidate_answer_set(ex["question"])

            fout.write(json.dumps({
                "id": f"{ex['id']}_{ex['instance']}",
                "question": ex["question"],
                "gold_answer": str(ex["answer"]),
                "samples": samples
            }, ensure_ascii=False) + "\n")

            qbar.update(1)

########################################
# Run
########################################

if __name__ == "__main__":
    start = time.time()

    if args.dataset == "gsm8k-train":
        process_dataset(INPUT_JSONL, OUTPUT_JSONL, NUM_RANDOM_EXAMPLES, "main_train")

    elif args.dataset == "gsm8k-test":
        process_dataset(INPUT_JSONL, OUTPUT_JSONL, NUM_RANDOM_EXAMPLES, "main_test")

    elif args.dataset == "gsm1k":
        process_dataset_gsm1k(INPUT_JSONL, OUTPUT_JSONL)

    elif args.dataset == "gsmsym":
        process_dataset_gsmsym(INPUT_JSONL, OUTPUT_JSONL)

    elif args.dataset == "metamath":
        process_dataset(INPUT_JSONL, OUTPUT_JSONL, NUM_RANDOM_EXAMPLES, "metamath")

    else:
        raise ValueError(args.dataset)

    elapsed = time.time() - start
    print(f"\n✅ Model: {args.model_size}")
    print(f"✅ Dataset: {args.dataset}")
    print(f"✅ Output: {OUTPUT_JSONL}")
    print(f"✅ Time: {elapsed/60:.2f} min")
