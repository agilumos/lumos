"""
Portions of this implementation are adapted from TruthTorchLM / MARS.

TruthTorchLM is licensed under the MIT License.
Copyright (c) 2024 Yavuz Faruk Bakman

See THIRD_PARTY_LICENSES/TruthTorchLM_LICENSE.txt for the full license text.
"""

import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import TruthTorchLM as ttlm


# ==========================================
# 1. Configuration and paths
# ==========================================
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "your-model-name-or-path"
)

INPUT_FILE = os.environ.get(
    "INPUT_FILE",
    "data/example.jsonl"
)

OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE",
    "outputs/mars_scores.jsonl"
)

DEVICE = os.environ.get(
    "DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Load the main LLM
# ==========================================
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 3. Initialize TruthTorchLM MARS
# ==========================================
print("Initializing TruthTorchLM MARS...")
mars_method = ttlm.truth_methods.MARS()

if not hasattr(mars_method.mars_tokenizer, "encode_plus"):
    mars_method.mars_tokenizer.encode_plus = mars_method.mars_tokenizer.__call__


# ==========================================
# 4. Utilities
# ==========================================
CHOICE_PREFIX_RE = re.compile(r"^[A-D]\.\s*")


def strip_choice_prefix(text: str) -> str:
    text = (text or "").strip()
    return CHOICE_PREFIX_RE.sub("", text).strip()


# ==========================================
# 5. Build input strings
# ==========================================
def build_question_context(question: str, choices: list, reasoning: str) -> str:
    """
    Context string passed to the importance model
    (get_importance_vector_MARS).

    - Does not include 'Answer:'
    - Keeps the original MCQ choice format
    """
    choice_block = "\n".join(choices)
    reasoning = (reasoning or "").strip()

    text = (
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_block}\n"
    )

    if reasoning:
        text += f"\nReasoning:\n{reasoning}\n"

    return text.strip()


def build_input_text(question: str, choices: list, reasoning: str) -> str:
    """
    Prefix string used for next-token scoring in the LLM.

    - Includes 'Answer:' at the end
    """
    question_context = build_question_context(question, choices, reasoning)
    return question_context + "\n\nAnswer:"


# ==========================================
# 6. Compute MARS for a single example
# ==========================================
def calculate_single_mars(data_row: dict) -> float | None:
    question = (data_row.get("question") or "").strip()
    choices = data_row.get("choices")
    answer_idx = data_row.get("answer")
    reasoning = data_row.get("raw_samples") or ""

    if not question or choices is None:
        return 0.0

    if answer_idx is None:
        return None

    if not isinstance(answer_idx, int) or not (0 <= answer_idx < len(choices)):
        return 0.0

    raw_answer_text = choices[answer_idx]
    answer_text = strip_choice_prefix(raw_answer_text)
    question_context = build_question_context(question, choices, reasoning)
    input_text = build_input_text(question, choices, reasoning)

    try:
        input_ids_only = tokenizer.encode(input_text, add_special_tokens=True)
        answer_ids_only = tokenizer.encode(answer_text, add_special_tokens=False)

        all_ids_list = input_ids_only + answer_ids_only
        if tokenizer.eos_token_id is not None:
            all_ids_list = all_ids_list + [tokenizer.eos_token_id]

        all_ids = torch.tensor([all_ids_list], dtype=torch.long).to(model.device)

        result = mars_method.forward_hf_local(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            generated_text=answer_text,
            question=question_context,
            all_ids=all_ids,
        )

        return float(result["truth_value"])

    except Exception as e:
        print(f"[ERROR] id={data_row.get('id')}: {e}")
        return 0.0


# ==========================================
# 7. Process the JSONL file
# ==========================================
print(f"Processing: {INPUT_FILE} -> {OUTPUT_FILE}")

output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

num_total = 0
num_written = 0
num_skipped = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in tqdm(fin):
        if not line.strip():
            continue

        num_total += 1
        data = json.loads(line)

        if data.get("source") == "gold":
            num_skipped += 1
            continue

        score = calculate_single_mars(data)
        if score is None:
            num_skipped += 1
            continue

        data["mars_score"] = score
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        num_written += 1

print("\nDone.")
print(f"Total read   : {num_total}")
print(f"Total written: {num_written}")
print(f"Total skipped: {num_skipped}")
print(f"Saved to     : {OUTPUT_FILE}")