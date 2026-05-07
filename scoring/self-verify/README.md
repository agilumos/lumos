# Self-Verification (SV) Evaluation Pipeline

This repository provides a two-step pipeline for computing **Self-Verification (SV)** metrics. The implementation follows the methodology described in **Appendix A.7 of the paper**.


## 📌 Overview

The SV evaluation is performed in two sequential steps:

1. **Step 1**: Generate model judgments (`cA`) for each sample
2. **Step 2**: Compute confusion matrices and evaluation metrics (SV_A, SV_B)


## ⚙️ Installation 

```bash
conda env create -f environment.yml
conda activate pkex
```

## 🧩 Step 1: 'step1_cA_for_SV.py'

### Description

This step runs the model to produce **self-verification judgments** (`judge_answer`) for each sample in the input dataset.

### Usage

```bash
CUDA_VISIBLE_DEVICES=0 python cA_for_SV.py \
  --model_size 7B \
  --prompt_id 1 \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl
```

### Output

* A new JSONL file is created in the same directory as the input
* The filename will have the suffix:

  ```
  *_labeling_judge.jsonl
  ```
* This file includes model-generated **self-verification judgments (`judge_answer`)**


## 🧩 Step 2: 'step2_SV_confusion_matrix.py'

### Description

This step computes confusion matrices and evaluation metrics based on:

* Ground truth labels (`label`)
* Model self-verification outputs (`judge_answer`)

### Usage

```bash
python SV_confusion_matrix.py \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling_judge.jsonl
```

### Outputs

Two result files are generated:

1. **`*_SVexp_A.jsonl`**

   * Includes all samples in evaluation
   * **Null answers (`answer=None`) are included**
   * Reflects **overall performance**, including generation failures

2. **`*_SVexp_B.jsonl`**

   * Excludes samples where `answer=None`
   * Evaluates only valid model outputs
   * Reflects **classification performance given a valid response**


## 📊 Metric Definitions

* **SV_A**
  Accuracy and related metrics computed over **all samples**, including cases where the model failed to produce an answer.

* **SV_B**
  Metrics computed only on **valid responses**, excluding `answer=None` cases.

This distinction allows analysis of:

* End-to-end system reliability (**SV_A**)
* Pure verification capability (**SV_B**)


## 📎 Reference

For full methodological details, refer to **Appendix A.7 of the paper**.
