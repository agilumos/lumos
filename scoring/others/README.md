# Calculate General Metrics

This repository implements a pipeline for computing general evaluation metrics (including normalized probability-based metrics) and analyzing model behavior under different prompting setups.



## 📌 Overview

The pipeline consists of two main steps:

### Step 1: Normalized Probability Computation

Run:

```bash
step1_normP.py
```

### Step 2: Metric Computation

Run:

```bash
step2_metrics_runner.py
```


## ⚙️ Installation 

```bash
conda env create -f environment.yml
conda activate pkex
```


## 🚀 Pipeline Workflow

```text
Input dataset
   ↓
Step 1: compute normalized probabilities (normP)
   ↓
Step 2: compute evaluation metrics (accuracy / consistency / correlation / normP )
   ↓
Final analysis outputs
```


## 🧩 Step 1: `step1_normP.py`

### Description

This script computes normalized probabilities (normP) and prepares intermediate outputs for downstream metric computation.


### Usage

```bash
CUDA_VISIBLE_DEVICES=0 python external/normP.py \
  --model_size 7B \
  --prompt_id 1 \
  --input_jsonl /mnt/PK/Inside-out/7B/internal_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl
```


### Output

* Saved under:

```
/external_prompt{prompt_id}/...
```


## 🧩 Step 2: `step2_metrics_runner.py`

### Description

This script computes evaluation metrics based on outputs from Step 1.


### Usage

```bash
python metrics_runner.py \
  --input_jsonl /mnt/PK/Inside-out/7B/external_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl
```


### Input

* Output JSONL from Step 1 (`normP-computed dataset`)


### Outputs

The script generates multiple derived files:

#### 1. Accuracy (`_acc`)

* Measures correctness of predictions
* Defined in Appendix A.1


#### 2. Consistency (`_consistency`)

* Measures answer stability across samples
* Defined in Appendix A.2


#### 3. Correlation (`_corrl`)

* Correlation between accuracy and consistency


#### 4. Normalized Probability (`_normp`)

* Aggregated/Averaged normalized probability values
* Defined in Appendix A.3


## 📎 References

* Accuracy: Appendix A.1
* Consistency: Appendix A.2
* Normalized Probability: Appendix A.3

