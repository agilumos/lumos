# Lumos: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs

This repository contains the official codebase for the paper:

**“Lumos: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs”** 


## 📌 Overview

Lumos is a framework for analyzing how parametric knowledge in large language models (LLMs) is reflected in their behavioral outputs. The pipeline constructs factual datasets from entity-level information and evaluates model behavior using multiple metrics.


## 🚀 Pipeline Workflow

The full experimental pipeline consists of three stages:

1. MCQ Construction (entity extraction + MCQ generation)
2. Candidate Answer Set Construction
3. Scoring & Evaluation

Each stage is implemented as an independent module with its own README for detailed configuration and usage.


## 📁 Repository Structure

- mcq/ : Entity extraction, chunk retrieval, and MCQ generation (see mcq/README.md for details)
- candidate_answer_set/ : Generation of candidate answers
- scoring/ : Evaluation scripts (Accuracy, NormP, MARS, etc.)  


## 🔄 End-to-End Workflow

### 1. MCQ Construction
Extract entities from raw corpora, retrieve context chunks, and generate multiple-choice questions.
This stage covers the full pipeline from entity extraction to MCQ generation, using [CRAFT](https://github.com/ziegler-ingo/CRAFT) internally (clone required).

### 2. Candidate Answer Generation
Generate a set of candidate answers for each question using LLM-based sampling or decoding strategies.

### 3. Scoring & Evaluation
Compute evaluation metrics over model outputs, including:
- Accuracy, Decisional consistency
- Normalized probability
- MARS
- SAPLMA
- Linear probing
- Self-verify


## Usage

Each stage can be executed independently or as part of the full pipeline:

MCQ → Candidate Answer Set → Scoring


## 📎 Notes

- All experiments in the paper are reproducible using this codebase.
- For questions or issues, please contact the authors via the email listed in the paper.
