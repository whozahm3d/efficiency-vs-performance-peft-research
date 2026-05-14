# Efficiency vs. Performance in Parameter-Efficient Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-0.10%2B-5C16C5)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Objectives](#2-research-objectives)
3. [Methodology](#3-methodology)
4. [Experimental Setup](#4-experimental-setup)
5. [Results Summary](#5-results-summary)
6. [Ablation Studies](#6-ablation-studies)
7. [Statistical Validation](#7-statistical-validation)
8. [Repository Structure](#8-repository-structure)
9. [Installation & Reproduction](#9-installation--reproduction)
10. [Model Card](#10-model-card)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Author](#12-author)

---

## 1. Project Overview

This repository presents a **systematic empirical study** of Parameter-Efficient Fine-Tuning (PEFT) techniques benchmarked against full fine-tuning using **DistilBERT** as the backbone model. The study rigorously evaluates how competing fine-tuning strategies navigate the fundamental trade-off between:

- **Predictive performance** — accuracy and macro F1 across text classification tasks
- **Computational efficiency** — trainable parameter count, training time, and GPU memory footprint
- **Inference latency** — per-sample prediction speed
- **Result stability** — multi-seed variance and bootstrap confidence intervals

The research is motivated by the growing need to deploy NLP models under real-world resource constraints, where fine-tuning large language models at full scale is often impractical.

The findings are documented in a full [IEEE-format research paper](research_paper/PEFT_Efficiency_vs_Performance_IEEE_Paper.pdf) and a detailed [experimental report](reports/research_report.pdf).

---

## 2. Research Objectives

| # | Objective |
|---|-----------|
| O1 | Compare Full Fine-Tuning against four PEFT strategies on standardized benchmarks |
| O2 | Quantify the efficiency–accuracy trade-off across parameter budgets |
| O3 | Evaluate stability via multi-seed experimentation and statistical testing |
| O4 | Assess prompt-based tuning viability for smaller transformer models |
| O5 | Conduct ablation analysis on dataset size and LoRA rank sensitivity |

---

## 3. Methodology

### 3.1 Fine-Tuning Strategies

**Full Fine-Tuning (Baseline)**
- All 66.9M parameters of `distilbert-base-uncased` updated during training
- Serves as the performance upper bound

**PEFT Methods**

| Method | Description | Trainable Params |
|--------|-------------|-----------------|
| LoRA (r=4) | Low-Rank Adaptation injected into attention layers | ~666K (0.99%) |
| LoRA (r=8) | LoRA with rank-8 decomposition matrices | ~740K (1.10%) |
| LoRA (r=16) | LoRA with rank-16 decomposition matrices | ~887K (1.33%) |
| LoRA (r=32) | LoRA with rank-32 decomposition matrices | ~1.18M (1.77%) |
| Adapter Tuning | Bottleneck adapter layers inserted between transformer blocks | ~1.21M (1.81%) |
| Prompt Tuning | Learnable soft prompt tokens prepended to input | ~607K (0.91%) |
| Soft Prompt | Variant of prompt tuning with continuous embeddings | ~607K (0.91%) |

**Classical ML Baselines**

| Method | Description |
|--------|-------------|
| TF-IDF + Logistic Regression | Sparse bag-of-words representation with linear classifier |
| TF-IDF + Linear SVM | Sparse representation with support vector classifier |
| Zero-Shot DistilBERT | Inference with no task-specific training |

### 3.2 Datasets

| Dataset | Task | Classes | Train Size | Evaluation Size |
|---------|------|---------|-----------|----------------|
| SST-2 | Sentiment Classification | 2 | 6,999 | Held-out test set |
| AG News | Topic Classification | 4 | 6,999 | Held-out test set |
| Amazon Polarity | Cross-domain Sentiment | 2 | 6,999 | Held-out test set |

> All datasets are capped at 10K samples per split. Consistent preprocessing and tokenization are applied across all methods.

---

## 4. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Backbone Model | `distilbert-base-uncased` |
| Max Sequence Length | 128 tokens |
| Random Seeds | `[42, 0, 123]` |
| LoRA Config | `r ∈ {4, 8, 16, 32}`, `alpha=16`, `dropout=0.1` |
| Data Split | Train / Validation / Test |
| Evaluation | Multi-seed mean ± std, 95% bootstrap CI |
| Hardware | GPU (CUDA), monitored via `psutil` |

---

## 5. Results Summary

### 5.1 Performance vs. Efficiency

The table below reports single-run results for all methods across all three datasets. Full fine-tuning (Full FT) serves as the performance ceiling.

| Method | Type | SST-2 F1 | AG News F1 | Amazon F1 | Trainable Params | Param % | Train Time (s) | Infer. (ms) | GPU (MB) |
|--------|------|----------|-----------|----------|-----------------|---------|---------------|------------|---------|
| Full FT | Baseline | 0.8978 | 0.9191 | 0.9090 | 66.96M | 100.0% | 68.8 | 0.126 | 2,225 |
| LoRA r=4 | PEFT | 0.8380 | 0.8717 | 0.8616 | 666K | 0.99% | 54.9 | 0.172 | 1,207 |
| LoRA r=8 | PEFT | 0.8342 | 0.8712 | 0.8586 | 740K | 1.10% | 56.0 | 0.214 | 1,197 |
| LoRA r=16 | PEFT | 0.8336 | 0.8712 | 0.8596 | 887K | 1.33% | 56.8 | 0.234 | 1,198 |
| LoRA r=32 | PEFT | 0.8348 | 0.8702 | 0.8611 | 1.18M | 1.77% | 57.1 | 0.218 | 1,205 |
| Adapter | PEFT | 0.8516 | 0.8648 | 0.8626 | 1.21M | 1.81% | 58.2 | 0.162 | 1,283 |
| Prompt Tuning | PEFT | 0.4596 | 0.8701 | 0.7332 | 607K | 0.91% | 51.9 | 0.119 | 1,136 |
| Soft Prompt | PEFT | 0.4947 | 0.8711 | 0.7571 | 607K | 0.91% | 52.1 | 0.164 | 1,138 |
| TF-IDF + LR | Classical | 0.7874 | 0.8948 | 0.8434 | — | — | 0.3 | 0.0002 | — |
| TF-IDF + SVM | Classical | 0.8112 | 0.8978 | 0.8448 | — | — | 0.1 | 0.0001 | — |
| Zero-Shot | Baseline | 0.6465 | 0.3468 | 0.6108 | 0 | 0% | 0.0 | 12.2 | — |

### 5.2 Key Findings

**LoRA is the most efficient high-performing PEFT method.** LoRA r=4 achieves 0.838 F1 on SST-2 using only 0.99% of full fine-tuning parameters, with training time reduced by ~20% and GPU memory usage cut by approximately 46% (from 2,225 MB to 1,207 MB).

**Adapter Tuning offers a strong accuracy–efficiency balance.** Adapters outperform all LoRA variants on SST-2 (0.8516 F1) and Amazon Polarity (0.8626 F1), at the cost of a slightly larger parameter footprint (~1.81%).

**Prompt Tuning and Soft Prompt fail on sentiment tasks.** Both methods collapse on SST-2 (F1 of 0.46 and 0.49 respectively), demonstrating that prompt-based tuning is unreliable for smaller backbone models like DistilBERT in binary sentiment classification.

**LoRA rank has diminishing returns.** Performance plateaus beyond r=4 on all datasets. Increasing from r=4 to r=32 adds ~78% more parameters with negligible F1 gain (~0.001 on most tasks).

**Classical ML baselines remain competitive.** TF-IDF + SVM achieves 0.8978 F1 on AG News — matching Full FT — while requiring no GPU and training in milliseconds. This underscores their continued relevance for resource-constrained deployments.

---

## 6. Ablation Studies

### 6.1 Dataset Size Sensitivity (LoRA r=8)

| Dataset | 10% Data F1 | 50% Data F1 | 100% Data F1 | 50% Retention Rate |
|---------|------------|------------|-------------|-------------------|
| SST-2 | 0.358 | 0.557 | 0.834 | 66.8% |
| AG News | 0.592 | 0.852 | 0.871 | 97.8% |
| Amazon Polarity | 0.482 | 0.819 | 0.859 | 95.4% |

> SST-2 shows the highest sensitivity to data scarcity — performance degrades sharply at 10% and 50% training size relative to other datasets.

### 6.2 LoRA Rank Sensitivity (SST-2)

| Rank | F1 | Trainable Params | F1 per Million Params |
|------|----|-----------------|----------------------|
| r=4 | 0.838 | 666K | 1.259 |
| r=8 | 0.834 | 740K | 1.128 |
| r=16 | 0.834 | 887K | 0.940 |
| r=32 | 0.835 | 1.18M | 0.706 |

> Parameter efficiency (F1 per million trainable parameters) decreases monotonically with rank, confirming that r=4 provides the best efficiency-per-parameter ratio.

---

## 7. Statistical Validation

Multi-seed experiments (seeds: 42, 0, 123) with 95% bootstrap confidence intervals (n=1,000 resamples) and McNemar's test for pairwise significance.

| Method | Dataset | Mean F1 | Std F1 | 95% CI |
|--------|---------|---------|--------|--------|
| Full FT | SST-2 | 0.8902 | ±0.0019 | [0.8881, 0.8917] |
| Full FT | AG News | 0.9175 | ±0.0028 | [0.9143, 0.9196] |
| Full FT | Amazon | 0.9077 | ±0.0006 | [0.9070, 0.9080] |
| LoRA r=8 | SST-2 | 0.8308 | ±0.0028 | [0.8276, 0.8326] |
| LoRA r=8 | AG News | 0.8641 | ±0.0050 | [0.8592, 0.8692] |
| LoRA r=8 | Amazon | 0.8596 | ±0.0023 | [0.8576, 0.8621] |
| Adapter | SST-2 | 0.8503 | ±0.0023 | [0.8486, 0.8529] |
| Adapter | AG News | 0.8371 | ±0.0313 | [0.8056, 0.8682] |
| Adapter | Amazon | 0.8669 | ±0.0023 | [0.8656, 0.8696] |
| Prompt Tuning | SST-2 | 0.4489 | ±0.0284 | [0.4230, 0.4792] |
| Prompt Tuning | AG News | 0.8531 | ±0.0190 | [0.8323, 0.8696] |
| Prompt Tuning | Amazon | 0.7567 | ±0.0074 | [0.7482, 0.7616] |

> **Stability note:** Adapter Tuning on AG News shows higher variance (std=0.031) compared to all LoRA variants, indicating sensitivity to initialization. Prompt Tuning on SST-2 has the widest variance, reflecting instability on binary sentiment tasks.

### Hypothesis Testing Results

| Hypothesis | Description | Outcome |
|------------|-------------|---------|
| H1 | LoRA achieves ≤3pp F1 gap vs. Full FT with <10% params on ≥2 datasets | ❌ Not supported (gap: 4.8–6.4pp) |
| H2 | Adapter outperforms Prompt Tuning on all three datasets | ❌ Partially supported (2/3 datasets) |
| H3 | LoRA r=8 at 50% data achieves ≥90% of full-data F1 on SST-2 | ❌ Not supported (66.8% retention) |

---

## 8. Repository Structure

```
efficiency-vs-performance-peft-research/
│
├── notebooks/
│   ├── EFFICENCY_VS_PERFORMANCE_PEFT_Research_PHASE_COMPLETE.ipynb   # Full experiment pipeline
│   └── PEFT_Research_PHASE_COMPLETE.ipynb                            # Core PEFT training pipeline
│
├── research_paper/
│   ├── PEFT_Efficiency_vs_Performance_IEEE_Paper.pdf                 # IEEE-format publication
│   └── PEFT_Efficiency_vs_Performance_IEEE_Paper.tex                 # LaTeX source
│
├── reports/
│   └── research_report.pdf                                           # Detailed experimental report
│
├── results/
│   ├── model_card.json                                               # Model metadata and limitations
│   └── outputs/
│       ├── master_results_table.csv                                  # Consolidated performance metrics
│       ├── efficiency_metrics.csv                                    # GPU/time/parameter efficiency data
│       ├── statistical_summary.csv                                   # Multi-seed stats and CIs
│       ├── ablation_results.csv                                      # Dataset size & rank ablations
│       ├── hypothesis_results.csv                                    # Formal hypothesis test outcomes
│       ├── evaluation_master.csv                                     # Full evaluation log
│       ├── experiment_log.csv                                        # Run-level experiment tracking
│       ├── bias_fairness.csv                                         # Bias and fairness analysis
│       ├── mcnemar_full_matrix.csv                                   # Pairwise McNemar test matrix
│       └── [dataset splits: sst2, ag_news, amazon — train/val/test]
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. Installation & Reproduction

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended; CPU execution supported but slow)
- 4–8 GB GPU VRAM (Full FT requires ~2.2 GB; PEFT methods ~1.1–1.3 GB)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/efficiency-vs-performance-peft-research.git
cd efficiency-vs-performance-peft-research

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Experiments

**Full experiment pipeline (all phases):**
```bash
jupyter notebook notebooks/EFFICENCY_VS_PERFORMANCE_PEFT_Research_PHASE_COMPLETE.ipynb
```

**Core PEFT training only:**
```bash
jupyter notebook notebooks/PEFT_Research_PHASE_COMPLETE.ipynb
```

### Reproducibility

All experiments use fixed random seeds (`42`, `0`, `123`) set across Python, NumPy, and PyTorch. Preprocessing, tokenization, and data splits are deterministic and logged. All output CSVs in `results/outputs/` are included for direct verification without re-running experiments.

---

## 10. Model Card

> Best-performing PEFT model: **LoRA r=8 on DistilBERT**

| Field | Detail |
|-------|--------|
| Model Name | LoRA-DistilBERT-PEFT-Study |
| Version | 1.0 (research prototype) |
| Base Model | `distilbert-base-uncased` (Sanh et al., 2019) |
| PEFT Config | LoRA, r=8, alpha=16, dropout=0.1 |
| Trainable Parameters | 739,586 (1.09% of full fine-tuning) |
| Training Datasets | SST-2, AG News, Amazon Polarity |
| Best SST-2 F1 | 0.8342 |
| Intended Use | Text classification research and benchmarking |
| Out-of-Scope | Medical, legal, or safety-critical decisions; production deployment without safety evaluation; non-English text |
| Known Biases | Underperforms on negation-heavy text; may degrade on domain-shifted inputs |
| Carbon Footprint | Significantly reduced vs. full fine-tuning due to fewer active parameters |

---

## 11. Limitations & Future Work

### Current Limitations

- Backbone is limited to `distilbert-base-uncased`; findings may not generalize to larger models
- Evaluation is English-only and capped at 10K samples per dataset
- Prompt and Soft Prompt tuning results are sensitive to initialization and may require tuning
- Model card advises against deployment in safety-critical applications without further evaluation

### Future Work

- Extend experiments to larger backbones: BERT-base, RoBERTa-base, and GPT-2
- Explore additional PEFT methods: QLoRA (quantized LoRA), IA³, Prefix Tuning
- Systematic hyperparameter search for prompt length and adapter bottleneck dimension
- Multilingual evaluation using mBERT or XLM-R
- Deployment benchmarking: ONNX export, TorchScript, quantization post-training

---

## 12. Author

**Ali Ahmad**

For questions regarding the research, methodology, or reproduction of results, please open a GitHub Issue in this repository.

---

*This project is released under the [MIT License](LICENSE).*
