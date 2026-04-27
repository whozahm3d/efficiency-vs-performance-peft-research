# Efficiency vs Performance PEFT Research

## Overview
This repository presents a **systematic benchmarking study** of Parameter-Efficient Fine-Tuning (PEFT) techniques compared to full fine-tuning using DistilBERT.

The project evaluates how different PEFT methods balance:
- model performance
- computational efficiency
- training cost
- stability across runs

The goal is to provide **practical insights for selecting fine-tuning strategies under resource constraints**.

---

## Key Highlights

- Benchmarking of **Full Fine-Tuning vs PEFT methods**
- Evaluation on **multiple NLP datasets**
- **Multi-seed experimentation** for robustness
- **Statistical validation** (Bootstrap CI, McNemar’s Test)
- Detailed **efficiency metrics** (time, memory, parameters)

---

## Methods

### Models
- Full Fine-Tuning (DistilBERT)
- LoRA (Low-Rank Adaptation)
- Adapter Tuning
- Prompt Tuning
- Soft Prompt Tuning

### Baselines
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Zero-shot DistilBERT

---

## Datasets

- SST-2 (Sentiment Classification)
- AG News (Topic Classification)
- Amazon Polarity (Cross-domain Sentiment)

---

## Evaluation

### Performance Metrics
- Accuracy
- Macro F1 Score

### Efficiency Metrics
- Trainable parameters (%)
- Training time
- Inference latency
- GPU memory usage

### Statistical Analysis
- Multi-seed mean and variance
- Bootstrap confidence intervals
- McNemar’s significance test

---

## Experimental Design

- Backbone: DistilBERT-base-uncased
- Fixed seeds: `[42, 0, 123]`
- Stratified splits (train/val/test)
- Consistent preprocessing pipeline

---

## Ablation Studies

- Effect of dataset size (10%, 50%, 100%)
- Impact of LoRA rank
- Adapter dimension sensitivity

---

## Results (Summary)

- LoRA achieves near full fine-tuning performance with significantly fewer parameters  
- Adapter tuning provides strong performance-efficiency balance  
- Prompt tuning underperforms on smaller models  
- Multi-seed results highlight stability differences across methods  

---

## Project Structure
├── notebooks/ # Main research workflow
├── src/ # Modular implementation (models, training, evaluation)
├── results/ # Metrics, logs, visualizations
├── reports/ # Final report / documentation


---

## Installation

```bash
git clone https://github.com/your-username/peft-efficiency-vs-performance.git
cd peft-efficiency-vs-performance
pip install -r requirements.txt

Usage

Run the main experiment pipeline:

jupyter notebook notebooks/research_pipeline.ipynb
Reproducibility
Fixed random seeds across experiments
Controlled experimental setup
Results logged for verification
Author

Ali Ahmad
