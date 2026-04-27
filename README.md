# Efficiency vs Performance PEFT Research

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This repository presents a **research-driven empirical analysis** of **Parameter-Efficient Fine-Tuning (PEFT)** methods compared to full fine-tuning using **DistilBERT**.

The project benchmarks how different fine-tuning strategies trade off:

- 📊 Model performance  
- ⚡ Computational efficiency  
- 💾 Resource utilization  
- 🔁 Stability across multiple runs  

The goal is to provide **practical insights for deploying NLP models under resource constraints**.

---

## 🎯 Objectives

- Compare **Full Fine-Tuning vs PEFT methods**
- Analyze **efficiency vs accuracy trade-offs**
- Evaluate **model stability using multi-seed experiments**
- Perform **statistical validation of results**
- Study performance across **multiple datasets**

---

## 🧠 Methods

### 🔹 Neural Models
- Full Fine-Tuning (DistilBERT)

### 🔹 PEFT Techniques
- LoRA (Low-Rank Adaptation)
- Adapter Tuning
- Prompt Tuning
- Soft Prompt Tuning

### 🔹 Classical Baselines
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Zero-shot DistilBERT

---

## 📊 Datasets

| Dataset | Task | Classes |
|--------|------|--------|
| SST-2 | Sentiment Classification | 2 |
| AG News | Topic Classification | 4 |
| Amazon Polarity | Sentiment (Cross-domain) | 2 |

---

## ⚙️ Experimental Setup

- Backbone Model: `distilbert-base-uncased`
- Sequence Length: 128
- Seeds: `[42, 0, 123]`
- Data Split: Train / Validation / Test
- Consistent preprocessing across all models

---

## 📈 Evaluation

### Performance Metrics
- Accuracy  
- Macro F1 Score  

### Efficiency Metrics
- Trainable Parameters (%)  
- Training Time  
- Inference Latency  
- GPU Memory Usage  

### Statistical Analysis
- Multi-seed mean ± standard deviation  
- Bootstrap Confidence Intervals (n=1000)  
- McNemar’s Significance Test  

---

## 🧪 Ablation Studies

- Dataset size: 10%, 50%, 100%  
- LoRA rank variation  
- Adapter dimension tuning  

---

## 📊 Key Findings

- LoRA achieves near full fine-tuning performance with significantly fewer parameters  
- Adapter tuning provides strong efficiency-performance balance  
- Prompt tuning underperforms for smaller models like DistilBERT  
- Multi-seed evaluation reveals stability differences across methods  

---

## 📁 Project Structure
├── notebooks/ # Main experiment pipeline
├── src/ # Modular code (models, training, evaluation)
├── results/ # Metrics, plots, logs
├── reports/ # Final research report
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/peft-efficiency-vs-performance.git
cd peft-efficiency-vs-performance
pip install -r requirements.txt

Run the main experiment pipeline:

jupyter notebook notebooks/EFFICENCY_VS_PERFORMANCE_PEFT_Research_PHASE_COMPLETE.ipynb


🔬 Reproducibility
Fixed random seeds across all experiments
Controlled and consistent preprocessing
Logged results for verification


🧾 Requirements

See requirements.txt

Tested on:

Python 3.10+
🚀 Future Work
Extend to larger transformer models (e.g., BERT, RoBERTa)
Explore additional PEFT techniques
Hyperparameter optimization
Deployment benchmarking
👤 Author

Ali Ahmad
