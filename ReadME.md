---
title: "Sokratik API"
emoji: "🏛️"
colorFrom: "gray"
colorTo: "blue"
sdk: "docker"
app_file: "dockerfile"
pinned: false
---

# 🏛️ SOKRATIK — The Stoic LLM

> _"Man is not worried by real problems so much as by his imagined anxieties about real problems."_  
> — Epictetus

**SOKRATIK** is an end-to-end **MLOps-driven Large Language Model project** that aims to bring Stoic philosophy to life through an AI that speaks with the calm wisdom of **Marcus Aurelius**, **Epictetus**, and **Seneca**.

The project fine-tunes **Llama 3 8B** on the _complete works of Stoic philosophers_ using **QLoRA** and builds a **production-ready, versioned MLOps pipeline** for reproducible, deployable model training and tracking.

---

## 📊 Project Overview

| Aspect                | Details                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Model**             | `meta-llama/Llama-3.1-8B-Instruct`                                                                         |
| **Training Strategy** | QLoRA (4-bit quantization) on a Colab T4 GPU                                                               |
| **Core Libraries**    | `transformers`, `peft`, `trl (SFTTrainer)`, `bitsandbytes`                                                 |
| **MLOps Stack**       | DVC for data versioning, MLflow for experiment tracking                                                    |
| **Storage**           | Google Drive (as DVC remote)                                                                               |
| **Compute**           | Google Colab                                                                                               |
| **Goal**              | Create a fine-tuned LLM that embodies Stoic philosophy, fully managed through reproducible MLOps pipelines |

---

## 🧱 Project Structure

```
SOKRATIK/
│
├── .dvc/                     # DVC configuration files
├── .venv/                    # Virtual environment (git-ignored)
│
├── data/
│   ├── raw/                  # Original, untouched Stoic texts
│   ├── processed/            # Cleaned and unified text corpus
│   └── final/                # ML-ready datasets (JSONL format)
│
├── scripts/                  # Core automation and logic
│   ├── 01_baseline_check.py      # Evaluate base Llama 3 performance
│   ├── 02_cleaning_text.py       # Clean raw text and remove artifacts
│   ├── 03_Formatting_Data.py     # Convert corpus to train.jsonl
│   └── 04_finetune.py            # Fine-tuning with QLoRA + MLflow logging
│
├── requirements.txt           # Project dependencies
├── .gitignore                 # Ignore unnecessary files
└── README.md                  # You are here 🏛️
```

---

## 🚀 Quickstart Guide

### 1️⃣ Setup Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SOKRATIK.git
cd SOKRATIK

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # (Windows: .\.venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Pull Data via DVC

This project uses **DVC (Data Version Control)** to manage large files such as cleaned corpora and training datasets.

```bash
# Authenticate DVC (opens a browser for Google Drive)
dvc pull
```

Once complete, your `train.jsonl` and `stoic_corpus.txt` will be automatically placed in the right folders.

---

### 3️⃣ Run Fine-Tuning

```bash
# Log in to Hugging Face
huggingface-cli login

# Run the fine-tuning script
python scripts/04_finetune.py
```

Monitor all experiment metrics (loss curves, learning rate, etc.) directly from **MLflow**.

---

## 📅 Project Milestones

### ✅ Phase 1: Data Foundation (Month 1)

- **Repo Setup:** Git + virtual environment initialized.
- **Baseline Established:** `01_baseline_check.py` logs the "pre-trained" behavior of Llama 3.
- **Data Sourcing:** Downloaded _Meditations_, _Discourses_, and _Letters from a Stoic_.
- **Cleaning Pipeline:** Built `02_cleaning_text.py` to unify and clean all texts → `stoic_corpus.txt`.
- **Data Formatting:** Created `03_Formatting_Data.py` → converts corpus into `train.jsonl`.
- **Data Versioning:** Integrated **DVC** with Google Drive for data reproducibility.
- **Security:** Secured GCloud credentials in `.dvc/config.local` (git-ignored).

---

### 🚧 Phase 2: ML Core (Month 2)

- **Training Strategy:** Implemented **QLoRA** to fit an 8B model on a single T4 GPU.
- **Fine-Tuning Pipeline:** Created `04_finetune.py` to handle:
  - 4-bit model quantization (`BitsAndBytesConfig`)
  - LoRA adapter configuration (`LoraConfig`)
  - Fine-tuning with `SFTTrainer`
- **Experiment Tracking:** Integrated **MLflow** for metrics tracking and visualization.

---

### ⏭️ Next Step

Run the **first fine-tuning job** on Colab and analyze experiment logs in MLflow to iterate on:

- Learning rate
- Dataset size
- LoRA rank and dropout parameters

---

## 🧠 Vision

> To create a deployable LLM that doesn’t just _answer_, but _guides._  
> SOKRATIK aims to bring the calm reasoning of Stoicism into AI interactions —  
> an assistant that helps you think, not react.

---

## 💡 Future Directions

- 🧩 Model evaluation and human feedback alignment (RLHF / DPO)
- 🗣️ Web interface for interactive “philosophical dialogue”
- ☁️ Deployment via Hugging Face Spaces or FastAPI backend
- 📚 Continuous integration with CI/CD for automated retraining

---

## 🏗️ Maintainer

**Project Author:** Vardan Mittal
📍 IIT Gandhinagar | 🤖 Robotics & AI Enthusiast  
💬 _"The obstacle is the way."_
