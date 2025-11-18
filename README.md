---
title: Sokratik API
emoji: 🏛️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🏛️ SOKRATIK — The Stoic LLM

> _"Man is not worried by real problems so much as by his imagined anxieties about real problems."_  
> — Epictetus

**SOKRATIK** is a **production-ready MLOps-driven Large Language Model** that brings Stoic philosophy to life through an AI fine-tuned on the complete works of **Marcus Aurelius**, **Epictetus**, and **Seneca**.

The project successfully fine-tunes **Llama 3.1 8B Instruct** on Stoic philosophical texts using **QLoRA (4-bit quantization)** and deploys via **FastAPI** with a fully reproducible, versioned MLOps pipeline featuring DVC for data management and MLflow for experiment tracking.

---

## � Deployment

This API is deployed on **Hugging Face Spaces** using Docker. The deployment automatically loads the fine-tuned adapter model and serves it via FastAPI.

**Live API:** [View on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/Sokratik)  
**Fine-tuned Model:** [`vardan10/Sokratik-v3`](https://huggingface.co/vardan10/Sokratik-v3)

### Required Environment Variables

For Hugging Face Spaces deployment, set the following in your Space settings:

- `HF_TOKEN`: Your Hugging Face access token (required for gated models like Llama 3.1)

### Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face with Docker SDK
2. Connect your GitHub repository for automatic deployments (CD)
3. Set the `HF_TOKEN` secret in Space settings
4. Push to your repository - the Space will automatically rebuild and deploy
5. Access your API at the provided Space URL

## �📊 Project Overview

| Aspect                | Details                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------- |
| **Base Model**        | `meta-llama/Llama-3.1-8B-Instruct`                                                       |
| **Fine-tuned Model**  | `vardan10/Sokratik-v3` (Available on Hugging Face Hub)                                   |
| **Training Strategy** | QLoRA (4-bit quantization) with resumable checkpointing                                  |
| **Training Compute**  | Google Colab T4 GPU                                                                      |
| **Core Libraries**    | `transformers`, `peft`, `trl (SFTTrainer)`, `bitsandbytes`, `torch`                      |
| **MLOps Stack**       | DVC for data versioning, MLflow for experiment tracking                                  |
| **Storage**           | Google Drive (as DVC remote)                                                             |
| **Deployment**        | FastAPI + Docker on Hugging Face Spaces                                                  |
| **Goal**              | Production-ready LLM that embodies Stoic philosophy with full MLOps lifecycle management |

---

## 🧱 Project Structure

```text
SOKRATIK/
│
├── app/
│   ├── __init__.py           # Package initializer
│   └── main.py               # FastAPI application with model loading
│
├── data/
│   ├── raw/                  # Original Stoic texts (4 books)
│   │   ├── meditations.txt
│   │   ├── Letter_from_a_stoic.txt
│   │   ├── The_Discourses.txt
│   │   └── The_Enchiridion.txt
│   ├── processed/            # Cleaned and unified text corpus
│   │   └── stoic_corpus.txt
│   └── final/                # ML-ready datasets (JSONL format)
│       ├── train.jsonl       # Original text-based format
│       └── train_chat.jsonl  # Chat-formatted for instruction tuning
│
├── scripts/                  # Core automation and training scripts
│   ├── 01_baseline_check.py            # Evaluate base Llama 3 performance
│   ├── 02_cleaning_text.py             # Clean raw text and remove artifacts
│   ├── 03_Formatting_Data.py           # Convert corpus to train.jsonl
│   ├── 03b_Formatting_Data.py          # Convert to chat format
│   ├── 04_Model_Finetuning.py          # Initial fine-tuning script
│   ├── 05_evaluation_script.py         # Model evaluation
│   └── 06_Optimized_Finetuning_v3.py   # Production training with checkpointing
│
├── mlruns/                   # MLflow experiment tracking data
├── Dockerfile                # Docker configuration for deployment
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1️⃣ Setup Environment

```bash
# Clone the repository
git clone https://github.com/VardanMittal/Sokratik.git
cd Sokratik

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Pull Data via DVC

This project uses **DVC (Data Version Control)** to manage large files such as cleaned corpora and training datasets.

```bash
# Authenticate DVC (opens a browser for Google Drive)
dvc pull
```

Once complete, your `train_chat.jsonl` and `stoic_corpus.txt` will be automatically placed in the right folders.

### 3️⃣ Run Fine-Tuning

```bash
# Log in to Hugging Face
huggingface-cli login

# Run the optimized fine-tuning script (with checkpointing)
python scripts/06_Optimized_Finetuning_v3.py
```

Monitor all experiment metrics (loss curves, learning rate, etc.) directly from **MLflow**:

```bash
mlflow ui
```

### 4️⃣ Local API Testing

```bash
# Run the FastAPI server locally
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Or use Docker
docker build -t sokratik-api .
docker run -p 7860:7860 -e HF_TOKEN=your_token_here sokratik-api
```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:7860/
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Generate Response (Example)

The API currently provides a health check endpoint. To add inference capabilities, you can extend the FastAPI app with a `/generate` endpoint that uses the loaded model and tokenizer.

---

## 📅 Project Milestones

### ✅ Phase 1: Data Foundation & Pipeline

- **Repo Setup:** Git + virtual environment initialized with proper `.gitignore`
- **Baseline Established:** `01_baseline_check.py` evaluates pre-trained Llama 3.1 behavior
- **Data Sourcing:** Curated 4 complete Stoic texts:
  - _Meditations_ by Marcus Aurelius
  - _Letters from a Stoic_ by Seneca
  - _The Discourses_ by Epictetus
  - _The Enchiridion_ by Epictetus
- **Cleaning Pipeline:** Built `02_cleaning_text.py` to unify and clean all texts → `stoic_corpus.txt`
- **Data Formatting:**
  - `03_Formatting_Data.py` → converts corpus into `train.jsonl`
  - `03b_Formatting_Data.py` → creates chat-formatted `train_chat.jsonl` for instruction tuning
- **Data Versioning:** Integrated **DVC** with Google Drive for reproducibility
- **Security:** Secured GCloud credentials in `.dvc/config.local` (git-ignored)

### ✅ Phase 2: Model Training & Optimization

- **Training Strategy:** Implemented **QLoRA (4-bit quantization)** to fit 8B model on single T4 GPU
- **Fine-Tuning Pipeline:** Created production-ready training scripts:
  - `04_Model_Finetuning.py` → Initial implementation
  - `06_Optimized_Finetuning_v3.py` → Optimized version with:
    - Resumable checkpointing (saves every 25 steps)
    - Gradient accumulation (effective batch size = 16)
    - Optimal hyperparameters (1 epoch, lr=2e-4)
    - LoRA rank 16 targeting 7 modules
- **Experiment Tracking:** Full **MLflow** integration for metrics tracking and visualization
- **Model Evaluation:** `05_evaluation_script.py` for testing model responses

### ✅ Phase 3: Deployment & Production

- **API Development:** Built FastAPI application (`app/main.py`) with:
  - Async model loading with lifespan events
  - 4-bit quantized inference
  - Proper HF token authentication
  - Health check endpoint
- **Containerization:** Created production `Dockerfile` with:
  - Multi-stage build optimization
  - Layer caching for fast rebuilds
  - Port 7860 for HF Spaces compatibility
- **Model Publishing:** Uploaded fine-tuned adapter to Hugging Face Hub as `vardan10/Sokratik-v3`
- **Deployment:** Configured for **Hugging Face Spaces** with proper YAML frontmatter
- **Documentation:** Comprehensive README with setup, deployment, and usage instructions

---

## 🔧 Technical Highlights

### Training Configuration

- **LoRA Parameters:**

  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0.1
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

- **Training Setup:**
  - Batch size: 1 (per device)
  - Gradient accumulation: 16 steps
  - Effective batch size: 16
  - Learning rate: 2e-4
  - Optimizer: `paged_adamw_32bit`
  - Max sequence length: 1024 tokens
  - Training epochs: 1 (optimal for this dataset)
  - Gradient checkpointing: Enabled
  - Sequence packing: Enabled

### Deployment Architecture

- **Model Loading:** Async lifespan events for proper initialization
- **Quantization:** 4-bit NF4 with FP16 compute dtype
- **Device Management:** Auto device mapping with CPU offloading support
- **Authentication:** Secure HF token handling via environment variables
- **API Framework:** FastAPI with Uvicorn ASGI server
- **Container:** Python 3.12 slim image with optimized layer caching

---

## 🧠 Vision

> To create a deployable LLM that doesn't just _answer_, but _guides._  
> SOKRATIK brings the calm reasoning of Stoicism into AI interactions —  
> an assistant that helps you think, not react.

---

## 💡 Future Enhancements

- 🧩 Model evaluation with human feedback alignment (RLHF / DPO)
- 🗣️ Interactive web interface for philosophical dialogue
- 📊 Inference endpoint with prompt templates for better responses
- 🔄 Continuous integration with CI/CD for automated retraining
- 📈 A/B testing framework for model versions
- 🎯 Fine-tuning on user feedback for continuous improvement

---

## 🏗️ Maintainer

**Project Author:** Vardan Mittal  
📍 IIT Gandhinagar | 🤖 Robotics & AI Enthusiast  
💬 _"The obstacle is the way."_
