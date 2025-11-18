import os
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- MLOPS: MLFLOW SETUP ---
mlflow.set_experiment("project-Sokratik-Finetuning")

# --- V3 CONFIGURATION ---
BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_FILE = os.path.join(BASE_FILE, "data/final/train_chat.jsonl")
NEW_MODEL_NAME = "Sokratik-v3"
OUTPUT_DIR = os.path.join(BASE_FILE, "results-v3") # Checkpoints save here

# --- 1. LOAD DATASET ---
# We load the full dataset. This is fine since RAM is not the issue.
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print(f"Loaded {len(dataset)} chat-formatted training examples.")

# --- 2. LOAD BASE MODEL (QUANTIZED) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False, 
    attn_implementation=None,
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. CONFIGURE LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, # Back to 16, since VRAM wasn't the issue
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 4. TRAINING ARGUMENTS ---
sft_config = SFTConfig(
    dataset_text_field=None,
    packing=True, # Packing is fine, the warnings are just warnings
    max_length=1024,
    output_dir=OUTPUT_DIR,
    num_train_epochs=1, # Our "Goldilocks" 1 epoch
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    
    # --- CHECKPOINTING STRATEGY ---
    save_strategy="steps",     # Save based on steps
    save_steps=25,             # Save a checkpoint every 25 steps
    save_total_limit=3,        # Only keep the 3 most recent checkpoints
    
    logging_steps=10,
    learning_rate=2e-4,
    report_to="mlflow",
    run_name="run-3-chat-1-epoch-resumable"
)

# --- 5. THE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# --- 6. START TRAINING (WITH ROBUST RESUME) ---
print("Starting V3 ('Goldilocks') training...")

# --- THIS IS THE FIX ---
# Check if output_dir exists and has a checkpoint
resume_from_checkpoint = False
if os.path.exists(sft_config.output_dir):
    checkpoints = [d for d in os.listdir(sft_config.output_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) > 0:
        # Sort checkpoints by step number (e.g., checkpoint-50)
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        resume_path = os.path.join(sft_config.output_dir, latest_checkpoint)
        print(f"Found existing checkpoint. Resuming from: {resume_path}")
        resume_from_checkpoint = resume_path
    else:
        print("Output directory exists but no checkpoints found. Starting from scratch.")
else:
    print("No output directory found. Starting from scratch.")

# Call train() with the correct resume flag
# This will be `False` on the first run, and a path string on subsequent runs
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# --- 7. SAVE THE FINAL MODEL ---
print(f"Saving final V3 model to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("--- V3 Training Complete! ---")

# --- 8. MLOPS: PUSH TO HUB ---
print(f"Pushing {NEW_MODEL_NAME} to Hugging Face Hub...")
try:
    hf_token = None
    try:
        from google.colab import userdata
        hf_token = userdata.get('COLAB_WRITE')
    except ImportError:
        hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        hf_model_repo = f"vardan10/{NEW_MODEL_NAME}"
        
        from huggingface_hub import create_repo, HfApi
        create_repo(hf_model_repo, exist_ok=True, repo_type="model", token=hf_token)

        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=NEW_MODEL_NAME,
            repo_id=hf_model_repo,
            path_in_repo="."
        )
        
        print(f"Successfully pushed model to https://huggingface.co/{hf_model_repo}")
    else:
        print("HF_WRITE_TOKEN not found. Skipping push to Hub. Please save manually.")

except Exception as e:
    print(f"Error pushing to Hub. Your model is saved locally in '{NEW_MODEL_NAME}'. Error: {e}")