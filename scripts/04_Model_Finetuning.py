import os
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- MLOPS: MLFLOW SETUP ---
# This tells MLflow which "project" to log to.
mlflow.set_experiment("project-Sokratik-Finetuning")

# --- V2 CONFIGURATION ---
BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# --- CHANGE #1: Use the new chat-formatted dataset ---
DATASET_FILE = os.path.join(BASE_FILE,"data/final/train_chat.jsonl")
# --- CHANGE #2: Define new model/output names for v2 ---
NEW_MODEL_NAME = "Sokratik-v3"
OUTPUT_DIR = "./results-v3"

# --- 1. LOAD DATASET ---
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
    attn_implementation=None, # No Flash Attention on T4
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. CONFIGURE LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, # Kept small for VRAM
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 4. TRAINING ARGUMENTS ---
sft_config = SFTConfig(
    dataset_text_field=None,
    packing=True,
    max_length=1024,
    output_dir=OUTPUT_DIR,

    # --- THE "GOLDILOCKS" FIX ---
    num_train_epochs=1,  # <-- BACK TO 1 EPOCH
    
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_strategy="epoch", 
    logging_steps=10,
    learning_rate=2e-4,
    # ... (rest of the args are the same) ...
    report_to="mlflow",
    
    run_name="run-3-chat-1-epoch" # <-- V3 run name
)

# --- 5. THE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer, # Use 'processing_class' (new TRL API)
    args=sft_config,
)

# --- 6. START TRAINING ---
print("Starting V3 ('Goldilocks') training...")
trainer.train()


# --- 7. SAVE THE MODEL ---
print(f"Saving V3 model to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("--- V2 Training Complete! ---")