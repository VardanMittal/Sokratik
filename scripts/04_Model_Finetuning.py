import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import mlflow

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BASEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASET_FILE = os.path.join(BASEFILE,"data/final/train.jsonl")
NEW_MODEL_NAME = "Sokratik-v1"
OUTPUT_DIR = os.path.join(BASEFILE,"results/") # Where checkpoints will be saved

# --- 1. LOAD DATASET ---
# DVC has already pulled this to our local folder
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print(f"Loaded {len(dataset)} training examples.")

# --- 2. LOAD BASE MODEL (QUANTIZED) ---
# This matches our baseline check setup exactly
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
    use_cache=False, # Required for training, must be True for inference later
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Crucial for Llama to train correctly

# --- 4. CONFIGURE LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, ### VRAM FIX 1: Reduced LoRA Rank from 64 to 16. (Fewer trainable params)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# --- 5. SFT CONFIG (Replaces TrainingArguments) ---
# This object now holds ALL training and dataset parameters.
sft_config = SFTConfig(
    # --- Args that moved ---
    dataset_text_field="text",
    max_length=1024,              ### VRAM FIX 2: Reduced Max Length from 2048 to 1024. (Big memory save)
    packing=False,

    # --- Standard Training Args ---
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1, ### VRAM FIX 3: Reduced Batch Size from 4 to 1. (Smallest possible)
    gradient_accumulation_steps=16,### VRAM FIX 4: Increased Accumulation from 4 to 16. (Keeps effective batch size 1*16=16)
    gradient_checkpointing=True,   ### VRAM FIX 5: Enable Gradient Checkpointing. (Trades compute for memory)
    gradient_checkpointing_kwargs={"use_reentrant": False}, # Silences a warning
    
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    
    # --- MLOps ---
    report_to="mlflow",
    run_name="run-1-llama3-8b-stoic"
)

# --- 6. THE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# --- 7. START TRAINING ---
print("Starting model training...")
trainer.train()
print("Training complete!")

# --- 8. SAVE THE MODEL ---
print(f"Saving model to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("Model saved locally.")