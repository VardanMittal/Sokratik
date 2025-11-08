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
from trl import SFTTrainer

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BASEFILE = os.path.join(os.path.abspath(__file__), "..")
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
    attn_implementation="flash_attention_2" # Speeds up training on T4/A100
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Crucial for Llama to train correctly

# --- 3. CONFIGURE LoRA ---
# This defines the tiny "adapters" we will actually train
peft_config = LoraConfig(
    lora_alpha=16, # How much weight to give the new adapters
    lora_dropout=0.1,
    r=64,          # The size of the adapters (larger = smarter but slower)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers for best results
)

# --- 4. TRAINING ARGUMENTS ---
# These control the hyperparameters. We'll tweak these later.
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,          # START SMALL! 1 epoch to test if it works.
    per_device_train_batch_size=4, # Keep small for Colab GPU VRAM
    gradient_accumulation_steps=4, # Simulates larger batch size (4*4 = 16)
    optim="paged_adamw_32bit",   # Memory-efficient optimizer
    save_steps=50,               # Save a checkpoint every 50 steps
    logging_steps=10,            # Log metrics every 10 steps
    learning_rate=2e-4,          # Standard starting rate for QLoRA
    weight_decay=0.001,
    fp16=True,                   # Use mixed precision training (faster, less memory)
    bf16=False,                  # T4 GPU doesn't support pure BF16 well, use FP16
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,        # Speeds up training by grouping similar length text
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# --- 5. THE TRAINER ---
# SFTTrainer handles the heavy lifting of the training loop
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # The column name in our JSONL file
    max_seq_length=None,       # Auto-detect max length
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# --- 6. START TRAINING ---
print("Starting training...")
trainer.train()

# --- 7. SAVE THE MODEL ---
print(f"Saving model to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("DONE!")