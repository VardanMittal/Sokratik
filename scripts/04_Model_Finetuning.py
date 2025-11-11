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
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# --- CHANGE #1: Use the new chat-formatted dataset ---
DATASET_FILE = "data/final/train_chat.jsonl"
# --- CHANGE #2: Define new model/output names for v2 ---
NEW_MODEL_NAME = "Sokratik-v2"
OUTPUT_DIR = "./results-v2"

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
    # --- CHANGE #3: Set 'dataset_text_field' to None ---
    # TRL will now automatically look for the "messages" column
    dataset_text_field=None,
    
    # --- CHANGE #4: Enable 'packing' ---
    # This is much more-efficient for chat data
    packing=True,

    max_length=1024, # Keep VRAM usage low
    
    # --- Args from TrainingArguments ---
    output_dir=OUTPUT_DIR,

    # --- CHANGE #5: Train for longer ---
    num_train_epochs=3, # v1 was 1 epoch, v2 will be 3
    
    per_device_train_batch_size=1, # VRAM FIX
    gradient_accumulation_steps=16, # VRAM FIX
    gradient_checkpointing=True, # VRAM FIX
    
    optim="paged_adamw_32bit",
    save_strategy="epoch", # Save a checkpoint after each epoch
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
    report_to="mlflow",
    
    # --- CHANGE #6: New MLflow run name ---
    run_name="run-2-chat-3-epochs"
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
print("Starting V2 training...")
trainer.train()

# --- 7. SAVE THE MODEL ---
print(f"Saving V2 model to {NEW_MODEL_NAME}...")
# We save the adapter, not the full model
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

# --- 8. MLOPS: PUSH TO HUB ---
# This saves your work to the cloud permanently.
print(f"Pushing {NEW_MODEL_NAME} to Hugging Face Hub...")
try:
    trainer.model.push_to_hub(NEW_MODEL_NAME)
    tokenizer.push_to_hub(NEW_MODEL_NAME)
    print(f"Successfully pushed model to https://huggingface.co/vardan10/{NEW_MODEL_NAME}")
except Exception as e:
    print(f"Error pushing to Hub. Manually push later. Error: {e}")

print("--- V2 Training Complete! ---")