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

# --- V3 OPTIMIZED CONFIGURATION ---
BASE_FILE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_FILE = os.path.join(BASE_FILE, "data/final/train_chat.jsonl")
NEW_MODEL_NAME = "Sokratik-v3-optimized" # New model name
OUTPUT_DIR = os.path.join(BASE_FILE, "results-v3-optimized") # New checkpoint dir

# --- 1. LOAD DATASET ---
# --- OPTIMIZATION #1: STREAMING ---
# This loads the dataset as an IterableDataset, reading from disk
# instead of loading the whole file into RAM.
dataset = load_dataset("json", data_files=DATASET_FILE, split="train", streaming=True)
print("Loaded dataset in streaming mode.")

# --- 2. LOAD BASE MODEL (QUANTIZED) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
# ... (model loading code is the same) ...
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False, 
    attn_implementation=None,
)
model.config.pretraining_tp = 1
# ... (tokenizer loading code is the same) ...
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. CONFIGURE LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    # --- OPTIMIZATION #2: Lower LoRA Rank ---
    r=8, # Was 16, now 8. Trains fewer params, uses less VRAM.
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
    num_train_epochs=1,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    
    logging_steps=10,
    learning_rate=2e-4,
    report_to="mlflow",
    run_name="run-4-chat-1-epoch-optimized" # New run name
)

# --- 5. THE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# --- 6. START TRAINING (WITH RESUME) ---
# ... (The resume logic is the same) ...
print("Starting V3 (Optimized) training...")
resume_from_checkpoint = False
if os.path.exists(sft_config.output_dir):
    checkpoints = [d for d in os.listdir(sft_config.output_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) > 0:
        print(f"Found {len(checkpoints)} existing checkpoints. Resuming from latest.")
        resume_from_checkpoint = True
    else:
        print("Output directory exists but no checkpoints found. Starting from scratch.")
else:
    print("No output directory found. Starting from scratch.")

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# --- 7. SAVE THE FINAL MODEL ---
print(f"Saving final V3 model to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("--- V3 Optimized Training Complete! ---")

# --- 8. MLOPS: PUSH TO HUB ---
# ... (The push-to-hub logic is the same, just update the names) ...
print(f"Pushing {NEW_MODEL_NAME} to Hugging Face Hub...")
try:
    hf_token = None
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_WRITE_TOKEN')
    except ImportError:
        hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        hf_model_repo = f"vardan10/{NEW_MODEL_NAME}"
        
        from huggingface_hub import create_repo, HfApi
        create_repo(hf_model_repo, exist_ok=True, repo_type="model", token=hf_token)

        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=NEW_MODEL_NAME, # Upload the new optimized model
            repo_id=hf_model_repo,
            path_in_repo="."
        )
        
        print(f"Successfully pushed model to https://huggingface.co/{hf_model_repo}")
    else:
        print("HF_WRITE_TOKEN not found. Skipping push to Hub. Please save manually.")

except Exception as e:
    print(f"Error pushing to Hub. Your model is saved locally in '{NEW_MODEL_NAME}'. Error: {e}")
