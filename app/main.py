from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not found in environment variables.")
    print("If you are using a gated model like Llama 3, this will fail.")

model = None
tokenizer = None

# --- Model Configuration ---
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "vardan10/Sokratik-v3" # Your "winner" model from the Hub

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    
    print(f"--- Loading base model: {BASE_MODEL_ID} ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- 2. PASS TOKEN TO BASE MODEL ---
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True,
        token=HF_TOKEN  # <-- CRITICAL FIX
    )
    
    # --- 3. PASS TOKEN TO TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN  # <-- CRITICAL FIX
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"--- Loading adapter: {ADAPTER_ID} ---")
    
    # --- 4. PASS TOKEN TO ADAPTER ---
    # (Sometimes PEFT needs verification to download config.json)
    model = PeftModel.from_pretrained(
        base_model, 
        ADAPTER_ID, 
        token=HF_TOKEN # <-- CRITICAL FIX
    )
    model.eval()
    
    print("--- Model and Tokenizer Loaded Successfully ---")
    yield
    print("--- Shutting down... ---")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    if model and tokenizer:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "loading", "model_loaded": False}