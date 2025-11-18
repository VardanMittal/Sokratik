import os
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pydantic import BaseModel

# --- CONFIGURATION ---
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "vardan10/Sokratik-v3"
SYSTEM_PROMPT = "You are a Stoic philosopher. Answer with wisdom, logic, and tranquility, drawing upon the principles of Stoicism."

# --- DATA MODELS ---
class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    answer: str

# --- 1. GET TOKEN SECURELY ---
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not found in environment variables.")
    print("If you are using a gated model like Llama 3, this will fail.")

# --- GLOBAL VARIABLES ---
model = None
tokenizer = None

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
    # Fix: Removed incompatible 'llm_int8_enable_fp32_cpu_offload' argument
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )
    
    # --- 3. PASS TOKEN TO TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"--- Loading adapter: {ADAPTER_ID} ---")
    
    # --- 4. PASS TOKEN TO ADAPTER ---
    model = PeftModel.from_pretrained(
        base_model, 
        ADAPTER_ID, 
        token=HF_TOKEN
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

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    The main endpoint to generate a Stoic response.
    """
    if not model or not tokenizer:
        return {"error": "Model is not loaded yet. Please wait."}

    # 1. Format the prompt using the Llama 3 Chat Template
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.prompt},
    ]
    
    prompt_formatted = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 2. Tokenize the input
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda")

    # 3. Generate the response
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # 4. Decode and parse the response
    response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Use our parsing logic from the evaluation script
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    if assistant_marker in response_full:
        response_answer = response_full.split(assistant_marker)[-1].strip()
    else:
        # Fallback if the marker isn't found
        response_answer = response_full

    return GenerateResponse(answer=response_answer)