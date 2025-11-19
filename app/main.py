import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
REPO_ID = "vardan10/Sokratik-v3-GGUF"
# CHECK YOUR HF REPO FOR THE EXACT NAME! 
# It might be "Sokratik-v3.Q4_K_M.gguf" or similar.
FILENAME = "Sokratik-v3.Q4_K_M.gguf" 

SYSTEM_PROMPT = "You are a Stoic philosopher. Answer with wisdom, logic, and tranquility, drawing upon the principles of Stoicism."

# --- GLOBAL VARIABLES ---
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    print(f"--- Downloading GGUF model: {REPO_ID} / {FILENAME} ---")
    
    try:
        # 1. Download the GGUF file from Hugging Face to local cache
        # This is fast because it caches the file
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=os.getenv("HF_TOKEN")
        )
        
        print(f"--- Loading model into RAM... ---")
        # 2. Load the model with llama.cpp
        # n_ctx=2048: The conversation history limit
        # verbose=False: Reduces log noise
        llm = Llama(
            model_path=model_path, 
            n_ctx=2048, 
            verbose=True
        )
        
        print("--- Sokratik GGUF Loaded Successfully! ---")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        print("Did you check the FILENAME variable matches your HF repo?")
    
    yield
    print("--- Shutting down... ---")

app = FastAPI(lifespan=lifespan)

# --- DATA MODELS ---
class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": (llm is not None)}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not llm:
        return {"answer": "Error: Model is not loaded. Check server logs."}

    # 1. Format the prompt manually using Llama 3 structure
    # This ensures the model knows who it is (System) and who you are (User)
    formatted_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # 2. Generate
    output = llm(
        formatted_prompt,
        max_tokens=256, # Limit response length
        stop=["<|eot_id|>", "<|end_of_text|>"], # Stop when finished
        echo=False, # Don't repeat the prompt
        temperature=0.6 # Creativity level
    )

    # 3. Parse the result
    answer = output["choices"][0]["text"].strip()
    return GenerateResponse(answer=answer)