import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from contextlib import asynccontextmanager

# --- CONFIG ---
REPO_ID = "vardan10/Sokratik-v3-GGUF"
FILENAME = "Sokratik-v3.Q4_K_M.gguf" 
SYSTEM_PROMPT = "You are a Stoic philosopher. Answer with wisdom, logic, and tranquility."

llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    print(f"--- Downloading GGUF model... ---")
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=os.getenv("HF_TOKEN")
        )
        print(f"--- Loading model... ---")
        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        print("--- Sokratik Ready! ---")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    yield

app = FastAPI(lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    answer: str

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": (llm is not None)}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not llm: return {"answer": "Model not loaded."}
    
    formatted_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    output = llm(formatted_prompt, max_tokens=256, stop=["<|eot_id|>"], echo=False)
    return GenerateResponse(answer=output["choices"][0]["text"].strip())