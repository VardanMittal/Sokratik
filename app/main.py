import os
import time
import logging
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from contextlib import asynccontextmanager

# --- IMPORT RAG MODULES ---
# These allow the API to "think" and "remember"
from modules.rag_graph import build_graph
from modules.telemetry import log_interaction

# --- CONFIGURATION ---
REPO_ID = "vardan10/Sokratik-v3-GGUF"
FILENAME = "Sokratik-v3.Q4_K_M.gguf"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sokratik_backend")

# --- GLOBAL STATE ---
rag_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_app
    logger.info("--- Startup: Initializing Sokratik Backend ---")
    
    try:
        # 1. Download Model
        logger.info(f"Downloading {FILENAME}...")
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=FILENAME, 
            token=os.getenv("HF_TOKEN")
        )
        
        # 2. Load Engine (CPU Optimized)
        logger.info("Loading Llama CPP...")
        llm = Llama(
            model_path=model_path, 
            n_ctx=2048, 
            verbose=False
        )
        
        # 3. Build the RAG Graph
        logger.info("Building LangGraph Logic...")
        # We pass the loaded LLM into the graph builder
        rag_app = build_graph(llm)
        
        logger.info("--- Sokratik Brain Ready! ---")
    except Exception as e:
        logger.error(f"CRITICAL STARTUP ERROR: {e}")
    
    yield
    logger.info("--- Shutdown ---")

app = FastAPI(lifespan=lifespan)

# --- DATA MODELS ---
class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    answer: str
    meta: dict

@app.get("/")
def health_check():
    """Health check for the frontend to ping."""
    return {
        "status": "ok", 
        "rag_loaded": (rag_app is not None),
        "service": "Sokratik-Backend"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Main inference endpoint.
    1. Receives prompt.
    2. Runs RAG Graph (Retrieve -> Augment -> Generate).
    3. Logs telemetry in background.
    4. Returns answer.
    """
    if not rag_app:
        return {"answer": "Backend initializing... please wait 1 min.", "meta": {}}

    start_time = time.time()
    
    try:
        # 1. Run the Graph
        # .invoke() runs the full StateGraph from Start to End
        result = rag_app.invoke({"question": request.prompt})
        
        answer = result.get("answer", "Error generating response.")
        context = result.get("context", [])
        
        duration = time.time() - start_time
        
        # 2. Schedule Logging (Don't block response)
        background_tasks.add_task(log_interaction, request.prompt, answer, context)

        return GenerateResponse(
            answer=answer,
            meta={
                "duration": round(duration, 2),
                "retrieved_docs": len(context)
            }
        )
    except Exception as e:
        logger.error(f"Generation Error: {e}")
        return {"answer": f"Internal Error: {str(e)}", "meta": {}}