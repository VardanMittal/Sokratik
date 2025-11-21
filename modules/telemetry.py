import os
import datetime
import json
import tempfile
from huggingface_hub import HfApi
import threading

# --- CONFIG ---
# This is where we will save the logs
LOG_DATASET_ID = "vardan10/sokratik-logs" 
HF_TOKEN = os.getenv("COLAB_WRITE")

def _log_worker(prompt: str, answer: str, context: list):
    """
    Background worker that does the actual uploading.
    """
    if not HF_TOKEN:
        return

    timestamp = datetime.datetime.now().isoformat()
    
    # Prepare the log entry
    log_entry = {
        "timestamp": timestamp,
        "prompt": prompt,
        "answer": answer,
        "retrieved_context": context,
        # We pre-format this for future SFTTrainer runs!
        "messages": [
            {"role": "system", "content": "You are a Stoic philosopher..."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
    }

    api = HfApi(token=HF_TOKEN)
    try:
        # Ensure the dataset repo exists
        api.create_repo(repo_id=LOG_DATASET_ID, repo_type="dataset", exist_ok=True)
        
        # Save to a temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(log_entry, tmp)
            tmp_path = tmp.name
            
        # Upload the file to the Hub
        filename = f"logs/{timestamp.replace(':','-')}.json"
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=LOG_DATASET_ID,
            repo_type="dataset"
        )
        
        # Clean up
        os.remove(tmp_path)
        print("✅ Interaction logged to Hugging Face.")
        
    except Exception as e:
        print(f"❌ Logging failed: {e}")

def log_interaction(prompt, answer, context):
    """
    Public function that spawns a thread.
    This ensures the user doesn't wait for the upload.
    """
    thread = threading.Thread(target=_log_worker, args=(prompt, answer, context))
    thread.start()