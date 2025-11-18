#!/bin/bash

# 1. Start the Backend (FastAPI) on internal port 8000
# using 'nohup' to keep it running in the background
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

echo "Waiting for API to start..."
sleep 5

# 2. Start the Frontend (Streamlit) on public port 7860
# We pass the internal API URL to the frontend env var
export API_URL="http://localhost:8000"
streamlit run frontend/ui.py --server.port 7860 --server.address 0.0.0.0