# 1. Use the OFFICIAL image (Guaranteed to work)
# This image uses Python 3.10 and has llama-cpp-python pre-installed correctly.
FROM ghcr.io/abetlen/llama-cpp-python:v0.2.90

WORKDIR /app

# 2. Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. OPTIMIZATION: Install CPU-Only PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY ./app ./app
COPY ./modules ./modules
COPY ./data ./data

# 6. Expose Port
EXPOSE 7860

# 7. Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]