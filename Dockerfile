# 1. Start from Python 3.10 (Best compatibility)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system tools (wget is needed to download the wheel)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. FORCE INSTALL llama-cpp-python from a specific pre-built wheel
# This URL points to a pre-compiled binary for Linux x86_64, Python 3.10, AVX2 support
# This completely bypasses the "Building wheel" step.
RUN wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl \
    && pip install llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl \
    && rm llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl

# 6. Copy app code
COPY ./app ./app

# 7. Run Uvicorn directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]