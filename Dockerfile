# 1. Use Python 3.10 (Stable standard)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system tools
# We keep build-essential/cmake just in case other packages need them,
# but llama-cpp-python won't need to compile anymore.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Install standard requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. INSTALL llama-cpp-python (The Fast Way)
RUN wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl \
    && pip install llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl \
    && rm llama_cpp_python-0.2.90-cp310-cp310-linux_x86_64.whl

# 6. Copy app code
COPY ./app ./app

# 7. Run Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]