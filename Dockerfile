# ---------------------------------------------------------
# 1. Use full Debian Python (NOT slim)
# ---------------------------------------------------------
FROM python:3.10

# ---------------------------------------------------------
# 2. Working directory
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# 3. Install system dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 4. Install Python dependencies
# (requirements.txt should NOT contain torch or bitsandbytes)
# ---------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# 5. Install llama-cpp-python using the official CPU wheel repo
# (pip will automatically select correct manylinux wheel)
# ---------------------------------------------------------
RUN pip install llama-cpp-python\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ---------------------------------------------------------
# 6. Copy your application code
# ---------------------------------------------------------
COPY ./app ./app

# ---------------------------------------------------------
# 7. Expose port + run API
# ---------------------------------------------------------
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
