# 1. Use full Debian image (NOT slim)
FROM python:3.10

# 2. Working directory
WORKDIR /app

# 3. System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install python deps (exclude torch, bitsandbytes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Install llama-cpp-python (correct Linux wheel)
RUN wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_31_x86_64.whl \
    && pip install llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_31_x86_64.whl \
    && rm llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_31_x86_64.whl

# 6. Copy app code
COPY ./app ./app

# 7. Expose port & run
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
