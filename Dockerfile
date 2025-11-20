# 1. Use Python 3.10 (Stable standard)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system tools (curl/wget needed for downloading)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Install standard requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. FORCE INSTALL llama-cpp-python (The Fix)
# We download the specific 'manylinux' wheel for Python 3.10 (cp310).
# This file is pre-compiled and installs in 2 seconds.
RUN wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_17_x86_64.whl \
    && pip install llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_17_x86_64.whl \
    && rm llama_cpp_python-0.2.90-cp310-cp310-manylinux_2_17_x86_64.whl

# 6. Copy app code
COPY ./app ./app

# 7. Run Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]