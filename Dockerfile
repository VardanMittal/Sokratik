# 1. Start from Python 3.10 (Best compatibility)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. INSTALL DEPENDENCIES (THE FIX)
# We remove the manual wget.
# We point --extra-index-url to the official llama-cpp-python wheel repo.
# pip will automatically pick the correct wheel (manylinux) for Debian.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 6. Copy app code
COPY ./app ./app

# 7. Run Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]