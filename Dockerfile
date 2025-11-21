# 1. Use a specific, pinned version of the official image
# This guarantees Python 3.10 and a working llama-cpp-python v0.2.90
FROM ghcr.io/abetlen/llama-cpp-python:v0.2.90

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
# We REMOVE the extra-index-url because the library is already installed in this image!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY ./app ./app
COPY ./modules ./modules
COPY ./data ./data

# 5. Expose port
EXPOSE 7860

# 6. Run the API Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]