# 1. Start from a lightweight, official Python image
FROM python:3.10

# 2. Set the working directory
WORKDIR /app

# 3. Install system build tools 
# (We still keep these just in case, but we hope not to use them)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .

# --- THE FIX IS HERE ---
# We add '--extra-index-url' to point to a repository of pre-built wheels.
# This prevents the "Building wheel..." step that caused the timeout.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 5. Copy application code
COPY ./app ./app
COPY ./frontend ./frontend
COPY start.sh .

# 6. Permissions and Ports
RUN chmod +x start.sh
EXPOSE 7860

# 7. Run
CMD ["./start.sh"]
