# 1. START FROM THE OFFICIAL IMAGE
# This image has Linux + Python + llama-cpp-python pre-installed and compiled.
# We don't need to build anything!
FROM ghcr.io/abetlen/llama-cpp-python:latest

# 2. Set working directory
WORKDIR /app

# 3. Install system tools (just in case)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install YOUR requirements
# Since we removed llama-cpp from this file, this step will be super fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code
COPY ./app ./app
COPY ./modules ./modules
COPY ./data ./data

# 6. Expose port
EXPOSE 7860

# 7. Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]