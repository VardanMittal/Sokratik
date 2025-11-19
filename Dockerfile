FROM python:3.12-slim

WORKDIR /app

# Install system build tools (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Increase timeout because downloading packages can be slow
RUN pip install --no-cache-dir -r requirements.txt --timeout 100

COPY ./app ./app
COPY ./frontend ./frontend
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]