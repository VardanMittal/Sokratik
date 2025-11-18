# 1. Start from a lightweight, official Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (Caching Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY ./app ./app
COPY ./frontend ./frontend
COPY start.sh .

# 5. Make the start script executable
# This is critical for Linux containers
RUN chmod +x start.sh

# 6. Expose port 7860 (Required by Hugging Face Spaces)
EXPOSE 7860

# 7. Run the start script
CMD ["./start.sh"]