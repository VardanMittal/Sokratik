# 1. Start from a lightweight, official Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (Caching Layer)
# Docker checks this layer first. If requirements.txt hasn't changed,
# it skips installing libraries again, making builds super fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the actual application code
# We do this last because code changes often.
COPY ./app ./app

# 5. Expose the port (Standard for FastAPI)
EXPOSE 7860

# 6. Run the Server
# Note: Hugging Face Spaces specifically expects port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]