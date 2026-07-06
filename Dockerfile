FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create uploads directory
RUN mkdir -p api/uploads

# Run on port 7860 for HF Spaces
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
