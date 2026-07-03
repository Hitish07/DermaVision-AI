# Use official Python lightweight image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# First copy requirements from the root project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn sqlalchemy python-multipart pydantic

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
