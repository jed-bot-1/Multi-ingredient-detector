# Use a lightweight Python base image
FROM python:3.10-slim

# Install OS-level dependencies (required for onnxruntime & OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files including your model and app
COPY . .

# Set Railway port environment variable default
ENV PORT=8000

# Expose port
EXPOSE ${PORT}

# Start FastAPI with dynamic port for Railway
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
