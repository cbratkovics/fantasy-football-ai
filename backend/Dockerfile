# Production Dockerfile for Railway deployment
FROM python:3.10-slim

# Install system dependencies for MySQL client and other requirements
RUN apt-get update && apt-get install -y \
    pkg-config \
    python3-dev \
    default-libmysqlclient-dev \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway provides PORT env var
# Using shell form to allow environment variable substitution
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}