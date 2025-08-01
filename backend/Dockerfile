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

# Set Python path
ENV PYTHONPATH=/app

# Make start scripts executable
RUN chmod +x start_app.py start_simple.py start_railway.py

# Add health check for local testing (disabled for Railway - they handle it)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:$PORT/health || exit 1

# Railway provides PORT env var
# Use Railway-optimized startup script with diagnostics and fallbacks
CMD ["python", "start_railway.py"]