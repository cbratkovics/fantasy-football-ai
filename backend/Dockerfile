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

# Make all Python files executable
RUN chmod +x *.py

# Debug: List files
RUN echo "Files in /app:" && ls -la

# Debug: Show Python version
RUN python --version

# Add procps for debugging
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Force unbuffered output
ENV PYTHONUNBUFFERED=1

# Use shell to ensure output and environment variable expansion
CMD ["/bin/sh", "-c", "echo 'Container CMD executing' && echo 'PORT='$PORT && python -u start_railway.py 2>&1"]

# Alternative CMD options for debugging:
# CMD ["python", "-u", "railway_direct_test.py"]
# CMD ["python", "-u", "emergency_server.py"]
# CMD ["/bin/sh", "-c", "python -m http.server $PORT"]