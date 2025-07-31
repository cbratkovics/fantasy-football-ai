#!/bin/sh
# Start script for Railway deployment

# Use PORT environment variable from Railway, default to 8000
PORT=${PORT:-8000}

echo "Starting FastAPI on port $PORT..."
echo "Environment PORT: ${PORT}"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"

# Run uvicorn with the PORT from environment
exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info