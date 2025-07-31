#!/bin/sh
# Start script for Railway deployment

# Use PORT environment variable from Railway, default to 8000
PORT=${PORT:-8000}

echo "Starting FastAPI on port $PORT..."

# Run uvicorn with the PORT from environment
exec uvicorn main:app --host 0.0.0.0 --port $PORT