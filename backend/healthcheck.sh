#!/bin/sh
# Healthcheck script for Railway deployment

# Get the PORT from environment, default to 8000
PORT=${PORT:-8000}

# Check health endpoint
curl -f http://localhost:${PORT}/health || exit 1