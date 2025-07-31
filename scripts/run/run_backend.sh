#!/bin/bash

# Activate conda environment
source ~/anaconda3/bin/activate agentic_ai_env

# Set PYTHONPATH
export PYTHONPATH=/Users/christopherbratkovics/Desktop/fantasy-football-ai:/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend

# Unset DATABASE_URL to use .env file
unset DATABASE_URL

# Navigate to backend
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai/backend

# Run migrations using conda python
echo "Running database migrations..."
python -m alembic upgrade head

# Initialize database
echo "Initializing database..."
cd ..
python scripts/init_database.py

# Start backend
echo "Starting backend server..."
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000