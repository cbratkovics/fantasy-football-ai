#!/bin/bash

echo "Running Fantasy Football AI with Conda environment..."

# Navigate to project root
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai

# Activate conda environment
source ~/anaconda3/bin/activate agentic_ai_env

# Set PYTHONPATH
export PYTHONPATH=/Users/christopherbratkovics/Desktop/fantasy-football-ai:/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend:$PYTHONPATH

# Check PostgreSQL status
echo "Checking PostgreSQL..."
if ! pg_isready -q; then
    echo "PostgreSQL is not running. Starting it..."
    brew services start postgresql@14
    sleep 3
fi

# Check Redis status
echo "Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis is not running. Starting it..."
    brew services start redis
    sleep 2
fi

# Create database if not exists
echo "Setting up database..."
psql postgres -c "SELECT 1 FROM pg_user WHERE usename = 'fantasy_user';" | grep -q 1 || psql postgres -f setup_postgres.sql

# Run migrations
echo "Running database migrations..."
cd backend
alembic upgrade head

# Initialize database
echo "Initializing database..."
cd ..
python scripts/init_database.py

echo "Setup complete! You can now run:"
echo "1. Backend: cd backend && uvicorn main:app --reload"
echo "2. Frontend: cd frontend-next && npm install && npm run dev"
echo "3. Celery: cd backend && celery -A celery_app worker --loglevel=info"