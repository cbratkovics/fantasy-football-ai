#!/bin/bash

# Navigate to backend directory
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai/backend

# Export the database URL explicitly
export DATABASE_URL="postgresql://postgres:-Pv95h_SjeXf!Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"
export REDIS_URL="redis://localhost:6379"

# Start uvicorn
/Users/christopherbratkovics/anaconda3/envs/agentic_ai_env/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000