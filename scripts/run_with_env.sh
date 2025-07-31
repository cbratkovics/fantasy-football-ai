#!/bin/bash
# Script to run Python scripts with correct environment variables

# Export the correct DATABASE_URL
export DATABASE_URL="postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"

# Also set REDIS_URL if needed
export REDIS_URL="redis://localhost:6379"

echo "Environment variables set:"
echo "DATABASE_URL: ${DATABASE_URL##*@}"  # Show only the host part for security

# Run the command passed as arguments
"$@"