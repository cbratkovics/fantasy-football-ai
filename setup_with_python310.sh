#!/bin/bash

echo "Setting up Fantasy Football AI with Python 3.10..."

# Navigate to project root
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai

# Remove old venv if exists
rm -rf venv

# Create virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install numpy first (specific version for compatibility)
pip install numpy==1.26.4

# Then install the rest of the requirements
pip install -r backend/requirements.txt

# Set PYTHONPATH
export PYTHONPATH=/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend:$PYTHONPATH

echo "Setup complete! Virtual environment created with Python 3.10"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"