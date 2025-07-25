#!/bin/bash

echo "🏈 Fantasy Football AI Setup Script"
echo "==================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected"
    echo "Consider running: conda activate agentic_ai_env"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Check for API key
echo ""
echo "🔑 Checking NFL API Key..."
if [ -z "$NFL_API_KEY" ]; then
    echo "⚠️  NFL_API_KEY environment variable not set"
    echo "Please get your API key from: https://rapidapi.com/api-sports/api/american-football/"
    echo "Then set it with: export NFL_API_KEY='your_key_here'"
    echo "Or add it to your ~/.bashrc or ~/.zshrc file"
else
    echo "✅ NFL_API_KEY is set"
fi

# Initialize database
echo ""
echo "🗃️  Initializing database..."
python -m fantasy_ai.cli.main database init

if [ $? -eq 0 ]; then
    echo "✅ Database initialized successfully"
else
    echo "❌ Database initialization failed"
    echo "Try running manually: python -m fantasy_ai.cli.main database init"
fi

# Test API connection (if key is set)
if [ ! -z "$NFL_API_KEY" ]; then
    echo ""
    echo "🔗 Testing API connection..."
    python -m fantasy_ai.cli.main api test
fi

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Set your NFL API key: export NFL_API_KEY='your_key_here'"
echo "2. Test the system: python -m fantasy_ai.cli.main collect quick --max-calls 10"
echo "3. View help: python -m fantasy_ai.cli.main --help"
echo ""