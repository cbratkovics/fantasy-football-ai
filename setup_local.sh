#!/bin/bash

echo "Setting up Fantasy Football AI locally..."

# Navigate to project root
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai

# Create and activate virtual environment in project directory
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install backend requirements
echo "Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create local .env file with proper credentials
echo "Creating local .env configuration..."
cat > backend/.env.local << 'EOF'
# Database
DATABASE_URL=postgresql://fantasy_user:fantasy_pass@localhost:5432/fantasy_football

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET_KEY=your-local-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Stripe (test keys)
STRIPE_SECRET_KEY=sk_test_local
STRIPE_WEBHOOK_SECRET=whsec_local

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000

# Backend URL
BACKEND_URL=http://localhost:8000

# Environment
ENVIRONMENT=development

# API Keys (if needed)
ESPN_API_KEY=
NFL_API_KEY=
WEATHER_API_KEY=
EOF

# Copy local env to main env file
cp backend/.env.local backend/.env

echo "Setup complete! Next steps:"
echo "1. Create PostgreSQL database and user:"
echo "   sudo -u postgres psql"
echo "   CREATE USER fantasy_user WITH PASSWORD 'fantasy_pass';"
echo "   CREATE DATABASE fantasy_football OWNER fantasy_user;"
echo "   GRANT ALL PRIVILEGES ON DATABASE fantasy_football TO fantasy_user;"
echo "   \q"
echo ""
echo "2. Make sure PostgreSQL and Redis are running:"
echo "   brew services start postgresql"
echo "   brew services start redis"
echo ""
echo "3. Run database migrations:"
echo "   source venv/bin/activate"
echo "   cd backend"
echo "   alembic upgrade head"
echo ""
echo "4. Initialize database:"
echo "   cd .."
echo "   python scripts/init_database.py"