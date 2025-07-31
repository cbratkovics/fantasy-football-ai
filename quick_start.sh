#!/bin/bash

echo "Quick start for Fantasy Football AI..."

# Navigate to project root
cd /Users/christopherbratkovics/Desktop/fantasy-football-ai

# Create .env file in root for docker-compose
cat > .env << EOF
# Stripe keys (optional for local development)
STRIPE_SECRET_KEY=sk_test_local
STRIPE_WEBHOOK_SECRET=whsec_local
STRIPE_PRICE_ID=price_test_local

# Clerk keys (optional for local development)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_local
CLERK_SECRET_KEY=sk_test_local

# Stripe public key (optional for local development)
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_local
EOF

# Start only the databases first
echo "Starting PostgreSQL and Redis..."
docker-compose up -d postgres redis

# Wait for them to be ready
echo "Waiting for databases to be ready..."
sleep 15

# Now you can run the backend locally with conda
echo ""
echo "Databases are ready!"
echo ""
echo "To run the backend:"
echo "1. Activate conda environment: source ~/anaconda3/bin/activate agentic_ai_env"
echo "2. Set PYTHONPATH: export PYTHONPATH=/Users/christopherbratkovics/Desktop/fantasy-football-ai:/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend"
echo "3. Run migrations: cd backend && alembic upgrade head"
echo "4. Start backend: uvicorn main:app --reload"
echo ""
echo "To run the frontend:"
echo "1. cd frontend-next"
echo "2. npm install"
echo "3. npm run dev"
echo ""
echo "Database connection info:"
echo "- PostgreSQL: postgresql://fantasy_user:fantasy_pass@localhost:5432/fantasy_football"
echo "- Redis: redis://localhost:6379"