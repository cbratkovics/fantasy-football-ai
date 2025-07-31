#!/bin/bash

echo "Starting Fantasy Football AI with Docker..."

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

# Stop any existing containers
docker-compose down

# Build and start services
echo "Building and starting services..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Run migrations using docker-compose
echo "Running database migrations..."
docker-compose run --rm backend alembic upgrade head

# Initialize database
echo "Initializing database..."
docker-compose run --rm backend python scripts/init_database.py

# Start all services
echo "Starting all services..."
docker-compose up -d

echo ""
echo "Services are starting up!"
echo ""
echo "Access the application at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "docker-compose logs -f"
echo ""
echo "To stop all services:"
echo "docker-compose down"