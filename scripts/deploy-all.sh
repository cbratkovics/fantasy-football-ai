#!/bin/bash

# Deploy Complete Fantasy Football AI Platform
# This script orchestrates the deployment of both backend and frontend

set -e

echo "🚀 Deploying Fantasy Football AI Platform..."
echo "========================================"

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check for required CLIs
MISSING_TOOLS=()
command -v railway &> /dev/null || MISSING_TOOLS+=("railway")
command -v vercel &> /dev/null || MISSING_TOOLS+=("vercel")
command -v docker &> /dev/null || MISSING_TOOLS+=("docker")

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo "❌ Missing required tools: ${MISSING_TOOLS[*]}"
    echo "Please install them first:"
    echo "  railway: npm install -g @railway/cli"
    echo "  vercel: npm install -g vercel"
    echo "  docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found."
    echo "Creating from .env.example..."
    cp .env.example .env
    echo "Please update .env with your actual values before continuing."
    exit 1
fi

# Option to deploy locally first
read -p "Would you like to test locally with Docker first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🐳 Starting local Docker environment..."
    docker-compose up -d
    echo "✅ Local environment started at:"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
    echo ""
    read -p "Press any key to continue with production deployment..."
fi

# Train ML models if not exists
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 2>/dev/null)" ]; then
    echo "🤖 Training ML models..."
    python scripts/train_and_save_models.py
fi

# Deploy Backend
echo ""
echo "1️⃣ Deploying Backend to Railway..."
echo "-----------------------------------"
./scripts/deploy-backend.sh

# Get backend URL
read -p "Enter the deployed backend URL (e.g., https://your-app.railway.app): " BACKEND_URL

# Update frontend environment
echo "NEXT_PUBLIC_API_URL=$BACKEND_URL" >> frontend-next/.env.production

# Deploy Frontend
echo ""
echo "2️⃣ Deploying Frontend to Vercel..."
echo "-----------------------------------"
./scripts/deploy-frontend.sh

# Post-deployment steps
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "📝 Post-deployment checklist:"
echo "[ ] Configure Stripe webhook endpoint in Stripe dashboard"
echo "[ ] Set up Clerk authentication in production"
echo "[ ] Configure custom domains for both frontend and backend"
echo "[ ] Set up monitoring (Sentry, LogRocket, etc.)"
echo "[ ] Configure SSL certificates if using custom domains"
echo "[ ] Test payment flow with Stripe test cards"
echo "[ ] Verify ML predictions are working correctly"
echo ""
echo "🔒 Security reminders:"
echo "- Rotate all secret keys for production"
echo "- Enable CORS only for your frontend domain"
echo "- Set up rate limiting and DDoS protection"
echo "- Regular security audits and dependency updates"