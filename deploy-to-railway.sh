#!/bin/bash

# Fantasy Football AI - Railway Deployment Script
# This script deploys the frontend and backend as separate Railway services

echo "🚀 Starting Railway deployment for Fantasy Football AI"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Railway CLI is installed
if ! command_exists railway; then
    echo -e "${RED}❌ Railway CLI not found. Please install it first:${NC}"
    echo "brew install railway || curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

# Deploy Frontend
echo -e "\n${BLUE}📦 Deploying Frontend Service${NC}"
echo "================================"
cd frontend-next

# Initialize and deploy frontend
echo -e "${YELLOW}→ Initializing frontend service...${NC}"
railway login
railway link || railway init --name fantasy-football-frontend

# Set frontend environment variables
echo -e "${YELLOW}→ Setting frontend environment variables...${NC}"
railway variables set NEXT_PUBLIC_API_URL=https://fantasy-football-ai-production-4441.up.railway.app
railway variables set NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY:-"your-clerk-key"}

# Deploy frontend
echo -e "${YELLOW}→ Deploying frontend...${NC}"
railway up

# Get frontend URL
FRONTEND_URL=$(railway open --json | jq -r '.url' 2>/dev/null || echo "Check Railway dashboard")
echo -e "${GREEN}✅ Frontend deployed at: ${FRONTEND_URL}${NC}"

# Deploy Backend
echo -e "\n${BLUE}📦 Deploying Backend Service${NC}"
echo "================================"
cd ../backend

# Initialize and deploy backend
echo -e "${YELLOW}→ Initializing backend service...${NC}"
railway link || railway init --name fantasy-football-backend

# Set backend environment variables
echo -e "${YELLOW}→ Setting backend environment variables...${NC}"
railway variables set DATABASE_URL=${DATABASE_URL:-"postgresql://user:pass@host:5432/db"}
railway variables set REDIS_URL=${REDIS_URL:-"redis://host:6379"}
railway variables set JWT_SECRET_KEY=${JWT_SECRET_KEY:-"your-secret-key"}
railway variables set STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY:-"your-stripe-key"}
railway variables set STRIPE_WEBHOOK_SECRET=${STRIPE_WEBHOOK_SECRET:-"your-webhook-secret"}
railway variables set FRONTEND_URL=https://fantasy-football-frontend.railway.app
railway variables set BACKEND_URL=https://fantasy-football-ai-production-4441.up.railway.app

# Deploy backend
echo -e "${YELLOW}→ Deploying backend...${NC}"
railway up

# Get backend URL
BACKEND_URL=$(railway open --json | jq -r '.url' 2>/dev/null || echo "Check Railway dashboard")
echo -e "${GREEN}✅ Backend deployed at: ${BACKEND_URL}${NC}"

# Summary
echo -e "\n${GREEN}🎉 Deployment Complete!${NC}"
echo "========================"
echo -e "Frontend URL: ${FRONTEND_URL}"
echo -e "Backend URL: ${BACKEND_URL}"
echo -e "\n${YELLOW}⚠️  Important Next Steps:${NC}"
echo "1. Update environment variables in Railway dashboard"
echo "2. Set up PostgreSQL and Redis databases"
echo "3. Configure Clerk authentication keys"
echo "4. Set up Stripe payment keys"
echo "5. Update CORS settings if needed"

echo -e "\n${BLUE}📊 Monitor your deployments:${NC}"
echo "railway logs --service fantasy-football-frontend"
echo "railway logs --service fantasy-football-backend"