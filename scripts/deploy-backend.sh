#!/bin/bash

# Deploy Backend to Railway
# This script deploys the FastAPI backend to Railway

set -e

echo "🚀 Deploying Fantasy Football AI Backend to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway. Please run: railway login"
    exit 1
fi

# Environment check
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Make sure to set environment variables in Railway dashboard."
fi

# Create/Update Railway project
echo "📦 Setting up Railway project..."
railway up

# Set environment variables (if .env exists)
if [ -f .env ]; then
    echo "🔧 Setting environment variables..."
    # Parse .env file and set variables
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^# ]] && [[ -n "$key" ]]; then
            # Remove quotes from value
            value="${value%\"}"
            value="${value#\"}"
            railway variables set "$key=$value"
        fi
    done < .env
fi

# Deploy
echo "🚢 Deploying to Railway..."
railway deploy

# Get deployment URL
DEPLOYMENT_URL=$(railway status --json | jq -r '.url')

echo "✅ Backend deployed successfully!"
echo "🌐 URL: $DEPLOYMENT_URL"
echo ""
echo "📝 Next steps:"
echo "1. Update NEXT_PUBLIC_API_URL in your frontend .env with: $DEPLOYMENT_URL"
echo "2. Set up your database connection in Railway dashboard"
echo "3. Configure Stripe webhook endpoint: ${DEPLOYMENT_URL}/api/payments/webhook"