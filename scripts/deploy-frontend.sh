#!/bin/bash

# Deploy Frontend to Vercel
# This script deploys the Next.js frontend to Vercel

set -e

echo "🚀 Deploying Fantasy Football AI Frontend to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Please install it first:"
    echo "npm install -g vercel"
    exit 1
fi

# Navigate to frontend directory
cd frontend-next

# Check if logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "❌ Not logged in to Vercel. Please run: vercel login"
    exit 1
fi

# Build the project
echo "🔨 Building Next.js application..."
npm run build

# Deploy to Vercel
echo "🚢 Deploying to Vercel..."
vercel --prod

# Get deployment URL
DEPLOYMENT_URL=$(vercel ls --json | jq -r '.[0].url')

echo "✅ Frontend deployed successfully!"
echo "🌐 URL: https://$DEPLOYMENT_URL"
echo ""
echo "📝 Next steps:"
echo "1. Configure custom domain in Vercel dashboard"
echo "2. Set up environment variables in Vercel:"
echo "   - NEXT_PUBLIC_API_URL"
echo "   - NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY"
echo "   - CLERK_SECRET_KEY"
echo "   - NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY"
echo "3. Test the live application"