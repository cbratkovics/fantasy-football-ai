# Vercel Environment Variables Setup

## Required Environment Variables

To properly deploy the fantasy football frontend on Vercel, you need to set the following environment variables in your Vercel dashboard:

### 1. Backend API Configuration
```
NEXT_PUBLIC_API_URL=https://fantasy-football-backend.railway.app
```
**Important**: This should point to your deployed Railway backend URL, not localhost.

### 2. Authentication (If using Clerk)
```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
CLERK_SECRET_KEY=sk_live_...
```

### 3. Stripe (If using payments)
```
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...
```

## Setting Environment Variables in Vercel

1. Go to your Vercel dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Add each variable with:
   - **Name**: The variable name (e.g., `NEXT_PUBLIC_API_URL`)
   - **Value**: The actual value (e.g., `https://fantasy-football-backend.railway.app`)
   - **Environment**: Select "Production", "Preview", and "Development" as needed

## Current Status

The tiers visualization now includes fallback mock data, so the frontend will work even if the API is temporarily unavailable. However, for full functionality, the backend API URL must be correctly configured.

## Testing the API Connection

After setting the environment variables, you can test the API connection by:

1. Checking the browser's Network tab for API calls
2. Looking for successful responses from the `/tiers/positions/{position}` endpoint
3. Verifying that real data loads instead of the "Using demo data" message

## Backend API Endpoints Used

The frontend expects these API endpoints to be available:

- `GET /tiers/positions/{position}?scoring_type={type}` - Get tiers for a specific position
- `GET /tiers/all?scoring_type={type}` - Get tiers for all positions
- `GET /predictions/week/{week}` - Get weekly predictions
- `GET /players/rankings` - Get player rankings

Make sure your Railway backend is deployed and these endpoints are working.