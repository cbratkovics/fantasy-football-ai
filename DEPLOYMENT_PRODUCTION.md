# Production Deployment Guide

This guide covers deploying the Fantasy Football AI platform to production using Railway (backend) and Vercel (frontend).

## Prerequisites

1. **Accounts Required**
   - [Railway](https://railway.app) account
   - [Vercel](https://vercel.com) account
   - [Stripe](https://stripe.com) account
   - [Clerk](https://clerk.com) account
   - [Supabase](https://supabase.com) or PostgreSQL database

2. **CLI Tools**
   ```bash
   npm install -g @railway/cli vercel
   ```

3. **Environment Variables**
   Copy `.env.example` to `.env` and fill in all values

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Vercel         │────▶│  Railway        │────▶│  Supabase      │
│  (Frontend)     │     │  (Backend API)  │     │  (PostgreSQL)  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        
         │                       │                        
         ▼                       ▼                        
┌─────────────────┐     ┌─────────────────┐              
│                 │     │                 │              
│  Clerk          │     │  Redis          │              
│  (Auth)         │     │  (Upstash)      │              
│                 │     │                 │              
└─────────────────┘     └─────────────────┘              
```

## Step 1: Database Setup (Supabase)

1. Create a new Supabase project
2. Run the database migrations:
   ```bash
   cd backend
   alembic upgrade head
   ```
3. Save the database URL for Railway configuration

## Step 2: Train and Save ML Models

```bash
python scripts/train_and_save_models.py
```

This creates the model files in the `models/` directory.

## Step 3: Deploy Backend to Railway

1. **Login to Railway**
   ```bash
   railway login
   ```

2. **Create new project**
   ```bash
   railway new fantasy-football-ai-backend
   ```

3. **Configure environment variables in Railway dashboard**
   - `DATABASE_URL` - PostgreSQL connection string
   - `REDIS_URL` - Redis connection string (use Railway Redis)
   - `JWT_SECRET_KEY` - Generate a secure key
   - `STRIPE_SECRET_KEY` - From Stripe dashboard
   - `STRIPE_WEBHOOK_SECRET` - Will get after setting up webhook
   - `STRIPE_PRICE_ID` - Create in Stripe dashboard

4. **Deploy**
   ```bash
   railway up
   ```

5. **Note the deployment URL** (e.g., `https://fantasy-football-ai.railway.app`)

## Step 4: Configure Stripe Webhook

1. Go to Stripe Dashboard → Webhooks
2. Add endpoint: `https://your-backend-url.railway.app/api/payments/webhook`
3. Select events:
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_failed`
4. Copy the webhook secret and update Railway environment

## Step 5: Deploy Frontend to Vercel

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Configure environment variables**
   Create `.env.production` in `frontend-next/`:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
   NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...
   ```

3. **Deploy**
   ```bash
   cd frontend-next
   vercel --prod
   ```

4. **Configure environment in Vercel dashboard**
   Add all environment variables from `.env.production`

## Step 6: Configure Clerk Authentication

1. Create a Clerk application
2. Configure OAuth providers (Google, GitHub)
3. Set up webhook for user sync (optional)
4. Update environment variables in both frontend and backend

## Step 7: Post-Deployment Tasks

### Security Hardening
1. **Enable CORS** - Update backend to only allow your frontend domain
2. **Rate Limiting** - Configure Redis-based rate limiting
3. **SSL/TLS** - Ensure all custom domains use HTTPS
4. **API Keys** - Rotate all development keys

### Monitoring Setup
1. **Error Tracking** - Configure Sentry
   ```bash
   npm install @sentry/nextjs @sentry/python
   ```

2. **Analytics** - Add Google Analytics or Plausible
3. **Uptime Monitoring** - Use UptimeRobot or similar

### Performance Optimization
1. **CDN** - Vercel automatically provides CDN
2. **Image Optimization** - Use Next.js Image component
3. **Database Indexes** - Ensure proper indexes on PostgreSQL
4. **Redis Caching** - Cache predictions for 1 hour

## Deployment Scripts

Use the provided scripts for easier deployment:

```bash
# Deploy everything
./scripts/deploy-all.sh

# Deploy backend only
./scripts/deploy-backend.sh

# Deploy frontend only
./scripts/deploy-frontend.sh
```

## Troubleshooting

### Backend Issues
- Check Railway logs: `railway logs`
- Verify environment variables are set
- Ensure database migrations ran successfully
- Check Redis connection

### Frontend Issues
- Check Vercel function logs
- Verify API URL is correct
- Check browser console for errors
- Ensure Clerk is configured properly

### Payment Issues
- Verify Stripe webhook is configured
- Check webhook signature matches
- Ensure price ID is correct
- Test with Stripe test cards first

## Scaling Considerations

### When you reach 100+ users:
1. **Database** - Enable connection pooling
2. **Redis** - Upgrade to larger instance
3. **API** - Add more Railway instances

### When you reach 1000+ users:
1. **Database** - Consider read replicas
2. **CDN** - Add Cloudflare
3. **API** - Implement GraphQL for efficiency
4. **ML Models** - Use model serving service

## Cost Estimation

- **Railway Backend**: ~$20/month (with database)
- **Vercel Frontend**: Free tier usually sufficient
- **Supabase**: Free tier for < 500 users
- **Redis (Upstash)**: ~$10/month
- **Total**: ~$30-50/month for 1000 users

## Support

For deployment issues:
- Railway: https://docs.railway.app
- Vercel: https://vercel.com/docs
- Stripe: https://stripe.com/docs
- Clerk: https://clerk.com/docs

Remember to monitor your costs and usage regularly!