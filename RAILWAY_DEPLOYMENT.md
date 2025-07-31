# Railway Deployment Instructions

## 🚀 Quick Start

1. **Run the deployment script:**
   ```bash
   ./deploy-to-railway.sh
   ```

2. **Or deploy manually:**

### Frontend Deployment
```bash
cd frontend-next
railway login
railway init --name fantasy-football-frontend
railway up

# Set environment variables
railway variables set NEXT_PUBLIC_API_URL=https://fantasy-football-backend.railway.app
railway variables set NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your-clerk-key
```

### Backend Deployment
```bash
cd backend
railway login
railway init --name fantasy-football-backend
railway up

# Set environment variables
railway variables set DATABASE_URL=your-postgres-url
railway variables set REDIS_URL=your-redis-url
railway variables set JWT_SECRET_KEY=your-secret-key
railway variables set STRIPE_SECRET_KEY=your-stripe-key
```

## 📊 Image Size Optimization Results

### Before:
- Single monolithic image: ~4GB+
- Full TensorFlow: 1.5GB
- All ML libraries: 2.2GB
- Both frontends included

### After:
- Backend: ~1.5-2GB (with tensorflow-cpu)
- Frontend: ~500MB (separate service)
- Total reduction: ~50%

## 🔧 Key Optimizations Applied

1. **Separated frontend and backend** into independent services
2. **Replaced tensorflow with tensorflow-cpu** (saves ~1GB)
3. **Removed training-only dependencies** (xgboost, lightgbm, optuna)
4. **Removed visualization libraries** (matplotlib, seaborn, plotly)
5. **Multi-stage Docker build** for backend
6. **Comprehensive .dockerignore** files

## 🌐 Environment Variables

### Backend Required:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET_KEY` - Secret for JWT tokens
- `STRIPE_SECRET_KEY` - Stripe API key
- `FRONTEND_URL` - Frontend URL for CORS

### Frontend Required:
- `NEXT_PUBLIC_API_URL` - Backend API URL
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` - Clerk public key
- `CLERK_SECRET_KEY` - Clerk secret key

## 🗄️ Database Setup

1. **Create PostgreSQL database:**
   - Use Railway's PostgreSQL plugin
   - Or use external service (Supabase, Neon, etc.)

2. **Run migrations:**
   ```bash
   railway run alembic upgrade head
   ```

3. **Create Redis instance:**
   - Use Railway's Redis plugin
   - Or use external service (Upstash, etc.)

## 🔍 Monitoring

```bash
# View logs
railway logs --service fantasy-football-frontend
railway logs --service fantasy-football-backend

# Check deployment status
railway status

# Open services
railway open --service fantasy-football-frontend
railway open --service fantasy-football-backend
```

## 🚨 Troubleshooting

### If deployment fails due to size:
1. Check Docker image size locally:
   ```bash
   docker build -t test-backend backend/
   docker images | grep test-backend
   ```

2. Further reduce image size:
   - Consider using Alpine Linux base
   - Remove more dependencies
   - Use model serving service (Hugging Face, etc.)

### If ML models are missing:
1. Upload models to Railway volumes
2. Or use external storage (S3, GCS)
3. Mount at runtime instead of bundling

## 📝 Next Steps

1. **Set up monitoring** (Railway metrics, Sentry)
2. **Configure auto-scaling** if needed
3. **Set up CI/CD** with GitHub Actions
4. **Configure custom domain**
5. **Enable SSL certificates** (automatic with Railway)

## 🎯 Performance Tips

1. **Use Railway's edge network** for global distribution
2. **Enable caching** for static assets
3. **Configure health checks** properly
4. **Set appropriate replica counts**
5. **Monitor resource usage** and scale accordingly