# Railway Deployment Guide - Optimized for 4GB Limit

## 🚨 Problem Summary
Your Docker image exceeds Railway's 4GB limit due to:
1. **TensorFlow** alone is ~1.5GB
2. **All ML libraries** total ~2.2GB
3. **Redundant frontend directories** (frontend + frontend-next)
4. **No Docker optimization** (no .dockerignore, single-stage build)

## ✅ Solution: Separate Deployments

### Step 1: Deploy Frontend Separately

1. **Create a new Railway service for frontend-next**:
   ```bash
   cd frontend-next
   railway init
   railway up
   ```

2. **Use Railway's Node.js buildpack** (automatic detection)

3. **Set environment variables**:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your-key
   ```

### Step 2: Optimize Backend Deployment

1. **Use the optimized files created**:
   - `requirements-prod.txt` - Uses tensorflow-cpu, removes training libraries
   - `Dockerfile.prod` - Multi-stage build, only copies needed files
   - `.dockerignore` - Excludes unnecessary files

2. **Update your Railway backend**:
   ```bash
   cd backend
   
   # Use production Dockerfile
   mv Dockerfile Dockerfile.dev
   mv Dockerfile.prod Dockerfile
   
   # Use production requirements
   mv requirements.txt requirements-dev.txt
   mv requirements-prod.txt requirements.txt
   
   # Deploy
   railway up
   ```

3. **Mount models from external storage** (recommended):
   - Upload models to Railway volumes or S3
   - Mount at runtime instead of bundling

### Step 3: Clean Up Project

1. **Remove redundant directories**:
   ```bash
   # Remove old Streamlit frontend
   rm -rf frontend
   
   # Remove training scripts from production
   mkdir scripts-training
   mv scripts/train_*.py scripts-training/
   mv scripts/test_*.py scripts-training/
   ```

2. **Separate ML artifacts**:
   ```bash
   # Move large models to cloud storage
   # Keep only small pickle files locally
   ```

## 📊 Expected Results

### Before Optimization:
- Backend image: ~3.5GB (with frontend)
- Contains all ML libraries
- Single-stage build

### After Optimization:
- Frontend: ~500MB (separate service)
- Backend: ~1.5-2GB (tensorflow-cpu + essentials)
- Multi-stage build
- Only production dependencies

## 🚀 Alternative: Serverless ML Inference

If still too large, consider:

1. **Separate ML inference service**:
   - Deploy models on Hugging Face Spaces
   - Use Railway for API only
   - Call ML service via HTTP

2. **Use TensorFlow Lite**:
   - Convert models to TFLite format
   - Much smaller runtime (~50MB)
   - Slightly less accurate but much faster

3. **Pre-compute predictions**:
   - Run batch predictions nightly
   - Store in database
   - Serve from cache

## 📝 Quick Commands

```bash
# Deploy frontend
cd frontend-next
railway init --name fantasy-football-frontend
railway up

# Deploy optimized backend
cd ../backend
railway init --name fantasy-football-backend
railway up

# Set up environment variables
railway variables set NEXT_PUBLIC_API_URL=https://fantasy-football-backend.railway.app
```

## 🔍 Monitoring Image Size

```bash
# Check Docker image size locally
docker build -t fantasy-backend .
docker images | grep fantasy-backend

# Analyze layer sizes
docker history fantasy-backend
```

## 💡 Key Optimizations Made

1. **tensorflow → tensorflow-cpu** (saves ~1GB)
2. **Removed visualization libraries** (matplotlib, seaborn, plotly)
3. **Removed training libraries** (xgboost, lightgbm, optuna)
4. **Multi-stage Docker build**
5. **Comprehensive .dockerignore**
6. **Separate frontend deployment**

This should get your deployment well under Railway's 4GB limit!