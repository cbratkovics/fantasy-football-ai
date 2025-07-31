# Deployment Guide

## Railway Deployment

### Prerequisites
- Railway account
- GitHub repository connected
- Environment variables configured

### Deployment Steps
1. Push code to GitHub main branch
2. Railway automatically builds using Dockerfile
3. Monitor deployment at railway.app dashboard

### Configuration Files
- `Dockerfile` - Production Docker configuration
- `railway.json` - Railway-specific settings
- `.python-version` - Python 3.10 specified

### Environment Variables
Set in Railway dashboard:
```
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
SECRET_KEY=<generate with openssl rand -hex 32>
ENVIRONMENT=production
```

### Troubleshooting

#### Dependency Conflicts
- **Redis conflict**: Already resolved by removing unused redis-py-cluster
- **Anthropic version**: Fixed to 0.17.0 for langchain compatibility

#### Build Failures
1. Check Railway build logs
2. Test Docker build locally first
3. Verify all system dependencies in Dockerfile

#### Connection Issues
- Ensure DATABASE_URL includes `?sslmode=require` for secure connections
- Verify Redis is accessible from Railway
- Check health endpoint: `/health`

## Local Docker Testing
```bash
# Build
docker build -t fantasy-football-ai .

# Run
docker run -p 8000:8000 --env-file .env fantasy-football-ai

# Test
curl http://localhost:8000/health
```

## Production Optimization
- Uses tensorflow-cpu to reduce image size
- Multi-stage builds considered for future
- Redis connection pooling recommended
- Monitor memory usage in Railway dashboard

## Rollback Strategy
1. Use Railway dashboard rollback button
2. Or revert Git commit and push