# Deployment Guide for Fantasy Football AI Backend

## Overview
This guide covers deploying the Fantasy Football AI backend to Railway and other cloud platforms.

## Prerequisites
- Docker installed locally (for testing)
- Railway CLI (optional): `npm install -g @railway/cli`
- Python 3.10
- Git

## Local Docker Testing

### 1. Build the Docker image
```bash
cd backend
docker build -t fantasy-football-ai-backend .
```

### 2. Run the container locally
```bash
# Create a .env file with your environment variables
docker run -p 8000:8000 --env-file .env fantasy-football-ai-backend
```

### 3. Test the health endpoint
```bash
curl http://localhost:8000/health
```

## Railway Deployment

### 1. Initial Setup
1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Select the `backend` directory as the root directory

### 2. Environment Variables
Set the following environment variables in Railway:

#### Required Variables
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - JWT secret key (generate with `openssl rand -hex 32`)
- `ENVIRONMENT` - Set to "production"

#### Optional API Keys
- `OPENAI_API_KEY` - For LLM features
- `ANTHROPIC_API_KEY` - For Claude integration
- `STRIPE_API_KEY` - For payment processing
- `STRIPE_WEBHOOK_SECRET` - For Stripe webhooks
- `ESPN_S2` - ESPN authentication
- `ESPN_SWID` - ESPN session ID

### 3. Deploy
Railway will automatically deploy when you push to your main branch.

```bash
# Manual deployment via CLI
railway up
```

## Troubleshooting

### Common Issues

#### 1. Redis Dependency Conflicts
**Problem**: `redis-py-cluster` conflicts with newer Redis versions

**Solution**: Removed `redis-py-cluster==2.1.3` from requirements.txt as it's not used in the codebase. The project uses standard Redis connections, not clustering.

#### 2. Dependency Installation Failures
**Problem**: `pip install` fails with compilation errors

**Solution**: The Dockerfile includes all necessary system dependencies for MySQL and PostgreSQL clients. If you still encounter issues:
- Ensure you're using the correct Python version (3.10)
- Check that all system dependencies are installed in the Dockerfile

#### 2. Port Binding Issues
**Problem**: Application doesn't respond on Railway

**Solution**: Railway provides the PORT environment variable. The application is configured to use it:
```python
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

#### 3. Database Connection Errors
**Problem**: Can't connect to PostgreSQL

**Solution**: 
- Verify DATABASE_URL format: `postgresql://user:password@host:port/database`
- Ensure the database allows connections from Railway's IP ranges
- Check if SSL is required (add `?sslmode=require` to the URL)

#### 4. Memory Issues
**Problem**: Application crashes with memory errors

**Solution**:
- The Dockerfile uses `tensorflow-cpu` instead of full TensorFlow
- Monitor memory usage in Railway dashboard
- Consider upgrading to a larger instance if needed

### Dependency Conflicts
If you encounter dependency conflicts:

1. Create a fresh virtual environment locally:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies one by one to identify conflicts:
```bash
pip install -r requirements.txt
```

3. Use `pipdeptree` to visualize dependencies:
```bash
pip install pipdeptree
pipdeptree --warn fail
```

## Performance Optimization

### 1. Docker Image Size
- Use `.dockerignore` to exclude unnecessary files
- Multi-stage builds can reduce final image size
- Consider using Alpine Linux for smaller base image (requires additional setup for Python packages)

### 2. Build Time
- Order Dockerfile commands to maximize layer caching
- Copy requirements.txt before application code
- Install system dependencies in a single RUN command

### 3. Runtime Performance
- Use Gunicorn with Uvicorn workers for production:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

## Monitoring

### Health Checks
The application exposes a `/health` endpoint that returns:
```json
{
  "status": "healthy",
  "service": "fantasy-football-ai-backend",
  "version": "1.0.0",
  "timestamp": "2024-01-31T12:00:00Z"
}
```

Railway automatically monitors this endpoint based on the `railway.json` configuration.

### Logs
View logs via Railway dashboard or CLI:
```bash
railway logs
```

## Security Considerations

1. **Environment Variables**: Never commit `.env` files to version control
2. **Secrets**: Use Railway's secret management for sensitive data
3. **Database**: Use SSL connections for production databases
4. **API Keys**: Rotate API keys regularly
5. **Dependencies**: Keep dependencies updated for security patches

## Rollback Strategy

If a deployment fails:

1. **Via Railway Dashboard**: Use the "Rollback" button to revert to a previous deployment
2. **Via Git**: Revert the problematic commit and push:
```bash
git revert HEAD
git push origin main
```

## Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For deployment issues:
1. Check Railway status page
2. Review application logs
3. Verify environment variables
4. Test locally with Docker first