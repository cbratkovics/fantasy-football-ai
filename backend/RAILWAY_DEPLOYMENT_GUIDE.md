# Railway Deployment Fix Guide

## Overview
This guide documents the fixes applied to resolve Railway health check failures for the Fantasy Football AI backend.

## Issues Identified

### 1. **Database Connection Failures**
- **Problem**: App tried to create database tables on startup without proper error handling
- **Impact**: If Supabase connection fails, entire app crashes during startup
- **Fix**: Added DATABASE_AVAILABLE flag and graceful degradation

### 2. **Import Dependencies**
- **Problem**: Optional LLM services could cause import failures
- **Impact**: App crashes if optional dependencies are missing
- **Fix**: Wrapped imports in try-catch blocks with fallback behavior

### 3. **Hardcoded Database URL**
- **Problem**: `models/database.py` had hardcoded Supabase URL as fallback
- **Impact**: Could connect to wrong database or fail silently
- **Fix**: Removed hardcoded URL, require DATABASE_URL environment variable

### 4. **Complex Startup Dependencies**
- **Problem**: Main app has too many dependencies for health check testing
- **Impact**: Hard to isolate health check issues
- **Fix**: Created `main_simple.py` with minimal dependencies

## Files Modified

### 1. **backend/main_simple.py** (NEW)
- Minimal FastAPI app with only health check endpoints
- No database dependencies
- Use for initial Railway deployment testing

### 2. **backend/start_simple.py** (NEW)
- Startup script for minimal app
- Enhanced logging for debugging
- Environment variable inspection

### 3. **backend/Dockerfile**
- Updated CMD to use `start_simple.py` for debugging
- Added executable permissions for both startup scripts

### 4. **backend/main.py**
- Added error handling for database imports
- Graceful degradation when database unavailable
- Conditional router registration

### 5. **backend/models/database.py**
- Removed hardcoded DATABASE_URL fallback
- Added null checks for engine and SessionLocal
- Better error handling in get_db()

## Deployment Steps

### Phase 1: Test Minimal App
1. **Deploy with Minimal App**
   ```bash
   # Current Dockerfile uses start_simple.py
   git add .
   git commit -m "fix: add minimal app for Railway health check testing"
   git push origin main
   ```

2. **Verify Health Check**
   - Check Railway logs for startup success
   - Test endpoints: `/`, `/health`, `/ready`
   - Confirm health check passes

### Phase 2: Environment Variables
1. **Set Required Variables in Railway Dashboard**
   ```
   DATABASE_URL=postgresql://your-supabase-url
   ENVIRONMENT=production
   PORT=(auto-provided by Railway)
   ```

2. **Optional Variables** (if using LLM features)
   ```
   OPENAI_API_KEY=your-key
   ANTHROPIC_API_KEY=your-key
   REDIS_URL=your-redis-url
   ```

### Phase 3: Switch to Full App
1. **Update Dockerfile**
   ```dockerfile
   # Change CMD back to full app
   CMD ["sh", "-c", "python start_app.py"]
   ```

2. **Deploy Full App**
   ```bash
   git add .
   git commit -m "feat: enable full app after health check fix"
   git push origin main
   ```

## Testing Commands

### Local Testing
```bash
# Test minimal app
cd backend
python start_simple.py

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/

# Test with Docker (simulate Railway)
docker build -t fantasy-test .
docker run -p 8000:8000 -e PORT=8000 fantasy-test
```

### Railway Logs
```bash
# View deployment logs
railway logs

# Check specific service logs
railway logs --service backend
```

## Health Check Verification

### Expected Response from `/health`
```json
{
  "status": "healthy",
  "service": "fantasy-football-ai-backend",
  "version": "1.0.0",
  "timestamp": "2024-07-31T20:00:00.000Z",
  "port": "8000",
  "environment": "production"
}
```

### Debugging Checklist
- [ ] Health check endpoint returns 200 status
- [ ] App binds to `0.0.0.0` not `127.0.0.1`
- [ ] PORT environment variable is used correctly
- [ ] No startup crashes in Railway logs
- [ ] Database connection is optional for health check

## Rollback Plan

If full app still fails:

1. **Revert to Minimal App**
   ```dockerfile
   CMD ["sh", "-c", "python start_simple.py"]
   ```

2. **Disable Health Check Temporarily**
   ```json
   // railway.json
   {
     "deploy": {
       "healthcheckPath": "",
       "restartPolicyType": "ON_FAILURE"
     }
   }
   ```

## Next Steps

1. **Monitor Railway Deployment**: Check if minimal app passes health checks
2. **Add Environment Variables**: Configure DATABASE_URL and other required vars
3. **Gradually Enable Features**: Add database, then LLM services
4. **Update Domain Configuration**: Ensure winmyleague.ai points to Railway
5. **Update External Services**: Point Clerk/Supabase webhooks to new domain

## Additional Notes

- Railway automatically provides `PORT` environment variable
- Health check timeout is 5 minutes by default
- App must bind to `0.0.0.0:$PORT` for Railway to connect
- Database URL format: `postgresql://user:pass@host:port/db`
- Consider using Railway's PostgreSQL addon instead of external Supabase