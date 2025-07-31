# Railway Health Check Fix

## Issue
Railway deployment fails health checks with "service unavailable" errors.

## Root Causes Identified
1. **Import path issues**: Using `backend.` prefix in imports when working directory is `/app`
2. **Missing error handling**: App crashes on startup without proper error messages
3. **Environment variables**: May be missing critical configuration

## Fixes Applied

### 1. Fixed Import Paths in main.py
Changed all imports from:
```python
from backend.api import auth, players
```
To:
```python
from api import auth, players
```

### 2. Enhanced Dockerfile
- Added `PYTHONPATH=/app` environment variable
- Improved CMD with proper logging
- Added executable permissions for start script

### 3. Created Start Scripts
- `start_app.py`: Main application starter with logging
- `railway_debug.py`: Debug script to identify issues
- `health_check.py`: Minimal health check server

### 4. Added Comprehensive Logging
- Startup logs show environment variables
- Import errors are caught and logged
- Database connection errors won't crash the app

## Debugging Steps

### Option 1: Use Debug Script
Update Dockerfile temporarily:
```dockerfile
CMD ["python", "railway_debug.py"]
```

This will:
- Check all environment variables
- Test critical imports
- Fall back to minimal server if main app fails

### Option 2: Check Railway Logs
Look for:
- Missing environment variables
- Import errors
- Database connection failures

### Option 3: Use Minimal Health Check
Update Dockerfile:
```dockerfile
CMD ["python", "health_check.py"]
```

If this works, the issue is with the main app initialization.

## Required Environment Variables

Ensure these are set in Railway:

```
PORT=<auto-provided by Railway>
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
SECRET_KEY=<your-secret-key>
ENVIRONMENT=production
```

## Quick Fix Command

To test locally:
```bash
cd backend
export PORT=8000
export DATABASE_URL=postgresql://localhost/test
python start_app.py
```

## Next Steps

1. Deploy with the current fixes
2. Check Railway logs for specific errors
3. If still failing, use railway_debug.py to identify the issue
4. Ensure all environment variables are properly set