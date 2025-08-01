#!/usr/bin/env python3
"""
Railway-optimized startup script for Fantasy Football AI Backend
Handles environment diagnostics and graceful fallbacks
"""

import os
import sys
import logging
import time
from datetime import datetime

# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check critical environment variables and system status"""
    logger.info("=" * 80)
    logger.info("RAILWAY STARTUP DIAGNOSTICS")
    logger.info("=" * 80)
    
    # Basic system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    
    # Critical environment variables
    port = os.getenv('PORT')
    logger.info(f"PORT environment variable: {port}")
    if not port:
        logger.error("CRITICAL: PORT environment variable not set!")
        return False
    
    # Check if port is valid
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            logger.error(f"CRITICAL: Invalid port number: {port_int}")
            return False
        logger.info(f"Port validation: OK ({port_int})")
    except ValueError:
        logger.error(f"CRITICAL: PORT is not a valid integer: {port}")
        return False
    
    # Optional environment variables
    env_vars = ['DATABASE_URL', 'REDIS_URL', 'ENVIRONMENT', 'PYTHONPATH']
    for var in env_vars:
        value = os.getenv(var)
        status = "SET" if value else "NOT SET"
        logger.info(f"{var}: {status}")
    
    # Check file system
    critical_files = ['main.py', 'requirements.txt']
    for file in critical_files:
        if os.path.exists(file):
            logger.info(f"File check: {file} - EXISTS")
        else:
            logger.error(f"File check: {file} - MISSING")
            return False
    
    return True

def test_imports():
    """Test critical imports without full startup"""
    logger.info("Testing critical imports...")
    
    try:
        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        logger.error(f"FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        logger.info(f"Uvicorn import: OK")
    except ImportError as e:
        logger.error(f"Uvicorn import failed: {e}")
        return False
    
    try:
        # Test main app import
        from main import app
        logger.info("Main app import: OK")
        return True
    except Exception as e:
        logger.error(f"Main app import failed: {e}")
        logger.exception("Full traceback:")
        return False

def start_minimal_server():
    """Start minimal FastAPI server as fallback"""
    logger.info("Starting minimal fallback server...")
    
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Fantasy Football AI - Minimal Fallback")
    
    @app.get("/")
    async def root():
        return JSONResponse({
            "status": "running",
            "message": "Fantasy Football AI - Minimal Mode",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    @app.get("/health")  
    async def health():
        return JSONResponse({
            "status": "healthy",
            "mode": "minimal",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting minimal server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)

def start_main_server():
    """Start the main FastAPI application"""
    logger.info("Starting main FastAPI application...")
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting main server on 0.0.0.0:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        # Add timeout and worker settings for Railway
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )

def main():
    """Main startup function with progressive fallbacks"""
    try:
        # Step 1: Environment diagnostics
        if not check_environment():
            logger.error("Environment check failed! Exiting...")
            sys.exit(1)
        
        # Step 2: Import testing
        if not test_imports():
            logger.warning("Main app imports failed, starting minimal server...")
            start_minimal_server()
            return
        
        # Step 3: Start main application
        logger.info("All checks passed, starting main application...")
        start_main_server()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.exception("Full traceback:")
        
        # Final fallback
        logger.info("Attempting minimal server as last resort...")
        try:
            start_minimal_server()
        except Exception as fallback_error:
            logger.error(f"Minimal server also failed: {fallback_error}")
            sys.exit(1)

if __name__ == "__main__":
    main()