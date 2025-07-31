#!/usr/bin/env python3
"""
Debug script for Railway deployment issues
"""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check and log environment variables"""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 60)
    
    critical_vars = [
        "PORT",
        "DATABASE_URL",
        "REDIS_URL",
        "SECRET_KEY",
        "ENVIRONMENT"
    ]
    
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            if var in ["DATABASE_URL", "SECRET_KEY"]:
                logger.info(f"{var}: [SET - HIDDEN]")
            else:
                logger.info(f"{var}: {value}")
        else:
            logger.warning(f"{var}: NOT SET")
    
    logger.info("=" * 60)

def test_imports():
    """Test critical imports"""
    logger.info("Testing imports...")
    
    try:
        import fastapi
        logger.info("✓ FastAPI imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        logger.info("✓ Uvicorn imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Uvicorn: {e}")
        return False
    
    try:
        import sqlalchemy
        logger.info("✓ SQLAlchemy imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import SQLAlchemy: {e}")
        return False
    
    try:
        from models.database import engine
        logger.info("✓ Database module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import database module: {e}")
        logger.info("This might be due to missing DATABASE_URL")
    
    return True

def start_minimal_server():
    """Start a minimal server for health checks"""
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"status": "running", "message": "Railway debug server"}
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "port": os.getenv("PORT", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "unknown")
        }
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting minimal server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    logger.info("Railway Debug Script Starting...")
    
    # Check environment
    check_environment()
    
    # Test imports
    if not test_imports():
        logger.error("Import tests failed")
    
    # Try to start the actual app
    try:
        logger.info("Attempting to start main application...")
        from main import app
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start main app: {e}", exc_info=True)
        logger.info("Starting minimal debug server instead...")
        start_minimal_server()