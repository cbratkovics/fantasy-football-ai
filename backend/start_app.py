#!/usr/bin/env python3
"""
Startup script for Fantasy Football AI Backend
Ensures proper initialization and error handling
"""

import os
import sys
import logging

# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI application"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Fantasy Football AI Backend")
        logger.info("=" * 60)
        
        # Log environment
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"PORT: {os.getenv('PORT', '8000')}")
        logger.info(f"DATABASE_URL: {'Set' if os.getenv('DATABASE_URL') else 'Not set'}")
        logger.info(f"REDIS_URL: {'Set' if os.getenv('REDIS_URL') else 'Not set'}")
        
        # Import and run uvicorn
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()