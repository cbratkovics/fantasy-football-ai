#!/usr/bin/env python3
"""
Simple startup script for Railway debugging
Uses minimal FastAPI app without complex dependencies
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
    """Start the minimal FastAPI application"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Simple Fantasy Football AI Backend")
        logger.info("=" * 60)
        
        # Log environment
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"PORT: {os.getenv('PORT', '8000')}")
        logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH', 'Not set')}")
        
        # Log all environment variables (excluding sensitive ones)
        env_vars = {k: v for k, v in os.environ.items() if not any(sensitive in k.lower() for sensitive in ['key', 'secret', 'password', 'token'])}
        logger.info(f"Environment variables: {env_vars}")
        
        # Import and run uvicorn
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting server on 0.0.0.0:{port}")
        
        uvicorn.run(
            "main_simple:app",
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