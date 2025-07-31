"""
Minimal FastAPI Backend for Railway Health Check Testing
This version removes all complex dependencies and focuses on basic health check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Fantasy Football AI API",
    version="1.0.0",
    description="Advanced ML-powered fantasy football predictions"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Fantasy Football AI API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "service": "fantasy-football-ai-backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "port": os.getenv("PORT", "8000"),
        "environment": os.getenv("ENVIRONMENT", "production")
    }

# Ready endpoint
@app.get("/ready")
async def ready_check():
    """Ready check endpoint"""
    logger.info("Ready check endpoint accessed")
    return {
        "status": "ready",
        "service": "fantasy-football-ai-backend",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)