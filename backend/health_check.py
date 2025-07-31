"""
Minimal health check endpoint for debugging Railway deployment
"""

from fastapi import FastAPI
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fantasy Football AI API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "port": os.getenv("PORT", "8000"),
        "environment": os.getenv("ENVIRONMENT", "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting health check server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)