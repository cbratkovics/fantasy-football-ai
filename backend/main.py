"""
FastAPI Backend for Fantasy Football AI - Full Version with Database
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

# Import API routers
from backend.api import auth, players, predictions, subscriptions
from backend.models.database import engine, Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Application shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Fantasy Football AI API",
    version="1.0.0",
    description="Advanced ML-powered fantasy football predictions",
    lifespan=lifespan
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
    return {
        "message": "Fantasy Football AI API",
        "version": "1.0.0",
        "status": "healthy"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "fantasy-football-ai-backend",
        "version": "1.0.0"
    }

# Import new routers
from backend.api import predictions_v2, payments

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(players.router, prefix="/players", tags=["Players"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(predictions_v2.router, prefix="/api/v2/predictions", tags=["Predictions V2"])
app.include_router(payments.router, prefix="/api/payments", tags=["Payments"])
app.include_router(subscriptions.router, prefix="/subscriptions", tags=["Subscriptions"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)