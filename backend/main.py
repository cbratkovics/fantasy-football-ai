"""
FastAPI Backend for Fantasy Football AI - Full Version with Database
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from backend.api import auth, players, predictions, subscriptions
from backend.models.database import engine, Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LLM components (optional)
try:
    from backend.api import llm_endpoints
    from backend.services.llm_service import LLMService
    from backend.services.vector_store import VectorStoreService
    LLM_AVAILABLE = True
    logger.info("LLM services imported successfully")
except ImportError as e:
    logger.warning(f"LLM services not available: {e}")
    LLM_AVAILABLE = False

# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Initialize LLM services if available
    if LLM_AVAILABLE:
        logger.info("Initializing LLM services...")
        try:
            # Initialize vector store
            vector_store = VectorStoreService(openai_api_key=os.getenv("OPENAI_API_KEY"))
            await vector_store.initialize()
            
            # Index player profiles and draft scenarios
            await vector_store.index_player_profiles()
            await vector_store.index_draft_scenarios()
            
            logger.info("LLM services initialized successfully")
        except Exception as e:
            logger.error(f"LLM service initialization failed: {e}")
    else:
        logger.info("LLM services not available - running without AI features")
    
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
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "fantasy-football-ai-backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
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

# Include LLM router if available
if LLM_AVAILABLE:
    app.include_router(llm_endpoints.router, prefix="/api/llm", tags=["LLM Services"])
    logger.info("LLM endpoints registered successfully")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)