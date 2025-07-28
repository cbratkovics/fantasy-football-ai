"""
FastAPI Backend for Fantasy Football AI - Minimal Version
Working API without database dependencies for testing
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    """Application settings"""
    API_TITLE: str = "Fantasy Football AI API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Advanced ML-powered fantasy football predictions"

settings = Settings()

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserBase(BaseModel):
    """Base user model"""
    email: str
    username: Optional[str] = None

class UserCreate(UserBase):
    """User creation model"""
    password: str

class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class PlayerRanking(BaseModel):
    """Player ranking response"""
    player_id: str
    name: str
    position: str
    team: Optional[str]
    tier: int
    tier_label: str
    predicted_points: float
    confidence_interval: Dict[str, float]
    trend: str

# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Fantasy Football AI API",
        "version": settings.API_VERSION,
        "status": "healthy"
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.API_VERSION
    }

# Authentication endpoints (mock for now)
@app.post("/auth/register", response_model=Token, tags=["Authentication"])
async def register(user_data: UserCreate):
    """Register new user - mock implementation"""
    logger.info(f"Mock registration for {user_data.email}")
    return {
        "access_token": "mock_token_for_" + user_data.email,
        "token_type": "bearer",
        "expires_in": 3600
    }

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(email: str, password: str):
    """Login user - mock implementation"""
    logger.info(f"Mock login for {email}")
    return {
        "access_token": "mock_token_for_" + email,
        "token_type": "bearer",
        "expires_in": 3600
    }

# Player endpoints with mock data
@app.get("/players/rankings", response_model=List[PlayerRanking], tags=["Players"])
async def get_player_rankings(
    position: Optional[str] = Query(None, description="Filter by position"),
    limit: int = Query(10, ge=1, le=50),
    scoring: str = Query("ppr", description="Scoring type")
):
    """Get player rankings - mock implementation"""
    
    # Mock data for testing
    mock_players = [
        PlayerRanking(
            player_id="1",
            name="Josh Allen",
            position="QB",
            team="BUF",
            tier=1,
            tier_label="Elite - Round 1",
            predicted_points=24.5,
            confidence_interval={"low": 20.0, "high": 29.0},
            trend="up"
        ),
        PlayerRanking(
            player_id="2",
            name="Christian McCaffrey",
            position="RB",
            team="SF",
            tier=1,
            tier_label="Elite - Round 1",
            predicted_points=22.3,
            confidence_interval={"low": 18.0, "high": 26.5},
            trend="stable"
        ),
        PlayerRanking(
            player_id="3",
            name="Tyreek Hill",
            position="WR",
            team="MIA",
            tier=1,
            tier_label="Elite - Round 1",
            predicted_points=19.8,
            confidence_interval={"low": 16.0, "high": 23.5},
            trend="up"
        )
    ]
    
    # Filter by position if specified
    if position:
        mock_players = [p for p in mock_players if p.position == position]
    
    return mock_players[:limit]

# API Documentation info
@app.get("/docs/info", tags=["Documentation"])
async def api_info():
    """Get API information"""
    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "health": "/health",
            "auth": ["/auth/register", "/auth/login"],
            "players": ["/players/rankings"],
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
