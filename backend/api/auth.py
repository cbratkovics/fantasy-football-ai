"""Authentication API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import logging

from backend.models.database import User, get_db
from backend.models.schemas import UserCreate, UserLogin, UserResponse, Token

logger = logging.getLogger(__name__)

router = APIRouter()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # For MVP, return mock user
    return UserResponse(
        id="12345",
        email=user.email,
        username=user.username,
        subscription_tier="free",
        created_at=datetime.utcnow(),
        is_active=True
    )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return JWT token"""
    # For MVP, return mock token
    return Token(
        access_token="mock_jwt_token_12345",
        token_type="bearer"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user info"""
    # For MVP, return mock user
    return UserResponse(
        id="12345",
        email="user@example.com",
        username="testuser",
        subscription_tier="free",
        created_at=datetime.utcnow(),
        is_active=True
    )