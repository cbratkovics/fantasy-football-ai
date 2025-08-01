"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class SubscriptionTier(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    username: Optional[str] = None


class UserCreate(UserBase):
    """User creation model"""
    password: str


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """User response model"""
    id: str
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    created_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"


class PlayerRanking(BaseModel):
    """Player ranking response model"""
    player_id: str
    name: str
    position: str
    team: str
    age: Optional[int] = None
    years_exp: Optional[int] = None
    tier: int = 1
    tier_label: str = "Tier 1"
    predicted_points: float
    confidence_interval: Dict[str, float]
    trend: str = "stable"  # up, down, stable
    injury_status: Optional[str] = None
    status: Optional[str] = None
    
    class Config:
        from_attributes = True


class PlayerDetail(PlayerRanking):
    """Detailed player information"""
    season_stats: Dict[str, Any]
    recent_games: List[Dict[str, Any]]
    projections: Dict[str, Any]
    fantasy_positions: List[str]
    meta_data: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """Prediction request model"""
    player_ids: List[str]
    week: int = Field(ge=1, le=18)
    scoring_type: str = "ppr"  # std, ppr, half


class PredictionResponse(BaseModel):
    """Prediction response model"""
    player_id: str
    player_name: str
    week: int
    predicted_points: float
    floor: float
    ceiling: float
    confidence: float
    factors: Dict[str, Any]
    

class LeagueSettings(BaseModel):
    """League settings model"""
    platform: str = "sleeper"  # sleeper, espn, yahoo
    league_id: str
    scoring_settings: Dict[str, Any]
    roster_positions: List[str]
    

class DraftRecommendation(BaseModel):
    """Draft recommendation model"""
    round: int
    pick: int
    recommended_players: List[PlayerRanking]
    position_needs: Dict[str, int]
    strategy_notes: str


class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction with transparency"""
    player: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: Dict[str, Any]
    explanation: Dict[str, Any]
    draft_tier: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class CheckoutSessionRequest(BaseModel):
    """Stripe checkout session request"""
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    """Stripe checkout session response"""
    checkout_url: str
    session_id: str
    trial_days: Optional[int] = None


class SubscriptionStatus(BaseModel):
    """User subscription status"""
    status: str
    has_access: bool
    is_trial: Optional[bool] = False
    trial_ends: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    weekly_predictions_remaining: Optional[int] = None
    canceled: Optional[bool] = False


class WeeklyRankingsResponse(BaseModel):
    """Weekly rankings response"""
    position: str
    week: int
    season: int
    generated_at: datetime
    model_accuracy: float
    rankings: List[Dict[str, Any]]


class AccuracyReport(BaseModel):
    """Model accuracy report"""
    week: int
    season: int
    predictions_evaluated: int
    metrics: Dict[str, float]
    best_predictions: List[Dict[str, Any]]
    worst_predictions: List[Dict[str, Any]]