"""
Enhanced Predictions API with Transparency and Monetization
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from backend.models.database import get_db, User, PredictionUsage
from backend.models.schemas import (
    EnhancedPredictionResponse,
    WeeklyRankingsResponse,
    AccuracyReport
)
from backend.services.predictor import EnhancedPredictor
from backend.services.stripe_service import StripeService
from backend.api.auth import get_current_user
from backend.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter()
predictor = EnhancedPredictor()
stripe_service = StripeService()
rate_limiter = RateLimiter()


@router.get("/player/{player_id}", response_model=EnhancedPredictionResponse)
async def get_player_prediction(
    player_id: str,
    season: int = Query(2024, ge=2020, le=2030),
    week: int = Query(1, ge=1, le=18),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get enhanced prediction for a specific player
    Includes ML predictions, confidence intervals, and plain English explanations
    """
    
    # Check subscription status and rate limits
    subscription = await stripe_service.check_subscription_status(current_user.id, db)
    
    if not subscription['has_access']:
        # Free tier rate limiting
        usage_allowed = await rate_limiter.check_prediction_usage(
            user_id=current_user.id,
            db=db,
            limit=5
        )
        
        if not usage_allowed:
            raise HTTPException(
                status_code=429,
                detail="Weekly prediction limit reached. Upgrade to Pro for unlimited predictions."
            )
    
    # Generate prediction with explanation
    try:
        prediction = await predictor.predict_with_explanation(
            player_id=player_id,
            season=season,
            week=week,
            db=db,
            include_tiers=True
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])
        
        # Track usage for free users
        if not subscription['has_access']:
            await rate_limiter.increment_usage(current_user.id, db)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed for player {player_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")


@router.post("/bulk")
async def get_bulk_predictions(
    player_ids: List[str],
    season: int = Query(2024, ge=2020, le=2030),
    week: int = Query(1, ge=1, le=18),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get predictions for multiple players
    Limited to 10 players for free users, unlimited for Pro
    """
    
    # Check subscription
    subscription = await stripe_service.check_subscription_status(current_user.id, db)
    
    # Limit players for free users
    if not subscription['has_access'] and len(player_ids) > 10:
        raise HTTPException(
            status_code=403,
            detail="Free users limited to 10 players per request. Upgrade to Pro for unlimited."
        )
    
    # Rate limiting for free users
    if not subscription['has_access']:
        usage_allowed = await rate_limiter.check_prediction_usage(
            user_id=current_user.id,
            db=db,
            limit=5
        )
        
        if not usage_allowed:
            raise HTTPException(
                status_code=429,
                detail="Weekly prediction limit reached. Upgrade to Pro for unlimited predictions."
            )
    
    # Generate predictions
    try:
        predictions = await predictor.bulk_predict(
            player_ids=player_ids,
            season=season,
            week=week,
            db=db
        )
        
        # Track usage
        if not subscription['has_access']:
            await rate_limiter.increment_usage(current_user.id, db)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "season": season,
            "week": week
        }
        
    except Exception as e:
        logger.error(f"Bulk prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Bulk prediction failed")


@router.get("/rankings/{position}", response_model=WeeklyRankingsResponse)
async def get_position_rankings(
    position: str,
    season: int = Query(2024, ge=2020, le=2030),
    week: int = Query(1, ge=1, le=18),
    limit: int = Query(50, ge=10, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get weekly rankings for a position with predictions
    Free users see top 20, Pro users see full rankings
    """
    
    # Validate position
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    if position.upper() not in valid_positions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid position. Must be one of: {', '.join(valid_positions)}"
        )
    
    # Check subscription
    subscription = await stripe_service.check_subscription_status(current_user.id, db)
    
    # Limit results for free users
    if not subscription['has_access']:
        limit = min(limit, 20)
    
    # Generate rankings
    try:
        rankings = await predictor.get_weekly_rankings(
            season=season,
            week=week,
            position=position.upper(),
            db=db,
            limit=limit
        )
        
        return rankings
        
    except Exception as e:
        logger.error(f"Rankings generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Rankings generation failed")


@router.get("/accuracy/report", response_model=AccuracyReport)
async def get_accuracy_report(
    season: int = Query(2024, ge=2020, le=2030),
    week: int = Query(1, ge=1, le=18),
    db: Session = Depends(get_db)
):
    """
    Get public accuracy report for completed weeks
    Shows model performance vs actual results
    """
    
    # Check if week is completed (simple check - in production would use schedule API)
    current_week = datetime.now().isocalendar()[1] - 35  # Rough NFL week calculation
    if season == 2024 and week >= current_week:
        raise HTTPException(
            status_code=400,
            detail="Accuracy reports only available for completed weeks"
        )
    
    # Get evaluation results
    evaluation = predictor.prediction_engine.evaluate_predictions(season, week)
    
    if "error" in evaluation:
        raise HTTPException(status_code=404, detail=evaluation["error"])
    
    return evaluation


@router.get("/lineup-optimizer")
async def optimize_lineup(
    week: int = Query(1, ge=1, le=18),
    league_settings: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Lineup optimizer - Pro feature only
    Suggests optimal lineup based on predictions
    """
    
    # Check Pro subscription
    subscription = await stripe_service.check_subscription_status(current_user.id, db)
    
    if not subscription['has_access']:
        raise HTTPException(
            status_code=403,
            detail="Lineup optimizer is a Pro feature. Upgrade to access."
        )
    
    # TODO: Implement lineup optimization logic
    # For now, return placeholder
    return {
        "message": "Lineup optimizer coming soon",
        "week": week,
        "feature": "pro"
    }


@router.get("/draft-assistant/recommendations")
async def get_draft_recommendations(
    round: int = Query(1, ge=1, le=16),
    pick: int = Query(1, ge=1, le=12),
    drafted_players: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get draft recommendations based on tiers and value
    Available to all users during draft season (August)
    """
    
    # Check if draft season or Pro user
    is_draft_season = datetime.now().month == 8
    subscription = await stripe_service.check_subscription_status(current_user.id, db)
    
    if not is_draft_season and not subscription['has_access']:
        raise HTTPException(
            status_code=403,
            detail="Draft assistant available to all users in August or Pro users year-round"
        )
    
    # TODO: Implement draft recommendation logic
    # For now, return placeholder
    return {
        "round": round,
        "pick": pick,
        "recommendations": [],
        "message": "Draft assistant implementation in progress"
    }