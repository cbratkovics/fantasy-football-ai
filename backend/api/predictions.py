"""Predictions API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import logging

from backend.models.database import Player, PlayerStats, Prediction, get_db
from backend.models.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/custom", response_model=List[PredictionResponse])
async def get_custom_predictions(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Get custom predictions for specified players and week"""
    
    predictions = []
    
    for player_id in request.player_ids:
        # Get player from database
        player = db.query(Player).filter(Player.player_id == player_id).first()
        
        if not player:
            continue
            
        # For MVP, use simple prediction logic
        base_points = {
            'QB': 20.0,
            'RB': 15.0,
            'WR': 14.0,
            'TE': 10.0,
            'K': 8.0,
            'DEF': 9.0
        }.get(player.position, 10.0)
        
        # Adjust for PPR scoring
        if request.scoring_type == "ppr" and player.position in ["RB", "WR", "TE"]:
            base_points += 3.0
        elif request.scoring_type == "half" and player.position in ["RB", "WR", "TE"]:
            base_points += 1.5
            
        predictions.append(PredictionResponse(
            player_id=player_id,
            player_name=f"{player.first_name} {player.last_name}",
            week=request.week,
            predicted_points=base_points,
            floor=base_points - 5.0,
            ceiling=base_points + 5.0,
            confidence=0.75,
            factors={
                "matchup": "favorable",
                "recent_form": "trending up",
                "injury_risk": "low"
            }
        ))
    
    return predictions


@router.get("/week/{week}")
async def get_weekly_predictions(
    week: int,
    position: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all predictions for a specific week"""
    
    # Build query
    query = db.query(Player).filter(
        Player.status.in_(["Active", "Injured Reserve"])
    )
    
    if position:
        query = query.filter(Player.position == position)
        
    players = query.limit(limit).all()
    
    predictions = []
    for player in players:
        base_points = {
            'QB': 20.0,
            'RB': 15.0,
            'WR': 14.0,
            'TE': 10.0,
            'K': 8.0,
            'DEF': 9.0
        }.get(player.position, 10.0)
        
        predictions.append({
            "player_id": player.player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.team,
            "week": week,
            "predicted_points": base_points,
            "confidence": 0.75
        })
    
    return predictions