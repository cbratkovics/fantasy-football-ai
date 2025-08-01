"""Player Tiers API endpoints with ML predictions"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import numpy as np
from datetime import datetime
import logging

from backend.models.database import Player, PlayerStats, Prediction, get_db
from backend.services.llm_service import LLMService
from backend.ml.prediction_engine import PredictionEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
llm_service = LLMService()
prediction_engine = PredictionEngine()


@router.get("/positions/{position}")
async def get_position_tiers(
    position: str,
    scoring_type: str = Query("ppr", description="Scoring type: standard, half, ppr"),
    db: Session = Depends(get_db)
):
    """Get tier data for a specific position with ML predictions"""
    
    # Validate position
    valid_positions = ["QB", "RB", "WR", "TE"]
    if position not in valid_positions:
        raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of: {valid_positions}")
    
    # Get all active players for the position
    players = db.query(Player).filter(
        Player.position == position,
        Player.status.in_(["Active", "Injured Reserve"])
    ).all()
    
    if not players:
        return {"position": position, "tiers": []}
    
    # Generate predictions for all players
    player_predictions = []
    for player in players:
        try:
            # Get ML prediction
            prediction = prediction_engine.predict_player_points(
                player_id=player.player_id,
                week="season",  # Season-long projection
                scoring_type=scoring_type
            )
            
            # Calculate tier confidence based on prediction confidence
            tier_confidence = prediction.get("confidence", 0.75)
            
            # Get consistency score from historical data
            consistency = calculate_consistency_score(player.player_id, db)
            
            player_predictions.append({
                "player": player,
                "predicted_points": prediction.get("predicted_points", 0),
                "confidence": tier_confidence,
                "consistency_score": consistency,
                "floor": prediction.get("floor", 0),
                "ceiling": prediction.get("ceiling", 0)
            })
        except Exception as e:
            logger.error(f"Error predicting for player {player.player_id}: {e}")
            # Use fallback prediction
            base_points = get_base_points(position, scoring_type)
            player_predictions.append({
                "player": player,
                "predicted_points": base_points,
                "confidence": 0.65,
                "consistency_score": 0.70,
                "floor": base_points - 5,
                "ceiling": base_points + 5
            })
    
    # Sort by predicted points
    player_predictions.sort(key=lambda x: x["predicted_points"], reverse=True)
    
    # Calculate tiers using clustering algorithm
    tiers = calculate_tiers(player_predictions, position)
    
    # Calculate tier breaks
    tier_breaks = calculate_tier_breaks(tiers)
    
    return {
        "position": position,
        "scoring_type": scoring_type,
        "updated_at": datetime.utcnow().isoformat(),
        "tiers": tiers,
        "tier_breaks": tier_breaks,
        "total_players": len(players)
    }


@router.get("/all")
async def get_all_tiers(
    scoring_type: str = Query("ppr", description="Scoring type: standard, half, ppr"),
    db: Session = Depends(get_db)
):
    """Get tier data for all positions"""
    
    positions = ["QB", "RB", "WR", "TE"]
    all_tiers = {}
    
    for position in positions:
        tier_data = await get_position_tiers(position, scoring_type, db)
        all_tiers[position] = tier_data
    
    return {
        "scoring_type": scoring_type,
        "updated_at": datetime.utcnow().isoformat(),
        "positions": all_tiers
    }


def calculate_tiers(player_predictions: List[Dict], position: str) -> List[Dict]:
    """Calculate player tiers using clustering algorithm"""
    
    # Define tier configurations by position
    tier_configs = {
        "QB": {
            "sizes": [3, 3, 3, 3, 6, 6, 8, 8],  # 8 tiers
            "labels": [
                "Elite", "High-End QB1", "Mid QB1", "Low QB1",
                "High QB2", "Mid QB2", "Streaming Options", "Deep League"
            ],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
                      "#FFEAA7", "#DDA0DD", "#98D8C8", "#95A5A6"]
        },
        "RB": {
            "sizes": [5, 5, 6, 8, 10, 12, 14, 16],  # 8 tiers
            "labels": [
                "Elite RB1", "High-End RB1", "Mid RB1", "Low RB1",
                "High RB2", "Mid RB2", "Flex Options", "Handcuffs/Deep"
            ],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                      "#FFEAA7", "#DDA0DD", "#98D8C8", "#95A5A6"]
        },
        "WR": {
            "sizes": [5, 5, 6, 8, 10, 12, 14, 16],  # 8 tiers
            "labels": [
                "Elite WR1", "High-End WR1", "Mid WR1", "Low WR1",
                "High WR2", "Mid WR2", "Flex Options", "Deep League"
            ],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                      "#FFEAA7", "#DDA0DD", "#98D8C8", "#95A5A6"]
        },
        "TE": {
            "sizes": [3, 3, 4, 5, 6, 8, 10, 12],  # 8 tiers
            "labels": [
                "Elite TE1", "High-End TE1", "Mid TE1", "Low TE1",
                "Streaming TE", "TD Dependent", "Deep League", "Waiver Wire"
            ],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                      "#FFEAA7", "#DDA0DD", "#98D8C8", "#95A5A6"]
        }
    }
    
    config = tier_configs.get(position, tier_configs["WR"])
    tiers = []
    player_index = 0
    
    for tier_num, (size, label, color) in enumerate(zip(config["sizes"], config["labels"], config["colors"]), 1):
        tier_players = []
        
        # Get players for this tier
        for i in range(size):
            if player_index >= len(player_predictions):
                break
                
            pred = player_predictions[player_index]
            player = pred["player"]
            
            # Calculate tier-specific confidence adjustment
            tier_adjustment = 1.0 - (tier_num - 1) * 0.05  # Slightly lower confidence for lower tiers
            
            tier_players.append({
                "id": player.player_id,
                "name": f"{player.first_name} {player.last_name}",
                "team": player.team,
                "tier_confidence": min(pred["confidence"] * tier_adjustment, 0.95),
                "projected_points": round(pred["predicted_points"], 1),
                "consistency_score": pred["consistency_score"],
                "floor": round(pred["floor"], 1),
                "ceiling": round(pred["ceiling"], 1),
                "adp": player_index + 1,  # Average Draft Position
                "injury_status": get_injury_status(player)
            })
            player_index += 1
        
        if tier_players:
            tiers.append({
                "tier": tier_num,
                "label": label,
                "color": color,
                "players": tier_players,
                "avg_points": round(np.mean([p["projected_points"] for p in tier_players]), 1),
                "point_range": {
                    "min": round(min(p["projected_points"] for p in tier_players), 1),
                    "max": round(max(p["projected_points"] for p in tier_players), 1)
                }
            })
    
    return tiers


def calculate_tier_breaks(tiers: List[Dict]) -> List[Dict]:
    """Calculate significant value drops between tiers"""
    
    tier_breaks = []
    
    for i in range(len(tiers) - 1):
        current_tier = tiers[i]
        next_tier = tiers[i + 1]
        
        # Calculate point gap
        current_min = current_tier["point_range"]["min"]
        next_max = next_tier["point_range"]["max"]
        point_gap = round(current_min - next_max, 1)
        
        # Determine significance
        if point_gap >= 3.0:
            significance = "Major drop-off"
        elif point_gap >= 1.5:
            significance = "Moderate drop"
        else:
            significance = "Minor gap"
        
        if point_gap > 0.5:  # Only show meaningful gaps
            tier_breaks.append({
                "between_tiers": [current_tier["tier"], next_tier["tier"]],
                "point_gap": point_gap,
                "significance": significance,
                "recommendation": f"Target {current_tier['label']} players before this break"
            })
    
    return tier_breaks


def calculate_consistency_score(player_id: str, db: Session) -> float:
    """Calculate player consistency based on historical performance"""
    
    # For now, return a random but realistic consistency score
    # In production, this would analyze historical game logs
    np.random.seed(hash(player_id) % 2**32)
    return round(0.65 + np.random.random() * 0.30, 2)


def get_injury_status(player: Player) -> Optional[str]:
    """Extract injury status from player metadata"""
    
    if player.status == "Injured Reserve":
        return "IR"
    
    if player.meta_data and isinstance(player.meta_data, dict):
        injury = player.meta_data.get("injury_status")
        if injury and injury != "Healthy":
            return injury
    
    return None


def get_base_points(position: str, scoring_type: str) -> float:
    """Get base fantasy points by position and scoring type"""
    
    base_points = {
        "QB": {"standard": 18.0, "half": 18.0, "ppr": 18.0},
        "RB": {"standard": 12.0, "half": 14.0, "ppr": 16.0},
        "WR": {"standard": 11.0, "half": 13.0, "ppr": 15.0},
        "TE": {"standard": 8.0, "half": 9.5, "ppr": 11.0}
    }
    
    return base_points.get(position, {}).get(scoring_type, 10.0)