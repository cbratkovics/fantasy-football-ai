"""Player API endpoints using real database data"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from datetime import datetime
import logging

from backend.models.database import Player, PlayerStats, get_db
from backend.models.schemas import PlayerRanking

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/rankings", response_model=List[PlayerRanking])
async def get_player_rankings(
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE, K, DEF)"),
    limit: int = Query(100, le=500, description="Number of players to return"),
    db: Session = Depends(get_db)
):
    """Get player rankings from real database"""
    
    # Build query
    query = db.query(Player).filter(
        Player.status.in_(["Active", "Injured Reserve"])
    )
    
    # Apply position filter
    if position and position != "All":
        query = query.filter(Player.position == position)
    
    # Get players
    players = query.limit(limit).all()
    
    if not players:
        return []
    
    # Convert to rankings format
    rankings = []
    
    for idx, player in enumerate(players):
        # Calculate tier based on position and index
        tier, tier_label = calculate_player_tier(player.position, idx)
        
        # Get injury status from meta_data
        injury_status = None
        if player.meta_data and isinstance(player.meta_data, dict):
            injury_status = player.meta_data.get('injury_status')
        
        # For now, use placeholder values for fantasy points
        # These will be real once we have PlayerStats data
        base_points = {
            'QB': 20.0,
            'RB': 15.0,
            'WR': 14.0,
            'TE': 10.0,
            'K': 8.0,
            'DEF': 9.0
        }.get(player.position, 10.0)
        
        # Add some variance based on player attributes
        variance = (player.age or 25) * 0.1 + (player.years_exp or 0) * 0.2
        predicted_points = base_points - variance + (idx * -0.1)
        
        rankings.append({
            "player_id": player.player_id,
            "name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.team,
            "age": player.age,
            "years_exp": player.years_exp,
            "tier": tier,
            "tier_label": tier_label,
            "predicted_points": round(max(5.0, predicted_points), 1),
            "confidence_interval": {
                "low": round(max(3.0, predicted_points - 3), 1),
                "high": round(predicted_points + 3, 1)
            },
            "trend": "stable",  # Will be calculated from stats later
            "injury_status": injury_status,
            "status": player.status
        })
    
    # Sort by predicted points
    rankings.sort(key=lambda x: x["predicted_points"], reverse=True)
    
    return rankings


@router.get("/{player_id}")
async def get_player_detail(
    player_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific player"""
    
    # Find player by player_id
    player = db.query(Player).filter(Player.player_id == player_id).first()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get injury status from meta_data
    injury_status = None
    additional_info = {}
    if player.meta_data and isinstance(player.meta_data, dict):
        injury_status = player.meta_data.get('injury_status')
        additional_info = {
            'height': player.meta_data.get('height'),
            'weight': player.meta_data.get('weight'),
            'college': player.meta_data.get('college'),
            'birth_date': player.meta_data.get('birth_date')
        }
    
    # For now, return player info without stats
    # Stats will be added once we implement stats fetching
    return {
        "player_id": player.player_id,
        "name": f"{player.first_name} {player.last_name}",
        "first_name": player.first_name,
        "last_name": player.last_name,
        "position": player.position,
        "team": player.team,
        "age": player.age,
        "years_exp": player.years_exp,
        "status": player.status,
        "injury_status": injury_status,
        "fantasy_positions": player.fantasy_positions,
        "additional_info": additional_info,
        "last_updated": player.updated_at.isoformat() if player.updated_at else None,
        "season_stats": {
            "games_played": 0,  # Will be populated from PlayerStats
            "total_points_ppr": 0,
            "avg_points_ppr": 0
        }
    }


def calculate_player_tier(position: str, rank_index: int) -> tuple[int, str]:
    """Calculate player tier based on position and ranking"""
    
    # Position-specific tier sizes
    tier_sizes = {
        "QB": [3, 6, 9, 12, 18, 24],  # Top 3 are tier 1, next 3 are tier 2, etc.
        "RB": [5, 10, 15, 24, 36, 48],
        "WR": [5, 10, 15, 24, 36, 48],
        "TE": [3, 6, 9, 12, 18, 24],
        "K": [3, 6, 10, 15, 20, 32],
        "DEF": [3, 6, 10, 15, 20, 32]
    }
    
    tier_labels = [
        "Elite - Round 1-2",
        "Strong - Round 3-4",
        "Solid - Round 5-7",
        "Good - Round 8-10",
        "Depth - Round 11-13",
        "Bench - Round 14+"
    ]
    
    sizes = tier_sizes.get(position, tier_sizes["WR"])
    
    current_tier = 1
    for size in sizes:
        if rank_index < size:
            return current_tier, tier_labels[current_tier - 1]
        current_tier += 1
    
    return len(tier_labels), tier_labels[-1]