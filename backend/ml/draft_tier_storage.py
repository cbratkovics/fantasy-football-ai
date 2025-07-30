"""
Draft Tier Storage Module
Handles storing and retrieving GMM cluster assignments in the database
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional
import joblib
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from backend.models.database import DraftTier as DraftTierModel, Player
from backend.ml.gmm_clustering import GMMDraftOptimizer, DraftTier

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class DraftTierStorage:
    """Manages draft tier storage and retrieval"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Load GMM model if exists
        self.gmm_model = None
        model_path = self.models_dir / 'gmm_draft_tiers.pkl'
        if model_path.exists():
            try:
                self.gmm_model = joblib.load(model_path)
                logger.info("Loaded GMM model from disk")
            except Exception as e:
                logger.error(f"Failed to load GMM model: {str(e)}")
    
    def store_draft_tiers(self, draft_tiers: List[DraftTier], season: int = 2024):
        """Store draft tier assignments in database"""
        with self.SessionLocal() as db:
            # Clear existing tiers for this season
            db.query(DraftTierModel).filter(DraftTierModel.season == season).delete()
            
            # Store each tier assignment
            stored_count = 0
            for tier in draft_tiers:
                try:
                    db_tier = DraftTierModel(
                        player_id=tier.player_id,
                        tier=tier.tier,
                        tier_label=tier.tier_label,
                        probability=tier.probability,
                        alt_tiers=tier.alternative_tiers,
                        cluster_features={'expected_points': tier.expected_points, 'cluster_distance': tier.cluster_center_distance},
                        season=season
                    )
                    db.add(db_tier)
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Failed to store tier for player {tier.player_id}: {str(e)}")
            
            db.commit()
            logger.info(f"Stored {stored_count} tier assignments in database")
    
    def get_player_tier(self, player_id: str, season: int = 2024) -> Optional[DraftTierModel]:
        """Get tier assignment for a specific player"""
        with self.SessionLocal() as db:
            return db.query(DraftTierModel).filter(
                DraftTierModel.player_id == player_id,
                DraftTierModel.season == season
            ).first()
    
    def get_tier_players(self, tier_number: int, season: int = 2024) -> List[Dict]:
        """Get all players in a specific tier"""
        with self.SessionLocal() as db:
            results = db.query(
                DraftTierModel,
                Player.first_name,
                Player.last_name,
                Player.position,
                Player.team
            ).join(
                Player, Player.player_id == DraftTierModel.player_id
            ).filter(
                DraftTierModel.tier == tier_number,
                DraftTierModel.season == season
            ).order_by(
                DraftTierModel.probability.desc()
            ).all()
            
            players = []
            for tier, first_name, last_name, position, team in results:
                players.append({
                    'player_id': tier.player_id,
                    'name': f"{first_name} {last_name}",
                    'position': position,
                    'team': team,
                    'expected_points': tier.cluster_features.get('expected_points', 0) if tier.cluster_features else 0,
                    'confidence': tier.probability,
                    'tier_label': tier.tier_label
                })
            
            return players
    
    def get_all_tiers_summary(self, season: int = 2024) -> Dict[int, Dict]:
        """Get summary of all tiers"""
        summary = {}
        
        for tier_num in range(1, 17):
            players = self.get_tier_players(tier_num, season)
            if players:
                summary[tier_num] = {
                    'label': players[0]['tier_label'],
                    'player_count': len(players),
                    'avg_points': sum(p['expected_points'] for p in players) / len(players),
                    'positions': {},
                    'top_players': players[:5]  # Top 5 players
                }
                
                # Count positions
                for player in players:
                    pos = player['position']
                    summary[tier_num]['positions'][pos] = summary[tier_num]['positions'].get(pos, 0) + 1
        
        return summary


# Example usage
if __name__ == "__main__":
    storage = DraftTierStorage()
    
    # Get tier summary
    summary = storage.get_all_tiers_summary()
    
    print("Draft Tier Summary:")
    print("="*60)
    
    for tier_num, tier_info in summary.items():
        print(f"\nTier {tier_num}: {tier_info['label']}")
        print(f"  Players: {tier_info['player_count']}")
        print(f"  Avg Points: {tier_info['avg_points']:.1f}")
        print(f"  Positions: {tier_info['positions']}")
        print("  Top Players:")
        for player in tier_info['top_players'][:3]:
            print(f"    - {player['name']} ({player['position']}, {player['team']}) - {player['expected_points']:.1f} pts")