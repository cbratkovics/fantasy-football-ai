"""
Simple prediction engine using RandomForest models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import joblib
import logging
import os
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats
from backend.ml.trend_analysis import PlayerTrendAnalyzer

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class SimplePredictionEngine:
    """Simple prediction engine using RandomForest models"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.trend_analyzer = PlayerTrendAnalyzer()
        
        # Load draft tier storage
        from backend.ml.draft_tier_storage import DraftTierStorage
        self.tier_storage = DraftTierStorage()
        
        # Load models
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self._load_models()
    
    def _load_models(self):
        """Load RandomForest models and scalers"""
        positions = ['QB', 'RB', 'WR', 'TE', 'K']
        
        for position in positions:
            model_path = self.models_dir / f'rf_model_{position}.pkl'
            scaler_path = self.models_dir / f'rf_scaler_{position}.pkl'
            feature_path = self.models_dir / f'rf_features_{position}.pkl'
            
            if model_path.exists() and scaler_path.exists() and feature_path.exists():
                try:
                    self.models[position] = joblib.load(model_path)
                    self.scalers[position] = joblib.load(scaler_path)
                    self.feature_names[position] = joblib.load(feature_path)
                    logger.info(f"Loaded RandomForest model for {position}")
                except Exception as e:
                    logger.error(f"Failed to load model for {position}: {str(e)}")
    
    def predict_player_week(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Dict[str, Any]:
        """Generate prediction for a player's performance in a specific week"""
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Check if model exists for position
            if player.position not in self.models:
                return {"error": f"No model available for position {player.position}"}
            
            # Get recent stats for feature preparation
            recent_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < week
            ).order_by(PlayerStats.week.desc()).limit(3).all()
            
            if not recent_stats:
                # Try previous season
                recent_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season - 1
                ).order_by(PlayerStats.week.desc()).limit(3).all()
        
        # Prepare features
        features = self._prepare_features(
            player, recent_stats, season, week
        )
        
        if features is None:
            return {"error": "Could not prepare features for prediction"}
        
        # Get prediction
        model = self.models[player.position]
        scaler = self.scalers[player.position]
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Get trend analysis
        trend_analysis = self.trend_analyzer.analyze_player_trends(player_id)
        
        # Apply trend adjustments
        if "error" not in trend_analysis:
            if trend_analysis['performance_trend']['overall_trend'] == 'improving':
                prediction *= 1.05
            elif trend_analysis['performance_trend']['overall_trend'] == 'declining':
                prediction *= 0.95
        
        # Get draft tier information
        tier_info = self.tier_storage.get_player_tier(player_id, season)
        
        # Build response
        response = {
            "player_id": player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.team,
            "season": season,
            "week": week,
            "predictions": {
                "ppr": {
                    "point_estimate": round(prediction, 2),
                    "lower_bound": round(prediction * 0.8, 2),
                    "upper_bound": round(prediction * 1.2, 2)
                },
                "standard": {
                    "point_estimate": round(prediction * 0.85, 2) if player.position in ['RB', 'WR', 'TE'] else round(prediction, 2)
                },
                "half_ppr": {
                    "point_estimate": round(prediction * 0.925, 2) if player.position in ['RB', 'WR', 'TE'] else round(prediction, 2)
                }
            },
            "confidence": {
                "score": 0.75,
                "level": "Medium"
            }
        }
        
        # Add tier information if available
        if tier_info:
            response["draft_tier"] = {
                "tier": tier_info.tier,
                "label": tier_info.tier_label,
                "confidence": float(tier_info.probability),
                "alternative_tiers": tier_info.alt_tiers or {}
            }
        
        return response
    
    def _prepare_features(
        self,
        player: Player,
        recent_stats: List[PlayerStats],
        season: int,
        week: int
    ) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        position = player.position
        feature_names = self.feature_names.get(position, [])
        
        # Create feature dict with default values
        features = {
            'season': season,
            'week': week,
            'age': player.age,
            'years_exp': player.years_exp
        }
        
        # Add average recent stats
        if recent_stats:
            # Average the recent stats
            avg_stats = {}
            for stat in recent_stats:
                if stat.stats:
                    for key, value in stat.stats.items():
                        if key in feature_names and isinstance(value, (int, float)):
                            if key not in avg_stats:
                                avg_stats[key] = []
                            avg_stats[key].append(value)
            
            # Calculate averages
            for key, values in avg_stats.items():
                features[key] = np.mean(values)
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        return np.array(feature_vector) if feature_vector else None


# Example usage
if __name__ == "__main__":
    predictor = SimplePredictionEngine()
    
    # Test prediction
    player_id = "6783"  # Example player
    prediction = predictor.predict_player_week(
        player_id=player_id,
        season=2024,
        week=10
    )
    
    if "error" not in prediction:
        print(f"Player: {prediction['player_name']}")
        print(f"Position: {prediction['position']}")
        print(f"Week {prediction['week']} Predictions:")
        print(f"  PPR: {prediction['predictions']['ppr']['point_estimate']}")
    else:
        print(f"Error: {prediction['error']}")