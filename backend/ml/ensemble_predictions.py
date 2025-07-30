"""
Ensemble Prediction Engine
Combines RandomForest, Neural Network, and GMM tier information for superior predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import joblib
import logging
import os
from pathlib import Path
from datetime import datetime
import tensorflow as tf

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats, Prediction
from backend.ml.trend_analysis import PlayerTrendAnalyzer
from backend.ml.draft_tier_storage import DraftTierStorage
from backend.ml.neural_network import FantasyNeuralNetwork
from backend.ml.momentum_detection import MomentumDetector

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class EnsemblePredictionEngine:
    """
    Advanced ensemble prediction engine combining:
    - RandomForest baseline predictions
    - Neural Network predictions
    - GMM tier information
    - Trend analysis adjustments
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.trend_analyzer = PlayerTrendAnalyzer()
        self.tier_storage = DraftTierStorage()
        self.momentum_detector = MomentumDetector()
        
        # Model storage
        self.rf_models = {}
        self.rf_scalers = {}
        self.rf_features = {}
        self.nn_models = {}
        self.nn_scalers = {}
        
        # Load all models
        self._load_models()
        
        # Ensemble weights (can be tuned)
        self.ensemble_weights = {
            'rf': 0.5,
            'nn': 0.3,
            'tier': 0.2
        }
    
    def _load_models(self):
        """Load all available models"""
        positions = ['QB', 'RB', 'WR', 'TE', 'K']
        
        for position in positions:
            # Load RandomForest
            rf_model_path = self.models_dir / f'rf_model_{position}.pkl'
            rf_scaler_path = self.models_dir / f'rf_scaler_{position}.pkl'
            rf_features_path = self.models_dir / f'rf_features_{position}.pkl'
            
            if rf_model_path.exists():
                try:
                    self.rf_models[position] = joblib.load(rf_model_path)
                    self.rf_scalers[position] = joblib.load(rf_scaler_path)
                    self.rf_features[position] = joblib.load(rf_features_path)
                    logger.info(f"Loaded RandomForest model for {position}")
                except Exception as e:
                    logger.error(f"Failed to load RF model for {position}: {str(e)}")
            
            # Load Neural Network
            nn_model_path = self.models_dir / f'nn_model_{position}.h5'
            nn_scaler_path = self.models_dir / f'nn_scaler_{position}.pkl'
            
            if nn_model_path.exists():
                try:
                    # Load the saved neural network
                    nn_model = FantasyNeuralNetwork(input_dim=17)  # Will be overridden by load
                    nn_model.load_model(str(self.models_dir / f'nn_model_{position}'))
                    self.nn_models[position] = nn_model
                    
                    if nn_scaler_path.exists():
                        self.nn_scalers[position] = joblib.load(nn_scaler_path)
                    logger.info(f"Loaded Neural Network model for {position}")
                except Exception as e:
                    logger.error(f"Failed to load NN model for {position}: {str(e)}")
    
    def predict_player_week(
        self,
        player_id: str,
        season: int,
        week: int,
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate ensemble prediction for a player's performance
        
        Returns comprehensive prediction with multiple models and confidence
        """
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Get recent stats
            recent_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < week
            ).order_by(PlayerStats.week.desc()).limit(5).all()
            
            if not recent_stats:
                # Try previous season
                recent_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season - 1
                ).order_by(PlayerStats.week.desc()).limit(5).all()
        
        # Get individual predictions
        predictions = {}
        
        # 1. RandomForest prediction
        if player.position in self.rf_models:
            rf_pred = self._get_rf_prediction(player, recent_stats, season, week)
            if rf_pred is not None:
                predictions['random_forest'] = rf_pred
        
        # 2. Neural Network prediction
        if player.position in self.nn_models:
            nn_pred = self._get_nn_prediction(player, recent_stats, season, week)
            if nn_pred is not None:
                predictions['neural_network'] = nn_pred
        
        # 3. Get tier-based baseline
        tier_info = self.tier_storage.get_player_tier(player_id, season)
        if tier_info:
            tier_baseline = self._get_tier_baseline(tier_info, player.position)
            predictions['tier_baseline'] = tier_baseline
        
        # 4. Get trend analysis
        trend_analysis = self.trend_analyzer.analyze_player_trends(player_id)
        
        # 5. Get momentum analysis
        momentum_analysis = self.momentum_detector.analyze_player_momentum(
            player_id, season, week
        )
        
        # 6. Combine predictions with momentum adjustment
        ensemble_prediction = self._combine_predictions(
            predictions, 
            trend_analysis,
            player.position,
            momentum_analysis if "error" not in momentum_analysis else None
        )
        
        # Build comprehensive response
        response = {
            "player_id": player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.team,
            "season": season,
            "week": week,
            "predictions": {
                "ensemble": ensemble_prediction,
                "models": predictions
            },
            "confidence": self._calculate_confidence(predictions, trend_analysis),
            "trend_adjustment": self._get_trend_adjustment(trend_analysis)
        }
        
        # Add tier information
        if tier_info:
            response["draft_tier"] = {
                "tier": tier_info.tier,
                "label": tier_info.tier_label,
                "confidence": float(tier_info.probability)
            }
        
        # Add momentum information
        if momentum_analysis and "error" not in momentum_analysis:
            response["momentum"] = {
                "score": momentum_analysis["momentum_score"],
                "trend": momentum_analysis["trend"],
                "streak": momentum_analysis["streak"],
                "breakout_probability": momentum_analysis["predictions"]["breakout_probability"],
                "recommendation": momentum_analysis["predictions"]["recommendation"]
            }
        
        # Add explanations
        if include_explanations:
            response["explanations"] = self._generate_explanations(
                predictions, 
                trend_analysis,
                tier_info,
                ensemble_prediction
            )
        
        # Store prediction
        self._store_prediction(response)
        
        return response
    
    def _get_rf_prediction(
        self, 
        player: Player, 
        recent_stats: List[PlayerStats],
        season: int,
        week: int
    ) -> Optional[float]:
        """Get RandomForest prediction"""
        try:
            model = self.rf_models[player.position]
            scaler = self.rf_scalers[player.position]
            feature_names = self.rf_features[player.position]
            
            # Prepare features
            features = self._prepare_rf_features(
                player, recent_stats, season, week, feature_names
            )
            
            if features is None:
                return None
            
            # Scale and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"RF prediction failed: {str(e)}")
            return None
    
    def _get_nn_prediction(
        self,
        player: Player,
        recent_stats: List[PlayerStats],
        season: int,
        week: int
    ) -> Optional[float]:
        """Get Neural Network prediction"""
        try:
            nn_model = self.nn_models[player.position]
            scaler = self.nn_scalers.get(player.position)
            
            # Prepare features (NN uses different feature set)
            features = self._prepare_nn_features(
                player, recent_stats, season, week
            )
            
            if features is None or scaler is None:
                return None
            
            # Scale and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = nn_model.predict(features_scaled)
            
            # Extract prediction value
            if isinstance(prediction, np.ndarray):
                return float(prediction[0])
            elif hasattr(prediction, 'point_estimate'):
                return float(prediction.point_estimate)
            else:
                return float(prediction)
                
        except Exception as e:
            logger.error(f"NN prediction failed: {str(e)}")
            return None
    
    def _get_tier_baseline(
        self,
        tier_info,
        position: str
    ) -> float:
        """Get tier-based baseline prediction"""
        # Extract expected points from cluster features
        if tier_info.cluster_features:
            return tier_info.cluster_features.get('expected_points', 15.0)
        
        # Default tier baselines
        tier_baselines = {
            1: 25.0, 2: 20.0, 3: 17.5, 4: 15.5,
            5: 14.0, 6: 12.5, 7: 11.0, 8: 9.5,
            9: 8.0, 10: 7.0, 11: 6.0, 12: 5.0,
            13: 4.5, 14: 4.0, 15: 3.5, 16: 3.0
        }
        
        return tier_baselines.get(tier_info.tier, 10.0)
    
    def _combine_predictions(
        self,
        predictions: Dict[str, float],
        trend_analysis: Dict[str, Any],
        position: str,
        momentum_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Combine predictions using weighted ensemble"""
        if not predictions:
            return {
                "point_estimate": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "method": "none"
            }
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        # Use adaptive weights based on available models
        weights = {
            'random_forest': 0.5,
            'neural_network': 0.4,
            'tier_baseline': 0.1
        }
        
        for model, pred in predictions.items():
            weight = weights.get(model, 0.2)
            weighted_sum += pred * weight
            total_weight += weight
        
        base_prediction = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Apply trend adjustments
        trend_factor = self._get_trend_factor(trend_analysis)
        
        # Apply momentum adjustments if available
        momentum_factor = 1.0
        if momentum_analysis and "error" not in momentum_analysis:
            momentum_score = momentum_analysis.get("momentum_score", 0)
            # Moderate momentum adjustment (5% max)
            momentum_factor = 1.0 + (momentum_score * 0.05)
            momentum_factor = max(0.9, min(1.1, momentum_factor))
        
        adjusted_prediction = base_prediction * trend_factor * momentum_factor
        
        # Calculate uncertainty bounds
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values) if len(pred_values) > 1 else base_prediction * 0.15
        
        # Create prediction ranges for different scoring formats
        ppr_prediction = adjusted_prediction
        
        # Position-based adjustments for other formats
        if position in ['RB', 'WR', 'TE']:
            std_prediction = ppr_prediction * 0.85
            half_prediction = ppr_prediction * 0.925
        else:
            std_prediction = ppr_prediction
            half_prediction = ppr_prediction
        
        return {
            "ppr": {
                "point_estimate": round(ppr_prediction, 2),
                "lower_bound": round(ppr_prediction - 1.5 * pred_std, 2),
                "upper_bound": round(ppr_prediction + 1.5 * pred_std, 2)
            },
            "standard": {
                "point_estimate": round(std_prediction, 2),
                "lower_bound": round(std_prediction - 1.5 * pred_std, 2),
                "upper_bound": round(std_prediction + 1.5 * pred_std, 2)
            },
            "half_ppr": {
                "point_estimate": round(half_prediction, 2),
                "lower_bound": round(half_prediction - 1.5 * pred_std, 2),
                "upper_bound": round(half_prediction + 1.5 * pred_std, 2)
            },
            "method": "weighted_ensemble",
            "models_used": list(predictions.keys())
        }
    
    def _get_trend_factor(self, trend_analysis: Dict[str, Any]) -> float:
        """Calculate trend adjustment factor"""
        if "error" in trend_analysis:
            return 1.0
        
        factor = 1.0
        
        # Performance trend
        if 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']['overall_trend']
            if trend == 'improving':
                factor *= 1.05
            elif trend == 'declining':
                factor *= 0.95
        
        # Current form
        if 'hot_cold_streaks' in trend_analysis:
            form = trend_analysis['hot_cold_streaks']['current_form']
            if form == 'Hot':
                factor *= 1.03
            elif form == 'Cold':
                factor *= 0.97
        
        return factor
    
    def _get_trend_adjustment(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed trend adjustment information"""
        if "error" in trend_analysis:
            return {"factor": 1.0, "reasons": []}
        
        factor = self._get_trend_factor(trend_analysis)
        reasons = []
        
        if 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']['overall_trend']
            if trend != 'stable':
                reasons.append(f"Player is {trend}")
        
        if 'hot_cold_streaks' in trend_analysis:
            form = trend_analysis['hot_cold_streaks']['current_form']
            if form != 'Average':
                reasons.append(f"Current form: {form}")
        
        return {
            "factor": round(factor, 3),
            "percentage": round((factor - 1) * 100, 1),
            "reasons": reasons
        }
    
    def _calculate_confidence(
        self,
        predictions: Dict[str, float],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate prediction confidence"""
        base_confidence = 0.5
        
        # More models = higher confidence
        model_count = len(predictions)
        base_confidence += model_count * 0.1
        
        # Agreement between models
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            cv = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) > 0 else 1
            if cv < 0.15:  # High agreement
                base_confidence += 0.2
            elif cv < 0.30:  # Moderate agreement
                base_confidence += 0.1
        
        # Consistency bonus
        if 'consistency_metrics' in trend_analysis:
            consistency = trend_analysis['consistency_metrics']['consistency_rating']
            if consistency in ['Very Consistent', 'Consistent']:
                base_confidence += 0.1
        
        # Cap confidence
        confidence_score = min(0.95, base_confidence)
        
        return {
            "score": round(confidence_score, 3),
            "level": "High" if confidence_score >= 0.8 else "Medium" if confidence_score >= 0.6 else "Low",
            "factors": {
                "models_used": model_count,
                "model_agreement": "High" if len(predictions) > 1 and cv < 0.15 else "Moderate" if len(predictions) > 1 and cv < 0.30 else "Low",
                "player_consistency": trend_analysis.get('consistency_metrics', {}).get('consistency_rating', 'Unknown')
            }
        }
    
    def _generate_explanations(
        self,
        predictions: Dict[str, float],
        trend_analysis: Dict[str, Any],
        tier_info,
        ensemble_prediction: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable explanations"""
        explanations = []
        
        # Model predictions
        if predictions:
            avg_pred = np.mean(list(predictions.values()))
            explanations.append(f"Models predict an average of {avg_pred:.1f} points")
        
        # Tier context
        if tier_info:
            explanations.append(f"Player is in {tier_info.tier_label}")
        
        # Trend insights
        if 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']['overall_trend']
            if trend == 'improving':
                explanations.append("Recent performance shows improvement")
            elif trend == 'declining':
                explanations.append("Recent performance shows decline")
        
        # Form
        if 'hot_cold_streaks' in trend_analysis:
            form = trend_analysis['hot_cold_streaks']['current_form']
            if form == 'Hot':
                explanations.append("Player is currently on a hot streak")
            elif form == 'Cold':
                explanations.append("Player is in a cold streak")
        
        # Confidence
        ppr_est = ensemble_prediction['ppr']['point_estimate']
        ppr_lower = ensemble_prediction['ppr']['lower_bound']
        ppr_upper = ensemble_prediction['ppr']['upper_bound']
        explanations.append(f"Expected range: {ppr_lower:.1f} to {ppr_upper:.1f} PPR points")
        
        return explanations
    
    def _prepare_rf_features(
        self,
        player: Player,
        recent_stats: List[PlayerStats],
        season: int,
        week: int,
        feature_names: List[str]
    ) -> Optional[np.ndarray]:
        """Prepare features for RandomForest model"""
        features = {
            'season': season,
            'week': week,
            'age': player.age,
            'years_exp': player.years_exp
        }
        
        # Add average recent stats
        if recent_stats:
            avg_stats = {}
            for stat in recent_stats:
                if stat.stats:
                    for key, value in stat.stats.items():
                        if key in feature_names and isinstance(value, (int, float)):
                            if key not in avg_stats:
                                avg_stats[key] = []
                            avg_stats[key].append(value)
            
            for key, values in avg_stats.items():
                features[key] = np.mean(values)
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        return np.array(feature_vector) if feature_vector else None
    
    def _prepare_nn_features(
        self,
        player: Player,
        recent_stats: List[PlayerStats],
        season: int,
        week: int
    ) -> Optional[np.ndarray]:
        """Prepare features for Neural Network model"""
        # Similar to RF but may include additional engineered features
        # This is a simplified version - in production would use full feature engineering
        return self._prepare_rf_features(
            player, recent_stats, season, week,
            ['season', 'week', 'age', 'years_exp'] + 
            self._get_position_features(player.position)
        )
    
    def _get_position_features(self, position: str) -> List[str]:
        """Get position-specific features"""
        position_features = {
            'QB': ['pass_att', 'pass_cmp', 'pass_yd', 'pass_td', 'pass_int', 'rush_att', 'rush_yd'],
            'RB': ['rush_att', 'rush_yd', 'rec', 'rec_yd', 'rec_td', 'rec_tgt'],
            'WR': ['rec', 'rec_yd', 'rec_td', 'rec_tgt', 'rush_att', 'rush_yd'],
            'TE': ['rec', 'rec_yd', 'rec_td', 'rec_tgt'],
            'K': ['fgm', 'fga', 'xpm', 'xpa']
        }
        return position_features.get(position, [])
    
    def _store_prediction(self, prediction_result: Dict[str, Any]):
        """Store prediction in database"""
        try:
            with self.SessionLocal() as db:
                ppr_est = prediction_result['predictions']['ensemble']['ppr']['point_estimate']
                
                new_prediction = Prediction(
                    player_id=prediction_result['player_id'],
                    season=prediction_result['season'],
                    week=prediction_result['week'],
                    predicted_points=ppr_est,
                    confidence_interval={
                        'low': prediction_result['predictions']['ensemble']['ppr']['lower_bound'],
                        'high': prediction_result['predictions']['ensemble']['ppr']['upper_bound']
                    },
                    prediction_std=float(
                        (prediction_result['predictions']['ensemble']['ppr']['upper_bound'] - 
                         prediction_result['predictions']['ensemble']['ppr']['lower_bound']) / 3.0
                    ),
                    model_version='ensemble_v1',
                    model_type='ensemble',
                    features_used={
                        'models': prediction_result['predictions']['ensemble']['models_used'],
                        'confidence': prediction_result['confidence'],
                        'trend_adjustment': prediction_result['trend_adjustment']
                    }
                )
                
                db.add(new_prediction)
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")


# Example usage
if __name__ == "__main__":
    engine = EnsemblePredictionEngine()
    
    # Test prediction
    result = engine.predict_player_week(
        player_id="6783",
        season=2024,
        week=10
    )
    
    if "error" not in result:
        print(f"Player: {result['player_name']}")
        print(f"Position: {result['position']}")
        print(f"\nEnsemble Prediction (PPR): {result['predictions']['ensemble']['ppr']['point_estimate']}")
        print(f"Confidence: {result['confidence']['level']} ({result['confidence']['score']:.1%})")
        print(f"\nModel Predictions:")
        for model, pred in result['predictions']['models'].items():
            print(f"  - {model}: {pred:.2f}")
        print(f"\nExplanations:")
        for exp in result['explanations']:
            print(f"  - {exp}")
    else:
        print(f"Error: {result['error']}")