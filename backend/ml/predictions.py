"""
Data-Driven Predictions System
Uses trained ML models and historical data to generate weekly predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats, Prediction
from backend.ml.features import FeatureEngineer
from backend.ml.trend_analysis import PlayerTrendAnalyzer
from backend.ml.neural_network import FantasyNeuralNetwork

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class PredictionEngine:
    """Generates data-driven predictions using ML models and historical analysis"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.trend_analyzer = PlayerTrendAnalyzer()
        
        # Load models
        self.models = {}
        self.scalers = {}
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers"""
        positions = ['QB', 'RB', 'WR', 'TE', 'K']
        
        for position in positions:
            model_path = self.models_dir / f'nn_model_{position}.h5'
            scaler_path = self.models_dir / f'nn_scaler_{position}.pkl'
            
            if model_path.exists() and scaler_path.exists():
                try:
                    # Load neural network
                    predictor = FantasyNeuralNetwork()
                    predictor.load_model(model_path)
                    self.models[position] = predictor
                    
                    # Load scaler
                    self.scalers[position] = joblib.load(scaler_path)
                    
                    logger.info(f"Loaded model for {position}")
                except Exception as e:
                    logger.error(f"Failed to load model for {position}: {str(e)}")
    
    def predict_player_week(
        self,
        player_id: str,
        season: int,
        week: int,
        include_confidence: bool = True,
        include_factors: bool = True
    ) -> Dict[str, Any]:
        """
        Generate prediction for a player's performance in a specific week
        
        Returns:
            - Predicted fantasy points (all scoring formats)
            - Confidence intervals
            - Key factors influencing prediction
            - Risk assessment
        """
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Check if model exists for position
            if player.position not in self.models:
                return {"error": f"No model available for position {player.position}"}
            
            # Get historical stats for feature preparation
            historical_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season <= season
            ).order_by(PlayerStats.season, PlayerStats.week).all()
            
            if len(historical_stats) < 3:
                return {"error": "Insufficient historical data for prediction"}
        
        # Prepare features
        features = self._prepare_prediction_features(
            player, historical_stats, season, week
        )
        
        # Get base prediction from neural network
        nn_prediction = self._get_nn_prediction(player.position, features)
        
        # Analyze trends for adjustments
        trend_analysis = self.trend_analyzer.analyze_player_trends(player_id)
        
        # Combine predictions with trend adjustments
        final_prediction = self._combine_predictions(
            nn_prediction, trend_analysis, player.position
        )
        
        # Build response
        prediction_result = {
            "player_id": player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.team,
            "season": season,
            "week": week,
            "predictions": final_prediction
        }
        
        if include_confidence:
            prediction_result["confidence"] = self._calculate_confidence(
                historical_stats, trend_analysis
            )
        
        if include_factors:
            prediction_result["key_factors"] = self._identify_key_factors(
                features, trend_analysis, player.position
            )
        
        # Store prediction in database
        self._store_prediction(prediction_result)
        
        return prediction_result
    
    def predict_multiple_players(
        self,
        player_ids: List[str],
        season: int,
        week: int
    ) -> List[Dict[str, Any]]:
        """Generate predictions for multiple players"""
        predictions = []
        
        for player_id in player_ids:
            try:
                prediction = self.predict_player_week(
                    player_id, season, week,
                    include_confidence=True,
                    include_factors=False  # Lighter response for bulk
                )
                if "error" not in prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict for player {player_id}: {str(e)}")
        
        # Sort by predicted points
        predictions.sort(
            key=lambda x: x['predictions']['ppr']['point_estimate'],
            reverse=True
        )
        
        return predictions
    
    def _prepare_prediction_features(
        self,
        player: Player,
        historical_stats: List[PlayerStats],
        season: int,
        week: int
    ) -> np.ndarray:
        """Prepare features for prediction"""
        # Convert stats to DataFrame
        stats_data = []
        for stat in historical_stats:
            stat_dict = {
                'player_id': stat.player_id,
                'season': stat.season,
                'week': stat.week,
                'age': player.age,
                'years_exp': player.years_exp,
                'fantasy_points_ppr': stat.fantasy_points_ppr
            }
            # Add stats from JSONB
            if stat.stats:
                stat_dict.update(stat.stats)
            stats_data.append(stat_dict)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create lag features
        stats_df = stats_df.sort_values(['season', 'week'])
        lag_columns = ['fantasy_points_ppr', 'pts_ppr', 'pts_std']
        
        for col in lag_columns:
            if col in stats_df.columns:
                stats_df[f'{col}_lag1'] = stats_df[col].shift(1)
                stats_df[f'{col}_lag2'] = stats_df[col].shift(2)
                stats_df[f'{col}_rolling_avg'] = stats_df[col].rolling(3, min_periods=1).mean()
        
        # Get latest features (last row represents current state)
        latest_features = stats_df.iloc[-1].copy()
        
        # Add prediction week features
        latest_features['season'] = season
        latest_features['week'] = week
        
        # Get position-specific features
        feature_columns = self.feature_engineer.get_position_features(player.position)
        available_features = [col for col in feature_columns if col in latest_features.index]
        
        # Extract feature vector
        feature_vector = latest_features[available_features].values
        
        # Handle missing values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        return feature_vector.reshape(1, -1)
    
    def _get_nn_prediction(self, position: str, features: np.ndarray) -> Dict[str, float]:
        """Get prediction from neural network model"""
        model = self.models[position]
        scaler = self.scalers[position]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get point prediction
        point_prediction = model.predict(features_scaled)[0]
        
        # Get prediction intervals using MC Dropout
        prediction_intervals = model.predict_with_uncertainty(
            features_scaled, n_iterations=100
        )
        
        return {
            'point_estimate': float(point_prediction),
            'lower_bound': float(prediction_intervals['lower'][0]),
            'upper_bound': float(prediction_intervals['upper'][0]),
            'uncertainty': float(prediction_intervals['uncertainty'][0])
        }
    
    def _combine_predictions(
        self,
        nn_prediction: Dict[str, float],
        trend_analysis: Dict[str, Any],
        position: str
    ) -> Dict[str, Dict[str, float]]:
        """Combine NN predictions with trend adjustments"""
        base_prediction = nn_prediction['point_estimate']
        
        # Apply trend adjustments
        trend_factor = 1.0
        
        if 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']['overall_trend']
            if trend == 'improving':
                trend_factor = 1.05
            elif trend == 'declining':
                trend_factor = 0.95
        
        # Apply form adjustments
        if 'hot_cold_streaks' in trend_analysis:
            form = trend_analysis['hot_cold_streaks']['current_form']
            if form == 'Hot':
                trend_factor *= 1.03
            elif form == 'Cold':
                trend_factor *= 0.97
        
        # Calculate for different scoring formats
        # PPR is base, adjust for others
        ppr_prediction = base_prediction * trend_factor
        
        # Position-based adjustments for other formats
        if position in ['RB', 'WR', 'TE']:
            std_prediction = ppr_prediction * 0.85  # Rough approximation
            half_prediction = ppr_prediction * 0.925
        else:
            std_prediction = ppr_prediction
            half_prediction = ppr_prediction
        
        return {
            'ppr': {
                'point_estimate': round(ppr_prediction, 2),
                'lower_bound': round(nn_prediction['lower_bound'] * trend_factor, 2),
                'upper_bound': round(nn_prediction['upper_bound'] * trend_factor, 2),
                'uncertainty': nn_prediction['uncertainty']
            },
            'standard': {
                'point_estimate': round(std_prediction, 2),
                'lower_bound': round(std_prediction * 0.8, 2),
                'upper_bound': round(std_prediction * 1.2, 2)
            },
            'half_ppr': {
                'point_estimate': round(half_prediction, 2),
                'lower_bound': round(half_prediction * 0.85, 2),
                'upper_bound': round(half_prediction * 1.15, 2)
            }
        }
    
    def _calculate_confidence(
        self,
        historical_stats: List[PlayerStats],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate prediction confidence"""
        confidence_score = 0.5  # Base confidence
        
        # More historical data = higher confidence
        if len(historical_stats) > 30:
            confidence_score += 0.2
        elif len(historical_stats) > 15:
            confidence_score += 0.1
        
        # Consistency improves confidence
        if 'consistency_metrics' in trend_analysis:
            consistency = trend_analysis['consistency_metrics']['consistency_rating']
            if consistency == 'Very Consistent':
                confidence_score += 0.2
            elif consistency == 'Consistent':
                confidence_score += 0.1
            elif consistency == 'Volatile':
                confidence_score -= 0.1
        
        # Recent form clarity
        if 'hot_cold_streaks' in trend_analysis:
            if trend_analysis['hot_cold_streaks']['current_form'] in ['Hot', 'Cold']:
                confidence_score += 0.1  # Clear trend
        
        # Cap confidence
        confidence_score = min(0.95, max(0.2, confidence_score))
        
        # Classify confidence
        if confidence_score >= 0.8:
            confidence_level = "High"
        elif confidence_score >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            "score": round(confidence_score, 3),
            "level": confidence_level,
            "factors": {
                "sample_size": len(historical_stats),
                "consistency": trend_analysis.get('consistency_metrics', {}).get('consistency_rating', 'Unknown'),
                "recent_form": trend_analysis.get('hot_cold_streaks', {}).get('current_form', 'Unknown')
            }
        }
    
    def _identify_key_factors(
        self,
        features: np.ndarray,
        trend_analysis: Dict[str, Any],
        position: str
    ) -> List[Dict[str, Any]]:
        """Identify key factors influencing the prediction"""
        factors = []
        
        # Recent performance trend
        if 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']
            factors.append({
                "factor": "Performance Trend",
                "value": trend['overall_trend'],
                "impact": "positive" if trend['overall_trend'] == 'improving' else "negative" if trend['overall_trend'] == 'declining' else "neutral",
                "weight": "high"
            })
        
        # Current form
        if 'hot_cold_streaks' in trend_analysis:
            form = trend_analysis['hot_cold_streaks']
            factors.append({
                "factor": "Current Form",
                "value": form['current_form'],
                "details": f"Last 5 games: {form['last_5_games_avg']} pts",
                "impact": "positive" if form['current_form'] == 'Hot' else "negative" if form['current_form'] == 'Cold' else "neutral",
                "weight": "high"
            })
        
        # Consistency
        if 'consistency_metrics' in trend_analysis:
            consistency = trend_analysis['consistency_metrics']
            factors.append({
                "factor": "Consistency",
                "value": consistency['consistency_rating'],
                "details": f"CV: {consistency['coefficient_of_variation']}",
                "impact": "positive" if consistency['consistency_rating'] in ['Very Consistent', 'Consistent'] else "negative",
                "weight": "medium"
            })
        
        # Seasonal patterns
        if 'seasonal_patterns' in trend_analysis:
            patterns = trend_analysis['seasonal_patterns']
            # Check if current week is in player's strong weeks
            factors.append({
                "factor": "Seasonal Pattern",
                "value": "Historical performance this week",
                "impact": "neutral",  # Would need to check specific week
                "weight": "low"
            })
        
        return factors
    
    def _store_prediction(self, prediction_result: Dict[str, Any]):
        """Store prediction in database"""
        try:
            with self.SessionLocal() as db:
                # Create prediction record
                new_prediction = Prediction(
                    player_id=prediction_result['player_id'],
                    season=prediction_result['season'],
                    week=prediction_result['week'],
                    predicted_points=prediction_result['predictions']['ppr']['point_estimate'],
                    confidence_score=prediction_result.get('confidence', {}).get('score', 0.5),
                    prediction_data=prediction_result,
                    created_at=datetime.utcnow()
                )
                
                db.add(new_prediction)
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")
    
    def evaluate_predictions(
        self,
        season: int,
        week: int
    ) -> Dict[str, Any]:
        """Evaluate prediction accuracy for a completed week"""
        with self.SessionLocal() as db:
            # Get predictions for the week
            predictions = db.query(Prediction).filter(
                Prediction.season == season,
                Prediction.week == week
            ).all()
            
            if not predictions:
                return {"error": "No predictions found for this week"}
            
            # Get actual results
            evaluation_results = []
            
            for pred in predictions:
                actual_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == pred.player_id,
                    PlayerStats.season == season,
                    PlayerStats.week == week
                ).first()
                
                if actual_stats:
                    error = actual_stats.fantasy_points_ppr - pred.predicted_points
                    evaluation_results.append({
                        'player_id': pred.player_id,
                        'predicted': pred.predicted_points,
                        'actual': actual_stats.fantasy_points_ppr,
                        'error': error,
                        'absolute_error': abs(error),
                        'percentage_error': abs(error) / actual_stats.fantasy_points_ppr * 100 if actual_stats.fantasy_points_ppr > 0 else 0
                    })
            
            if not evaluation_results:
                return {"error": "No actual results available for evaluation"}
            
            # Calculate metrics
            errors = [r['absolute_error'] for r in evaluation_results]
            
            return {
                "week": week,
                "season": season,
                "predictions_evaluated": len(evaluation_results),
                "metrics": {
                    "mae": round(np.mean(errors), 2),
                    "rmse": round(np.sqrt(np.mean([e**2 for e in errors])), 2),
                    "median_error": round(np.median(errors), 2),
                    "within_3_points": sum(1 for e in errors if e <= 3) / len(errors) * 100,
                    "within_5_points": sum(1 for e in errors if e <= 5) / len(errors) * 100
                },
                "best_predictions": sorted(evaluation_results, key=lambda x: x['absolute_error'])[:5],
                "worst_predictions": sorted(evaluation_results, key=lambda x: x['absolute_error'], reverse=True)[:5]
            }


# Example usage
if __name__ == "__main__":
    import os
    
    predictor = PredictionEngine()
    
    # Predict for a player
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
        print(f"  PPR: {prediction['predictions']['ppr']['point_estimate']} ({prediction['predictions']['ppr']['lower_bound']}-{prediction['predictions']['ppr']['upper_bound']})")
        print(f"  Confidence: {prediction['confidence']['level']} ({prediction['confidence']['score']})")
        print("\nKey Factors:")
        for factor in prediction['key_factors']:
            print(f"  - {factor['factor']}: {factor['value']} ({factor['impact']})")
    else:
        print(f"Error: {prediction['error']}")