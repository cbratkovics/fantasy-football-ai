"""
Enhanced Feature Engineering Pipeline with Efficiency Ratio
Includes proprietary metrics and advanced feature creation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats
from backend.ml.efficiency_ratio import EfficiencyRatioCalculator
from backend.ml.trend_analysis import PlayerTrendAnalyzer

logger = logging.getLogger(__name__)

# Database connection
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class AdvancedFeatureEngineer:
    """
    Enhanced feature engineering with proprietary metrics including:
    - Efficiency Ratio
    - Momentum indicators
    - Advanced opponent adjustments
    - Game script predictions
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.efficiency_calculator = EfficiencyRatioCalculator()
        self.trend_analyzer = PlayerTrendAnalyzer()
        
        # Feature groups
        self.feature_groups = {
            'basic': ['age', 'years_exp', 'games_played'],
            'performance': ['avg_points', 'total_points', 'points_std'],
            'efficiency': ['efficiency_ratio', 'opp_efficiency', 'matchup_efficiency'],
            'momentum': ['momentum_3w', 'momentum_5w', 'trend_direction'],
            'consistency': ['consistency_score', 'floor', 'ceiling'],
            'opportunity': ['touches_per_game', 'target_share', 'red_zone_share'],
            'game_context': ['is_home', 'opponent_rank', 'implied_total', 'spread']
        }
    
    def engineer_features_for_player(
        self,
        player_id: str,
        season: int,
        week: int,
        lookback_weeks: int = 10
    ) -> Dict[str, Any]:
        """
        Create comprehensive feature set for a player including proprietary metrics
        """
        features = {}
        
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Basic features
            features.update(self._get_basic_features(player, season))
            
            # Get historical stats
            historical_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < week
            ).order_by(PlayerStats.week.desc()).limit(lookback_weeks).all()
            
            if not historical_stats:
                # Try previous season
                historical_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season - 1
                ).order_by(PlayerStats.week.desc()).limit(lookback_weeks).all()
        
        # Performance features
        features.update(self._get_performance_features(historical_stats))
        
        # Efficiency features (our proprietary metric)
        features.update(self._get_efficiency_features(player_id, season, historical_stats))
        
        # Momentum features
        features.update(self._get_momentum_features(historical_stats))
        
        # Consistency features
        features.update(self._get_consistency_features(historical_stats))
        
        # Opportunity features
        features.update(self._get_opportunity_features(historical_stats, player.position))
        
        # Game context features (would need game data)
        features.update(self._get_game_context_features(player_id, season, week))
        
        # Add metadata
        features['player_id'] = player_id
        features['player_name'] = f"{player.first_name} {player.last_name}"
        features['position'] = player.position
        features['season'] = season
        features['week'] = week
        
        return features
    
    def _get_basic_features(self, player: Player, season: int) -> Dict[str, float]:
        """Get basic player features"""
        return {
            'age': player.age or 25,  # Default age if missing
            'years_exp': player.years_exp or 1,
            'is_rookie': 1 if player.years_exp == 0 else 0
        }
    
    def _get_performance_features(self, stats: List[PlayerStats]) -> Dict[str, float]:
        """Calculate performance-based features"""
        if not stats:
            return {
                'avg_points_recent': 0,
                'total_points_recent': 0,
                'points_std': 0,
                'games_played_recent': 0
            }
        
        points = []
        for stat in stats:
            if stat.fantasy_points_ppr is not None:
                points.append(float(stat.fantasy_points_ppr))
        
        if not points:
            return {
                'avg_points_recent': 0,
                'total_points_recent': 0,
                'points_std': 0,
                'games_played_recent': 0
            }
        
        return {
            'avg_points_recent': np.mean(points),
            'total_points_recent': sum(points),
            'points_std': np.std(points) if len(points) > 1 else 0,
            'games_played_recent': len(points)
        }
    
    def _get_efficiency_features(
        self,
        player_id: str,
        season: int,
        stats: List[PlayerStats]
    ) -> Dict[str, float]:
        """Get efficiency-based features including our proprietary metric"""
        # Calculate overall efficiency ratio
        efficiency_result = self.efficiency_calculator.calculate_player_efficiency(
            player_id=player_id,
            season=season,
            include_components=True
        )
        
        if "error" in efficiency_result:
            return {
                'efficiency_ratio': 1.0,
                'efficiency_percentile': 50.0,
                'opp_efficiency': 1.0,
                'matchup_efficiency': 1.0
            }
        
        # Get recent efficiency trend
        trend_result = self.efficiency_calculator.calculate_weekly_efficiency_trend(
            player_id=player_id,
            season=season,
            last_n_weeks=5
        )
        
        recent_efficiency = 1.0
        efficiency_momentum = 0.0
        
        if "error" not in trend_result:
            recent_efficiency = trend_result['average_efficiency']
            efficiency_momentum = trend_result['trend_slope']
        
        return {
            'efficiency_ratio': efficiency_result['efficiency_ratio'],
            'efficiency_percentile': efficiency_result['percentile_rank'],
            'opp_efficiency': efficiency_result['components']['opportunity_efficiency'],
            'matchup_efficiency': efficiency_result['components']['matchup_efficiency'],
            'recent_efficiency': recent_efficiency,
            'efficiency_momentum': efficiency_momentum
        }
    
    def _get_momentum_features(self, stats: List[PlayerStats]) -> Dict[str, float]:
        """Calculate momentum indicators"""
        if len(stats) < 3:
            return {
                'momentum_3w': 0,
                'momentum_5w': 0,
                'trend_direction': 0,
                'volatility': 0
            }
        
        # Get points for momentum calculation
        points = []
        for stat in stats:
            if stat.fantasy_points_ppr is not None:
                points.append(float(stat.fantasy_points_ppr))
        
        if len(points) < 3:
            return {
                'momentum_3w': 0,
                'momentum_5w': 0,
                'trend_direction': 0,
                'volatility': 0
            }
        
        # 3-week momentum (recent vs prior)
        recent_3w = np.mean(points[:3]) if len(points) >= 3 else np.mean(points)
        prior_3w = np.mean(points[3:6]) if len(points) >= 6 else np.mean(points)
        momentum_3w = (recent_3w - prior_3w) / (prior_3w + 1e-6)
        
        # 5-week momentum
        recent_5w = np.mean(points[:5]) if len(points) >= 5 else np.mean(points)
        prior_5w = np.mean(points[5:10]) if len(points) >= 10 else np.mean(points)
        momentum_5w = (recent_5w - prior_5w) / (prior_5w + 1e-6)
        
        # Trend direction (simple linear regression slope)
        if len(points) >= 3:
            x = np.arange(len(points))
            slope = np.polyfit(x, points, 1)[0]
            trend_direction = 1 if slope > 0.5 else -1 if slope < -0.5 else 0
        else:
            trend_direction = 0
        
        # Volatility (coefficient of variation)
        volatility = np.std(points) / (np.mean(points) + 1e-6)
        
        return {
            'momentum_3w': momentum_3w,
            'momentum_5w': momentum_5w,
            'trend_direction': trend_direction,
            'volatility': volatility
        }
    
    def _get_consistency_features(self, stats: List[PlayerStats]) -> Dict[str, float]:
        """Calculate consistency metrics"""
        if not stats:
            return {
                'consistency_score': 0,
                'floor': 0,
                'ceiling': 0,
                'floor_ceiling_ratio': 0
            }
        
        points = []
        for stat in stats:
            if stat.fantasy_points_ppr is not None:
                points.append(float(stat.fantasy_points_ppr))
        
        if not points:
            return {
                'consistency_score': 0,
                'floor': 0,
                'ceiling': 0,
                'floor_ceiling_ratio': 0
            }
        
        # Consistency score (inverse of coefficient of variation)
        avg_points = np.mean(points)
        std_points = np.std(points) if len(points) > 1 else 0
        consistency = 1 - (std_points / (avg_points + 1e-6))
        
        # Floor and ceiling (20th and 80th percentiles)
        floor = np.percentile(points, 20) if len(points) >= 5 else min(points)
        ceiling = np.percentile(points, 80) if len(points) >= 5 else max(points)
        
        return {
            'consistency_score': consistency,
            'floor': floor,
            'ceiling': ceiling,
            'floor_ceiling_ratio': floor / (ceiling + 1e-6)
        }
    
    def _get_opportunity_features(
        self,
        stats: List[PlayerStats],
        position: str
    ) -> Dict[str, float]:
        """Calculate opportunity-based features"""
        if not stats:
            return {
                'touches_per_game': 0,
                'target_share': 0,
                'red_zone_share': 0,
                'snap_percentage': 0
            }
        
        # Position-specific opportunity metrics
        if position == 'QB':
            attempts = []
            for stat in stats:
                if stat.stats:
                    attempts.append(stat.stats.get('pass_att', 0))
            
            return {
                'attempts_per_game': np.mean(attempts) if attempts else 0,
                'rushing_attempts': 0,  # Would calculate from stats
                'red_zone_attempts': 0,  # Would need red zone data
                'snap_percentage': 0.95  # QBs typically play all snaps
            }
        
        elif position == 'RB':
            touches = []
            targets = []
            for stat in stats:
                if stat.stats:
                    rush_att = stat.stats.get('rush_att', 0)
                    rec = stat.stats.get('rec', 0)
                    touches.append(rush_att + rec)
                    targets.append(stat.stats.get('rec_tgt', 0))
            
            return {
                'touches_per_game': np.mean(touches) if touches else 0,
                'target_share': np.mean(targets) / 30 if targets else 0,  # Assume 30 team targets/game
                'red_zone_touches': 0,  # Would need red zone data
                'snap_percentage': 0  # Would need snap data
            }
        
        elif position in ['WR', 'TE']:
            targets = []
            receptions = []
            for stat in stats:
                if stat.stats:
                    targets.append(stat.stats.get('rec_tgt', 0))
                    receptions.append(stat.stats.get('rec', 0))
            
            return {
                'targets_per_game': np.mean(targets) if targets else 0,
                'target_share': np.mean(targets) / 30 if targets else 0,
                'catch_rate': np.mean(receptions) / np.mean(targets) if targets and np.mean(targets) > 0 else 0,
                'red_zone_targets': 0  # Would need red zone data
            }
        
        else:
            return {
                'touches_per_game': 0,
                'target_share': 0,
                'red_zone_share': 0,
                'snap_percentage': 0
            }
    
    def _get_game_context_features(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Get game context features (simplified - would need game data)"""
        # In production, would fetch actual game data
        # For now, return placeholder values
        return {
            'is_home': 0.5,  # 50/50 chance
            'opponent_rank_vs_position': 16,  # Average rank
            'implied_team_total': 24.5,  # Average implied total
            'game_spread': 0,  # Pick'em
            'game_total': 49,  # Average O/U
            'is_primetime': 0,  # Not primetime
            'is_division_game': 0.25  # 25% chance
        }
    
    def create_training_dataset(
        self,
        positions: List[str],
        seasons: List[int],
        min_games: int = 6
    ) -> pd.DataFrame:
        """Create comprehensive training dataset with all features"""
        all_features = []
        
        with self.SessionLocal() as db:
            # Get all players at positions
            players = db.query(Player).filter(
                Player.position.in_(positions)
            ).all()
            
            for player in players:
                player_id = player.player_id
                
                for season in seasons:
                    # Get all weeks for this player-season
                    weeks = db.query(PlayerStats.week).filter(
                        PlayerStats.player_id == player_id,
                        PlayerStats.season == season
                    ).distinct().all()
                    
                    if len(weeks) >= min_games:
                        for week_tuple in weeks:
                            week = week_tuple[0]
                            if week > 4:  # Need some history
                                # Engineer features for this player-week
                                features = self.engineer_features_for_player(
                                    player_id, season, week
                                )
                                
                                if "error" not in features:
                                    # Get target (actual points for that week)
                                    actual_stats = db.query(PlayerStats).filter(
                                        PlayerStats.player_id == player_id,
                                        PlayerStats.season == season,
                                        PlayerStats.week == week
                                    ).first()
                                    
                                    if actual_stats and actual_stats.fantasy_points_ppr:
                                        features['target_points'] = float(actual_stats.fantasy_points_ppr)
                                        all_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        logger.info(f"Created training dataset with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
    def get_feature_importance(
        self,
        model,
        feature_names: List[str],
        position: str
    ) -> pd.DataFrame:
        """Analyze feature importance for a trained model"""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'position': position
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add relative importance
        importance_df['relative_importance'] = (
            importance_df['importance'] / importance_df['importance'].sum()
        )
        
        return importance_df


# Example usage
if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    
    # Test feature engineering for a player
    features = engineer.engineer_features_for_player(
        player_id="6783",
        season=2024,
        week=10
    )
    
    if "error" not in features:
        print(f"Player: {features['player_name']}")
        print(f"Position: {features['position']}")
        print(f"\nKey Features:")
        print(f"  Efficiency Ratio: {features['efficiency_ratio']}")
        print(f"  Efficiency Percentile: {features['efficiency_percentile']}%")
        print(f"  Recent Momentum (3w): {features['momentum_3w']:+.2%}")
        print(f"  Consistency Score: {features['consistency_score']:.3f}")
        print(f"  Floor: {features['floor']:.1f}")
        print(f"  Ceiling: {features['ceiling']:.1f}")
    else:
        print(f"Error: {features['error']}")