"""
Fantasy Football AI - Feature Engineering Pipeline
Transforms raw NFL statistics into ML-ready features for GMM clustering and NN prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlayerFeatures:
    """Container for engineered player features"""
    player_id: str
    position: str
    week: int
    season: int
    # Core features
    ppg: float
    consistency_score: float
    efficiency_ratio: float
    momentum_score: float
    boom_bust_ratio: float
    recent_trend: float

class FeatureEngineer:
    """
    Transforms raw NFL statistics into engineered features for ML models.
    
    Core Features (6):
    1. Points Per Game (PPG) - Primary performance metric
    2. Consistency Score - PPG/Standard Deviation (higher = more consistent)
    3. Efficiency Ratio - Actual vs Projected performance
    4. Momentum Score - 3-week weighted rolling average
    5. Boom/Bust Ratio - Frequency of high variance games
    6. Recent Trend - Linear trend of last 4 games
    """
    
    def __init__(self, lookback_weeks: int = 10):
        self.lookback_weeks = lookback_weeks
        self.scalers: Dict[str, StandardScaler] = {}
        self.position_stats = {
            'QB': ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds'],
            'RB': ['rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'fumbles'],
            'WR': ['receiving_yards', 'receiving_tds', 'targets', 'receptions', 'fumbles'],
            'TE': ['receiving_yards', 'receiving_tds', 'targets', 'receptions', 'fumbles']
        }
    
    def engineer_features(self, player_data: pd.DataFrame) -> List[PlayerFeatures]:
        """
        Main feature engineering pipeline.
        
        Args:
            player_data: DataFrame with columns [player_id, position, week, season, fantasy_points, ...]
            
        Returns:
            List of PlayerFeatures objects with engineered metrics
        """
        logger.info(f"Engineering features for {len(player_data)} player records")
        
        # Validate input data
        required_cols = ['player_id', 'position', 'week', 'season', 'fantasy_points']
        missing_cols = set(required_cols) - set(player_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        features_list = []
        
        # Process each player individually
        for player_id in player_data['player_id'].unique():
            player_df = player_data[player_data['player_id'] == player_id].copy()
            player_df = player_df.sort_values(['season', 'week'])
            
            if len(player_df) < 4:  # Need minimum games for trend analysis
                logger.warning(f"Skipping player {player_id}: insufficient data ({len(player_df)} games)")
                continue
                
            position = player_df['position'].iloc[0]
            
            # Engineer features for each game
            for idx, row in player_df.iterrows():
                # Get historical data up to current week
                hist_data = player_df[(player_df['season'] < row['season']) | 
                                    ((player_df['season'] == row['season']) & 
                                     (player_df['week'] < row['week']))]
                
                if len(hist_data) < 3:  # Need minimum history
                    continue
                
                # Take recent games for feature calculation
                recent_data = hist_data.tail(self.lookback_weeks)
                
                try:
                    features = self._calculate_player_features(
                        player_id=str(row['player_id']),
                        position=position,
                        week=int(row['week']),
                        season=int(row['season']),
                        recent_data=recent_data,
                        current_points=float(row['fantasy_points'])
                    )
                    features_list.append(features)
                    
                except Exception as e:
                    logger.error(f"Error calculating features for {player_id} week {row['week']}: {e}")
                    continue
        
        logger.info(f"Successfully engineered features for {len(features_list)} player-week combinations")
        return features_list
    
    def _calculate_player_features(self, player_id: str, position: str, 
                                 week: int, season: int, recent_data: pd.DataFrame,
                                 current_points: float) -> PlayerFeatures:
        """Calculate all engineered features for a single player-week."""
        
        fantasy_points = recent_data['fantasy_points'].values
        
        # 1. Points Per Game (PPG)
        ppg = np.mean(fantasy_points)
        
        # 2. Consistency Score (PPG/StdDev - higher is better)
        std_dev = np.std(fantasy_points)
        consistency_score = ppg / (std_dev + 0.1) if std_dev > 0 else ppg * 10
        
        # 3. Efficiency Ratio (actual vs expected based on historical average)
        if 'projected_points' in recent_data.columns:
            projected = recent_data['projected_points'].values
            efficiency_ratio = np.mean(fantasy_points / (projected + 0.1))
        else:
            # Use season average as proxy for projection
            season_avg = np.mean(fantasy_points)
            efficiency_ratio = ppg / (season_avg + 0.1) if season_avg > 0 else 1.0
        
        # 4. Momentum Score (weighted 3-week average, recent games weighted more)
        if len(fantasy_points) >= 3:
            recent_3 = fantasy_points[-3:]
            weights = np.array([0.2, 0.3, 0.5])  # More weight on recent games
            momentum_score = np.average(recent_3, weights=weights)
        else:
            momentum_score = ppg
        
        # 5. Boom/Bust Ratio
        # Boom = games > 1.5x season average, Bust = games < 0.5x season average
        season_avg = np.mean(fantasy_points)
        boom_games = np.sum(fantasy_points > 1.5 * season_avg)
        bust_games = np.sum(fantasy_points < 0.5 * season_avg)
        total_games = len(fantasy_points)
        boom_bust_ratio = (boom_games - bust_games) / total_games if total_games > 0 else 0
        
        # 6. Recent Trend (linear trend of last 4 games)
        if len(fantasy_points) >= 4:
            recent_4 = fantasy_points[-4:]
            x = np.arange(len(recent_4))
            trend_slope = np.polyfit(x, recent_4, 1)[0]  # Linear slope
            recent_trend = trend_slope
        else:
            recent_trend = 0.0
        
        return PlayerFeatures(
            player_id=player_id,
            position=position,
            week=week,
            season=season,
            ppg=float(ppg),
            consistency_score=float(consistency_score),
            efficiency_ratio=float(efficiency_ratio),
            momentum_score=float(momentum_score),
            boom_bust_ratio=float(boom_bust_ratio),
            recent_trend=float(recent_trend)
        )
    
    def features_to_dataframe(self, features_list: List[PlayerFeatures]) -> pd.DataFrame:
        """Convert list of PlayerFeatures to DataFrame for ML models."""
        data = []
        for f in features_list:
            data.append({
                'player_id': f.player_id,
                'position': f.position,
                'week': f.week,
                'season': f.season,
                'ppg': f.ppg,
                'consistency_score': f.consistency_score,
                'efficiency_ratio': f.efficiency_ratio,
                'momentum_score': f.momentum_score,
                'boom_bust_ratio': f.boom_bust_ratio,
                'recent_trend': f.recent_trend
            })
        return pd.DataFrame(data)
    
    def get_feature_matrix(self, features_df: pd.DataFrame, 
                          position: Optional[str] = None) -> np.ndarray:
        """
        Get standardized feature matrix for ML models.
        
        Args:
            features_df: DataFrame from features_to_dataframe()
            position: Filter to specific position (QB, RB, WR, TE)
            
        Returns:
            Standardized feature matrix (n_samples, 6)
        """
        if position:
            features_df = features_df[features_df['position'] == position].copy()
        
        feature_cols = ['ppg', 'consistency_score', 'efficiency_ratio', 
                       'momentum_score', 'boom_bust_ratio', 'recent_trend']
        
        X = features_df[feature_cols].values
        
        # Standardize features
        scaler_key = position if position else 'all'
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
            X_scaled = self.scalers[scaler_key].fit_transform(X)
        else:
            X_scaled = self.scalers[scaler_key].transform(X)
        
        return X_scaled
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order."""
        return ['ppg', 'consistency_score', 'efficiency_ratio', 
                'momentum_score', 'boom_bust_ratio', 'recent_trend']

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    sample_data = []
    
    for player_id in [f"player_{i}" for i in range(5)]:
        position = np.random.choice(['QB', 'RB', 'WR', 'TE'])
        for week in range(1, 18):  # 17 weeks
            fantasy_points = np.random.normal(15, 5) if position == 'QB' else np.random.normal(12, 6)
            sample_data.append({
                'player_id': player_id,
                'position': position,
                'week': week,
                'season': 2023,
                'fantasy_points': max(0, fantasy_points),  # No negative points
                'projected_points': fantasy_points * 0.9 + np.random.normal(0, 1)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Test feature engineering
    engineer = FeatureEngineer(lookback_weeks=10)
    features = engineer.engineer_features(df)
    
    print(f"Engineered {len(features)} feature sets")
    
    # Convert to DataFrame
    features_df = engineer.features_to_dataframe(features)
    print("\nFeature DataFrame shape:", features_df.shape)
    print("\nSample features:")
    print(features_df.head())
    
    # Test feature matrix generation
    X = engineer.get_feature_matrix(features_df, position='QB')
    print(f"\nQB Feature matrix shape: {X.shape}")
    print("Feature names:", engineer.get_feature_names())