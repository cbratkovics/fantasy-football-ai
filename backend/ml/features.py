"""
Advanced Feature Engineering for Fantasy Football ML
Extracts 20+ features for GMM clustering and neural network predictions
Includes performance metrics, trends, matchup analysis, and advanced statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PlayerFeatures:
    """Container for all engineered features"""
    player_id: str
    season: int
    week: int
    
    # Basic Performance (5 features)
    points_per_game: float
    season_total_points: float
    games_played: int
    position_rank: int
    team_offensive_rank: int
    
    # Recent Form (5 features)
    last_3_games_avg: float
    last_5_games_avg: float
    momentum_score: float  # Trend in recent games
    form_consistency: float  # Std dev of recent games
    hot_streak_indicator: int  # Binary: on hot streak?
    
    # Variance & Risk (4 features)
    season_std_dev: float
    coefficient_of_variation: float
    boom_rate: float  # % games > 1.5x average
    bust_rate: float  # % games < 0.5x average
    
    # Matchup Difficulty (4 features)
    opponent_defensive_rank: int
    opponent_points_allowed_vs_position: float
    historical_vs_opponent: float  # Avg points in past matchups
    vegas_implied_team_total: float
    
    # Advanced Metrics (5 features)
    target_share: float  # For WR/TE
    red_zone_share: float
    snap_percentage: float
    yards_per_opportunity: float
    efficiency_rating: float
    
    # Context Features (3 features)
    is_home: int
    days_rest: int
    weather_impact_score: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for ML models"""
        return np.array([
            self.points_per_game,
            self.season_total_points,
            self.games_played,
            self.position_rank,
            self.team_offensive_rank,
            self.last_3_games_avg,
            self.last_5_games_avg,
            self.momentum_score,
            self.form_consistency,
            self.hot_streak_indicator,
            self.season_std_dev,
            self.coefficient_of_variation,
            self.boom_rate,
            self.bust_rate,
            self.opponent_defensive_rank,
            self.opponent_points_allowed_vs_position,
            self.historical_vs_opponent,
            self.vegas_implied_team_total,
            self.target_share,
            self.red_zone_share,
            self.snap_percentage,
            self.yards_per_opportunity,
            self.efficiency_rating,
            self.is_home,
            self.days_rest,
            self.weather_impact_score
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretability"""
        return [
            'points_per_game', 'season_total_points', 'games_played',
            'position_rank', 'team_offensive_rank', 'last_3_games_avg',
            'last_5_games_avg', 'momentum_score', 'form_consistency',
            'hot_streak_indicator', 'season_std_dev', 'coefficient_of_variation',
            'boom_rate', 'bust_rate', 'opponent_defensive_rank',
            'opponent_points_allowed_vs_position', 'historical_vs_opponent',
            'vegas_implied_team_total', 'target_share', 'red_zone_share',
            'snap_percentage', 'yards_per_opportunity', 'efficiency_rating',
            'is_home', 'days_rest', 'weather_impact_score'
        ]


class FeatureEngineer:
    """
    Sophisticated feature engineering for fantasy football ML
    Extracts predictive signals from historical data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_features(
        self,
        player_stats: pd.DataFrame,
        team_stats: pd.DataFrame,
        defensive_stats: pd.DataFrame,
        vegas_lines: Optional[pd.DataFrame] = None,
        weather_data: Optional[pd.DataFrame] = None
    ) -> PlayerFeatures:
        """
        Extract all features for a player in a given week
        
        Args:
            player_stats: Historical stats for the player
            team_stats: Team offensive rankings
            defensive_stats: Defensive rankings by position
            vegas_lines: Betting lines for implied totals
            weather_data: Weather conditions
            
        Returns:
            PlayerFeatures object with all engineered features
        """
        # Get current week data
        current_week = player_stats.iloc[-1]
        player_id = current_week['player_id']
        season = current_week['season']
        week = current_week['week']
        position = current_week['position']
        
        # Basic performance features
        basic_features = self._extract_basic_performance(player_stats)
        
        # Recent form features
        form_features = self._extract_recent_form(player_stats)
        
        # Variance and risk features
        risk_features = self._extract_variance_metrics(player_stats)
        
        # Matchup difficulty features
        matchup_features = self._extract_matchup_difficulty(
            current_week, 
            defensive_stats,
            player_stats
        )
        
        # Advanced metrics
        advanced_features = self._extract_advanced_metrics(player_stats)
        
        # Context features
        context_features = self._extract_context_features(
            current_week,
            vegas_lines,
            weather_data
        )
        
        # Combine all features
        return PlayerFeatures(
            player_id=player_id,
            season=season,
            week=week,
            **basic_features,
            **form_features,
            **risk_features,
            **matchup_features,
            **advanced_features,
            **context_features
        )
    
    def _extract_basic_performance(
        self, 
        player_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract basic performance metrics"""
        # Filter to current season
        current_season = player_stats['season'].max()
        season_stats = player_stats[player_stats['season'] == current_season]
        
        points_per_game = season_stats['fantasy_points'].mean()
        season_total = season_stats['fantasy_points'].sum()
        games_played = len(season_stats)
        
        # Calculate position rank (would need full dataset)
        position = season_stats.iloc[0]['position']
        position_rank = self._calculate_position_rank(
            points_per_game, 
            position
        )
        
        # Team offensive rank (simplified - would use team data)
        team = season_stats.iloc[0]['team']
        team_offensive_rank = self._get_team_offensive_rank(team)
        
        return {
            'points_per_game': points_per_game,
            'season_total_points': season_total,
            'games_played': games_played,
            'position_rank': position_rank,
            'team_offensive_rank': team_offensive_rank
        }
    
    def _extract_recent_form(
        self, 
        player_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract recent performance and momentum features"""
        # Get last N games
        recent_games = player_stats.tail(5)
        last_3_games = player_stats.tail(3)
        
        last_3_avg = last_3_games['fantasy_points'].mean()
        last_5_avg = recent_games['fantasy_points'].mean()
        
        # Calculate momentum (trend)
        if len(recent_games) >= 3:
            x = np.arange(len(recent_games))
            y = recent_games['fantasy_points'].values
            slope, _, _, _, _ = stats.linregress(x, y)
            momentum_score = slope
        else:
            momentum_score = 0.0
        
        # Form consistency (lower is better)
        form_consistency = last_3_games['fantasy_points'].std()
        
        # Hot streak indicator
        season_avg = player_stats['fantasy_points'].mean()
        hot_streak = int(
            (last_3_avg > season_avg * 1.2) and 
            (momentum_score > 0)
        )
        
        return {
            'last_3_games_avg': last_3_avg,
            'last_5_games_avg': last_5_avg,
            'momentum_score': momentum_score,
            'form_consistency': form_consistency,
            'hot_streak_indicator': hot_streak
        }
    
    def _extract_variance_metrics(
        self, 
        player_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract variance and risk metrics"""
        points = player_stats['fantasy_points']
        mean_points = points.mean()
        std_dev = points.std()
        
        # Coefficient of variation (normalized volatility)
        cv = std_dev / mean_points if mean_points > 0 else 0
        
        # Boom and bust rates
        boom_threshold = mean_points * 1.5
        bust_threshold = mean_points * 0.5
        
        boom_rate = (points > boom_threshold).mean()
        bust_rate = (points < bust_threshold).mean()
        
        return {
            'season_std_dev': std_dev,
            'coefficient_of_variation': cv,
            'boom_rate': boom_rate,
            'bust_rate': bust_rate
        }
    
    def _extract_matchup_difficulty(
        self,
        current_week: pd.Series,
        defensive_stats: pd.DataFrame,
        player_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract matchup-based features"""
        opponent = current_week['opponent']
        position = current_week['position']
        
        # Get opponent defensive rank
        opp_def_rank = self._get_defensive_rank(
            opponent, 
            position,
            defensive_stats
        )
        
        # Points allowed to position
        points_allowed = self._get_points_allowed_to_position(
            opponent,
            position,
            defensive_stats
        )
        
        # Historical performance vs this opponent
        historical_vs_opp = self._get_historical_vs_opponent(
            player_stats,
            opponent
        )
        
        # Vegas implied total (simplified)
        vegas_total = 24.5  # Default, would use actual data
        
        return {
            'opponent_defensive_rank': opp_def_rank,
            'opponent_points_allowed_vs_position': points_allowed,
            'historical_vs_opponent': historical_vs_opp,
            'vegas_implied_team_total': vegas_total
        }
    
    def _extract_advanced_metrics(
        self, 
        player_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract advanced efficiency metrics"""
        position = player_stats.iloc[0]['position']
        
        # Default values
        target_share = 0.0
        red_zone_share = 0.0
        snap_percentage = 0.85  # Default
        yards_per_opportunity = 0.0
        efficiency_rating = 0.0
        
        # Position-specific metrics
        if position in ['WR', 'TE']:
            # Calculate from stats if available
            total_targets = player_stats['targets'].sum() if 'targets' in player_stats else 0
            team_targets = 450  # Approximate team total
            target_share = total_targets / team_targets if team_targets > 0 else 0
            
            # Red zone targets
            rz_targets = player_stats['red_zone_targets'].sum() if 'red_zone_targets' in player_stats else 0
            team_rz_targets = 80  # Approximate
            red_zone_share = rz_targets / team_rz_targets if team_rz_targets > 0 else 0
            
            # Yards per target
            total_yards = player_stats['receiving_yards'].sum() if 'receiving_yards' in player_stats else 0
            yards_per_opportunity = total_yards / total_targets if total_targets > 0 else 0
        
        elif position == 'RB':
            # Opportunity share
            total_touches = player_stats['carries'].sum() if 'carries' in player_stats else 0
            total_touches += player_stats['targets'].sum() if 'targets' in player_stats else 0
            
            # Yards per touch
            total_yards = player_stats['rushing_yards'].sum() if 'rushing_yards' in player_stats else 0
            total_yards += player_stats['receiving_yards'].sum() if 'receiving_yards' in player_stats else 0
            yards_per_opportunity = total_yards / total_touches if total_touches > 0 else 0
        
        # Overall efficiency rating (fantasy points per snap)
        avg_points = player_stats['fantasy_points'].mean()
        efficiency_rating = avg_points / snap_percentage if snap_percentage > 0 else 0
        
        return {
            'target_share': target_share,
            'red_zone_share': red_zone_share,
            'snap_percentage': snap_percentage,
            'yards_per_opportunity': yards_per_opportunity,
            'efficiency_rating': efficiency_rating
        }
    
    def _extract_context_features(
        self,
        current_week: pd.Series,
        vegas_lines: Optional[pd.DataFrame],
        weather_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extract game context features"""
        is_home = int(current_week.get('is_home', 0))
        
        # Calculate days of rest (simplified)
        days_rest = 7  # Default for weekly games
        
        # Weather impact score
        weather_impact = 0.0
        if weather_data is not None:
            # Would calculate based on wind, precipitation, temperature
            weather_impact = self._calculate_weather_impact(
                weather_data,
                current_week['game_id']
            )
        
        return {
            'is_home': is_home,
            'days_rest': days_rest,
            'weather_impact_score': weather_impact
        }
    
    # Helper methods (simplified implementations)
    def _calculate_position_rank(self, ppg: float, position: str) -> int:
        """Calculate position rank based on PPG"""
        # Simplified - would use actual rankings
        thresholds = {
            'QB': [25, 22, 20, 18, 16],
            'RB': [18, 15, 12, 10, 8],
            'WR': [16, 14, 12, 10, 8],
            'TE': [12, 10, 8, 6, 4]
        }
        
        pos_thresholds = thresholds.get(position, [15, 12, 10, 8, 6])
        rank = len(pos_thresholds) + 1
        
        for i, threshold in enumerate(pos_thresholds):
            if ppg >= threshold:
                rank = i + 1
                break
        
        return rank
    
    def _get_team_offensive_rank(self, team: str) -> int:
        """Get team offensive ranking"""
        # Simplified - would use actual data
        team_ranks = {
            'KC': 1, 'BUF': 2, 'MIA': 3, 'PHI': 4, 'DAL': 5,
            'SF': 6, 'CIN': 7, 'JAX': 8, 'SEA': 9, 'LAC': 10
        }
        return team_ranks.get(team, 16)
    
    def _get_defensive_rank(
        self, 
        opponent: str, 
        position: str,
        defensive_stats: pd.DataFrame
    ) -> int:
        """Get opponent defensive ranking vs position"""
        # Simplified implementation
        return np.random.randint(1, 33)
    
    def _get_points_allowed_to_position(
        self,
        opponent: str,
        position: str,
        defensive_stats: pd.DataFrame
    ) -> float:
        """Get average points allowed to position"""
        # Simplified - would use actual data
        avg_allowed = {
            'QB': 18.5,
            'RB': 22.0,
            'WR': 28.5,
            'TE': 10.5
        }
        return avg_allowed.get(position, 15.0)
    
    def _get_historical_vs_opponent(
        self,
        player_stats: pd.DataFrame,
        opponent: str
    ) -> float:
        """Get historical performance vs opponent"""
        vs_opponent = player_stats[player_stats['opponent'] == opponent]
        if len(vs_opponent) > 0:
            return vs_opponent['fantasy_points'].mean()
        return player_stats['fantasy_points'].mean()  # Default to average
    
    def _calculate_weather_impact(
        self,
        weather_data: pd.DataFrame,
        game_id: str
    ) -> float:
        """Calculate weather impact on fantasy performance"""
        # Simplified - would use actual weather data
        return 0.0
    
    def fit_transform(
        self, 
        features_list: List[PlayerFeatures]
    ) -> np.ndarray:
        """Fit scaler and transform features for ML"""
        # Convert to matrix
        feature_matrix = np.array([f.to_vector() for f in features_list])
        
        # Fit and transform
        scaled_features = self.scaler.fit_transform(feature_matrix)
        self._is_fitted = True
        
        return scaled_features
    
    def transform(
        self, 
        features_list: List[PlayerFeatures]
    ) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        feature_matrix = np.array([f.to_vector() for f in features_list])
        return self.scaler.transform(feature_matrix)
    
    def get_position_features(self, position: str) -> List[str]:
        """Get relevant features for a specific position"""
        # Base features for all positions
        base_features = [
            'season', 'week', 'age', 'years_exp',
            'fantasy_points_ppr_lag1', 'fantasy_points_ppr_lag2',
            'fantasy_points_ppr_rolling_avg'
        ]
        
        # Position-specific features from Sleeper stats
        position_features = {
            'QB': base_features + [
                'pass_att', 'pass_cmp', 'pass_yd', 'pass_td', 'pass_int',
                'rush_att', 'rush_yd', 'rush_td', 'pass_cmp_pct',
                'pts_ppr_lag1', 'pts_ppr_rolling_avg'
            ],
            'RB': base_features + [
                'rush_att', 'rush_yd', 'rush_td', 'rec', 'rec_yd', 'rec_td',
                'fum_lost', 'rush_ypc', 'rec_tgt',
                'pts_ppr_lag1', 'pts_ppr_rolling_avg'
            ],
            'WR': base_features + [
                'rec', 'rec_yd', 'rec_td', 'rec_tgt', 'rush_att', 'rush_yd',
                'rec_ypr', 'rec_yac', 'st_snp',
                'pts_ppr_lag1', 'pts_ppr_rolling_avg'
            ],
            'TE': base_features + [
                'rec', 'rec_yd', 'rec_td', 'rec_tgt', 'rec_ypr',
                'st_snp', 'off_snp',
                'pts_ppr_lag1', 'pts_ppr_rolling_avg'
            ],
            'K': base_features + [
                'fgm', 'fga', 'xpm', 'xpa', 'fgm_pct',
                'fgm_0_19', 'fgm_20_29', 'fgm_30_39', 'fgm_40_49', 'fgm_50p',
                'pts_ppr_lag1', 'pts_ppr_rolling_avg'
            ]
        }
        
        return position_features.get(position, base_features)


# Example usage
def example_usage():
    """Demonstrate feature engineering"""
    # Create sample data
    player_stats = pd.DataFrame({
        'player_id': ['123'] * 10,
        'season': [2024] * 10,
        'week': list(range(1, 11)),
        'position': ['WR'] * 10,
        'team': ['KC'] * 10,
        'opponent': ['DEN', 'JAX', 'CHI', 'NYJ', 'MIN', 
                     'BUF', 'SF', 'LV', 'GB', 'CIN'],
        'fantasy_points': [15.2, 22.5, 8.4, 18.9, 25.6,
                          12.3, 19.8, 28.4, 14.5, 21.2],
        'is_home': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    })
    
    # Empty dataframes for demo
    team_stats = pd.DataFrame()
    defensive_stats = pd.DataFrame()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Extract features
    features = engineer.extract_features(
        player_stats,
        team_stats,
        defensive_stats
    )
    
    print(f"Player: {features.player_id}")
    print(f"Week: {features.week}")
    print(f"PPG: {features.points_per_game:.2f}")
    print(f"Recent form: {features.last_3_games_avg:.2f}")
    print(f"Momentum: {features.momentum_score:.2f}")
    print(f"Boom rate: {features.boom_rate:.2%}")
    print(f"Feature vector shape: {features.to_vector().shape}")


if __name__ == "__main__":
    example_usage()