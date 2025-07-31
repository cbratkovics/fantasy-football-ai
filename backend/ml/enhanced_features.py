"""
Enhanced Feature Engineering System
Incorporates all advanced features for improved ML accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """
    Advanced feature engineering with all requested enhancements
    """
    
    def __init__(self):
        self.feature_groups = {
            # Basic stats
            'basic_stats': [
                'pass_yards', 'pass_tds', 'interceptions', 'rush_yards', 
                'rush_tds', 'receptions', 'rec_yards', 'rec_tds'
            ],
            
            # Advanced efficiency metrics
            'efficiency': [
                'yards_per_attempt', 'yards_per_carry', 'yards_per_reception',
                'touchdown_rate', 'target_share', 'air_yards_share',
                'red_zone_touches', 'red_zone_efficiency'
            ],
            
            # Combine metrics
            'combine': [
                'forty_yard', 'bench_press', 'vertical_jump', 'broad_jump',
                'three_cone', 'shuttle', 'height_inches', 'weight_lbs',
                'bmi', 'speed_score', 'burst_score', 'agility_score'
            ],
            
            # Team context
            'team_context': [
                'offensive_line_rank', 'team_pass_rate', 'team_total_plays',
                'team_scoring_rate', 'offensive_coordinator_score',
                'quarterback_rating', 'offensive_pace'
            ],
            
            # Opponent features
            'opponent': [
                'opp_def_rank', 'opp_pass_def_rank', 'opp_rush_def_rank',
                'opp_fantasy_points_allowed', 'opp_dvoa', 'opp_pressure_rate',
                'historical_vs_opponent'
            ],
            
            # Weather features
            'weather': [
                'temperature', 'wind_speed', 'precipitation', 'humidity',
                'dome_game', 'weather_impact_score'
            ],
            
            # Injury features
            'injury': [
                'injury_status_encoded', 'games_missed_season', 
                'games_missed_career', 'injury_prone_score',
                'days_since_injury'
            ],
            
            # Momentum features
            'momentum': [
                'last_game_points', 'avg_last_3_games', 'avg_last_5_games',
                'trend_direction', 'hot_streak', 'consistency_score',
                'boom_bust_factor'
            ],
            
            # College features (for rookies)
            'college': [
                'college_dominator_rating', 'college_ypc', 'college_td_rate',
                'college_market_share', 'breakout_age', 'draft_capital'
            ],
            
            # Time-based features
            'temporal': [
                'week_of_season', 'month', 'is_primetime', 'is_division_game',
                'rest_days', 'timezone_change'
            ]
        }
        
        self.scalers = {}
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from raw data
        """
        logger.info("Starting enhanced feature engineering...")
        
        # Create copy to avoid modifying original
        data = df.copy()
        
        # 1. Basic transformations
        data = self._create_basic_features(data)
        
        # 2. Efficiency metrics
        data = self._create_efficiency_metrics(data)
        
        # 3. Combine-based features
        data = self._create_combine_features(data)
        
        # 4. Team context features
        data = self._create_team_context_features(data)
        
        # 5. Opponent strength features
        data = self._create_opponent_features(data)
        
        # 6. Weather impact features
        data = self._create_weather_features(data)
        
        # 7. Injury-related features
        data = self._create_injury_features(data)
        
        # 8. Momentum and trend features
        data = self._create_momentum_features(data)
        
        # 9. College performance features (for rookies/young players)
        data = self._create_college_features(data)
        
        # 10. Temporal features
        data = self._create_temporal_features(data)
        
        # 11. Interaction features
        data = self._create_interaction_features(data)
        
        # 12. Position-specific features
        data = self._create_position_specific_features(data)
        
        logger.info(f"Feature engineering complete. Total features: {len(data.columns)}")
        
        return data
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic statistical features"""
        # Yards per attempt/carry/reception
        df['yards_per_attempt'] = df['pass_yards'] / df['pass_attempts'].replace(0, 1)
        df['yards_per_carry'] = df['rush_yards'] / df['rush_attempts'].replace(0, 1)
        df['yards_per_reception'] = df['rec_yards'] / df['receptions'].replace(0, 1)
        
        # Touchdown rates
        df['pass_td_rate'] = df['pass_tds'] / df['pass_attempts'].replace(0, 1)
        df['rush_td_rate'] = df['rush_tds'] / df['rush_attempts'].replace(0, 1)
        df['rec_td_rate'] = df['rec_tds'] / df['receptions'].replace(0, 1)
        
        # Total opportunities
        df['total_touches'] = df['rush_attempts'] + df['receptions']
        df['total_opportunities'] = df['pass_attempts'] + df['total_touches']
        
        return df
    
    def _create_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced efficiency metrics"""
        # EPA (Expected Points Added) - simplified version
        df['epa_per_play'] = (
            (df['pass_yards'] * 0.1 + df['pass_tds'] * 6 - df['interceptions'] * 2) / 
            df['pass_attempts'].replace(0, 1)
        )
        
        # Success rate (4+ yards on 1st down, 50% of remaining on 2nd, all on 3rd)
        # Simplified approximation
        df['success_rate'] = df['first_downs'] / df['total_opportunities'].replace(0, 1)
        
        # RACR (Receiver Air Conversion Ratio)
        df['racr'] = df['rec_yards'] / df['air_yards'].replace(0, 1)
        
        # Opportunity share
        df['opportunity_share'] = df['total_opportunities'] / df['team_total_plays'].replace(0, 1)
        
        # Red zone efficiency
        df['rz_touch_rate'] = df['red_zone_touches'] / df['total_touches'].replace(0, 1)
        df['rz_td_conversion'] = df['red_zone_tds'] / df['red_zone_touches'].replace(0, 1)
        
        return df
    
    def _create_combine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from NFL Combine data"""
        # BMI calculation
        df['bmi'] = (df['weight_lbs'] / (df['height_inches'] ** 2)) * 703
        
        # Speed Score (for RBs) = (Weight * 200) / (40-time ^ 4)
        df['speed_score'] = np.where(
            df['position'] == 'RB',
            (df['weight_lbs'] * 200) / (df['forty_yard'] ** 4),
            0
        )
        
        # Burst Score = Vertical Jump + Broad Jump
        df['burst_score'] = df['vertical_jump'] + (df['broad_jump'] / 12)
        
        # Agility Score = 3-Cone + 20-Yard Shuttle
        df['agility_score'] = 2 / (1/df['three_cone'] + 1/df['shuttle'])
        
        # Height-Adjusted Speed Score
        df['has_score'] = df['speed_score'] * (df['height_inches'] / 70)
        
        # Weight-adjusted athleticism
        df['athleticism_score'] = (
            df['burst_score'] * 0.3 +
            (100 / df['forty_yard']) * 0.4 +
            (100 / df['agility_score']) * 0.3
        ) * (df['weight_lbs'] / 220)
        
        return df
    
    def _create_team_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team and offensive system features"""
        # Offensive line composite score
        df['ol_composite'] = 33 - df['offensive_line_rank']  # Higher is better
        
        # Adjusted line yards (rushing efficiency metric)
        df['adjusted_line_yards'] = df['team_rush_yards'] / df['team_rush_attempts'].replace(0, 1)
        
        # Pass protection rating
        df['pass_protection_rating'] = 100 - (df['sacks_allowed'] / df['pass_attempts'].replace(0, 1) * 100)
        
        # Offensive pace (plays per minute of possession)
        df['offensive_pace'] = df['team_total_plays'] / df['time_of_possession'].replace(0, 1)
        
        # Play action rate and efficiency
        df['play_action_rate'] = df['play_action_passes'] / df['pass_attempts'].replace(0, 1)
        df['play_action_efficiency'] = df['play_action_yards'] / df['play_action_passes'].replace(0, 1)
        
        # Offensive coordinator historical performance
        df['oc_historical_ppg'] = df.groupby('offensive_coordinator')['team_points'].transform('mean')
        
        return df
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent and matchup features"""
        # Defensive DVOA by position
        position_dvoa_map = {
            'QB': 'pass_dvoa',
            'RB': 'rush_dvoa',
            'WR': 'pass_dvoa',
            'TE': 'pass_dvoa'
        }
        
        df['opponent_dvoa'] = df.apply(
            lambda row: row[f"opp_{position_dvoa_map.get(row['position'], 'total_dvoa')}"],
            axis=1
        )
        
        # Historical performance vs opponent
        df['avg_vs_opponent'] = df.groupby(['player_id', 'opponent'])['fantasy_points'].transform('mean')
        df['games_vs_opponent'] = df.groupby(['player_id', 'opponent'])['fantasy_points'].transform('count')
        
        # Opponent pace factor
        df['opponent_pace'] = df['opp_plays_per_game'] / 65  # League average normalization
        
        # Specific defensive weaknesses
        df['opp_allows_big_plays'] = df['opp_plays_20plus_allowed'] / df['opp_plays_faced'].replace(0, 1)
        
        # Blitz rate impact
        df['blitz_rate_impact'] = df['opp_blitz_rate'] * df['qb_rating_vs_blitz']
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather impact features"""
        # Weather severity score
        df['weather_severity'] = (
            (np.abs(df['temperature'] - 65) / 35) * 0.3 +  # Temperature impact
            (df['wind_speed'] / 25) * 0.4 +  # Wind impact
            (df['precipitation'] / 2) * 0.3  # Precipitation impact
        ).clip(0, 1)
        
        # Position-specific weather impact
        weather_impact_by_position = {
            'QB': lambda row: row['weather_severity'] * 1.2 if row['wind_speed'] > 15 else row['weather_severity'],
            'RB': lambda row: row['weather_severity'] * 0.8,  # Less affected
            'WR': lambda row: row['weather_severity'] * 1.1,
            'TE': lambda row: row['weather_severity'] * 0.9
        }
        
        df['position_weather_impact'] = df.apply(
            lambda row: weather_impact_by_position.get(row['position'], lambda x: x['weather_severity'])(row),
            axis=1
        )
        
        # Dome advantage
        df['dome_advantage'] = np.where(df['dome_game'], 0.05, 0)
        
        # Cold weather specialist (based on historical performance)
        df['cold_weather_rating'] = df.groupby('player_id').apply(
            lambda x: x[x['temperature'] < 40]['fantasy_points'].mean() / 
                     x['fantasy_points'].mean() if len(x[x['temperature'] < 40]) > 3 else 1
        ).reindex(df.index, fill_value=1)
        
        return df
    
    def _create_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create injury-related features"""
        # Encode injury status
        injury_encoding = {
            'Healthy': 0,
            'Probable': 0.1,
            'Questionable': 0.3,
            'Doubtful': 0.7,
            'Out': 1.0
        }
        df['injury_impact'] = df['injury_status'].map(injury_encoding).fillna(0)
        
        # Injury history metrics
        df['injury_frequency'] = df.groupby('player_id')['games_missed'].transform('sum') / df['career_games']
        df['recent_injury_count'] = df.groupby('player_id')['injury_status'].transform(
            lambda x: (x != 'Healthy').rolling(window=10, min_periods=1).sum()
        )
        
        # Recovery time impact
        df['games_since_injury'] = df.groupby('player_id')['games_missed'].transform(
            lambda x: x.expanding().apply(lambda y: len(y) - np.argmax(y[::-1] > 0) if any(y > 0) else len(y))
        )
        
        # Position-specific injury risk
        injury_risk_by_position = {
            'RB': 1.5,  # Highest risk
            'WR': 1.2,
            'TE': 1.1,
            'QB': 1.0
        }
        df['position_injury_risk'] = df['position'].map(injury_risk_by_position) * df['injury_frequency']
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features"""
        # Rolling averages
        for window in [3, 5, 8]:
            df[f'avg_points_last_{window}'] = df.groupby('player_id')['fantasy_points'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        # Trend calculation (linear regression slope)
        def calculate_trend(series, window=5):
            return series.rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
            )
        
        df['point_trend'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: calculate_trend(x.shift(1))
        )
        
        # Hot/cold streaks
        df['consecutive_good_games'] = df.groupby('player_id').apply(
            lambda x: (x['fantasy_points'] > x['fantasy_points'].rolling(10).mean()).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Volatility/consistency
        df['point_volatility'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).std()
        )
        
        # Boom/bust potential
        df['boom_rate'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: (x > x.quantile(0.75)).rolling(window=10, min_periods=1).mean()
        )
        df['bust_rate'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: (x < x.quantile(0.25)).rolling(window=10, min_periods=1).mean()
        )
        
        return df
    
    def _create_college_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create college performance features for young players"""
        # Only applicable for players in first 3 years
        df['is_young_player'] = df['years_experience'] <= 3
        
        # Dominator rating (% of team's yards and TDs)
        df['college_dominator'] = np.where(
            df['is_young_player'],
            (df['college_yard_share'] + df['college_td_share']) / 2,
            0
        )
        
        # Breakout age (age when first dominated in college)
        df['breakout_age'] = np.where(
            df['is_young_player'],
            df['college_breakout_age'],
            df['age']  # Use current age for veterans
        )
        
        # Draft capital (inverse of draft position)
        df['draft_capital'] = np.where(
            df['draft_position'] > 0,
            300 - df['draft_position'],  # Higher pick = higher value
            150  # Undrafted free agents
        )
        
        # College production score
        df['college_production_score'] = (
            df['college_yards_per_game'] * 0.5 +
            df['college_td_per_game'] * 10 +
            df['college_dominator'] * 20
        )
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        # Convert game date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Basic temporal features
        df['week_of_season'] = df['week']
        df['month'] = df['game_date'].dt.month
        df['day_of_week'] = df['game_date'].dt.dayofweek
        
        # Game timing features
        df['is_primetime'] = df['game_time'].str.contains('8:|7:30', na=False)
        df['is_early_game'] = df['game_time'].str.contains('1:00', na=False)
        df['is_late_game'] = df['game_time'].str.contains('4:', na=False)
        
        # Rest days calculation
        df['days_since_last_game'] = df.groupby('player_id')['game_date'].diff().dt.days.fillna(7)
        df['is_short_week'] = df['days_since_last_game'] < 6
        df['is_long_rest'] = df['days_since_last_game'] > 10
        
        # Timezone impact
        df['timezone_change'] = np.abs(df['home_timezone'] - df['game_timezone'])
        df['west_coast_to_east'] = (df['home_timezone'] == 'PST') & (df['game_timezone'] == 'EST')
        
        # Season progression
        df['season_fatigue_factor'] = df['week'] / 18  # Increases as season progresses
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different feature groups"""
        # Matchup interactions
        df['matchup_advantage'] = df['team_total_offense_rank'] / df['opp_def_rank']
        df['pace_matchup'] = df['offensive_pace'] * df['opponent_pace']
        
        # Weather-injury interaction
        df['weather_injury_risk'] = df['weather_severity'] * df['injury_impact']
        
        # Momentum-matchup interaction
        df['hot_vs_weak'] = df['point_trend'] * (33 - df['opp_def_rank']) / 32
        
        # Volume-efficiency interaction
        df['efficient_volume'] = df['total_opportunities'] * df['yards_per_touch']
        
        # O-line and running back interaction
        df['rb_oline_synergy'] = np.where(
            df['position'] == 'RB',
            df['ol_composite'] * df['yards_per_carry'],
            0
        )
        
        # Stack correlation (QB-WR/TE on same team)
        df['qb_rating_impact'] = df['team_qb_rating'] / 100 * df['target_share']
        
        return df
    
    def _create_position_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to each position"""
        # QB features
        qb_mask = df['position'] == 'QB'
        df.loc[qb_mask, 'pocket_time'] = df.loc[qb_mask, 'time_to_throw']
        df.loc[qb_mask, 'pressure_rate_faced'] = df.loc[qb_mask, 'pressures'] / df.loc[qb_mask, 'dropbacks']
        df.loc[qb_mask, 'deep_ball_rate'] = df.loc[qb_mask, 'deep_attempts'] / df.loc[qb_mask, 'pass_attempts']
        
        # RB features
        rb_mask = df['position'] == 'RB'
        df.loc[rb_mask, 'yards_before_contact'] = df.loc[rb_mask, 'ybc_per_attempt']
        df.loc[rb_mask, 'broken_tackle_rate'] = df.loc[rb_mask, 'broken_tackles'] / df.loc[rb_mask, 'touches']
        df.loc[rb_mask, 'goal_line_share'] = df.loc[rb_mask, 'goal_line_carries'] / df.loc[rb_mask, 'team_goal_line_carries']
        
        # WR features
        wr_mask = df['position'] == 'WR'
        df.loc[wr_mask, 'separation_score'] = df.loc[wr_mask, 'avg_separation']
        df.loc[wr_mask, 'contested_catch_rate'] = df.loc[wr_mask, 'contested_catches'] / df.loc[wr_mask, 'contested_targets']
        df.loc[wr_mask, 'slot_rate'] = df.loc[wr_mask, 'slot_snaps'] / df.loc[wr_mask, 'total_snaps']
        
        # TE features
        te_mask = df['position'] == 'TE'
        df.loc[te_mask, 'blocking_snaps_pct'] = df.loc[te_mask, 'blocking_snaps'] / df.loc[te_mask, 'total_snaps']
        df.loc[te_mask, 'route_participation'] = df.loc[te_mask, 'routes_run'] / df.loc[te_mask, 'pass_plays']
        
        return df
    
    def select_features(self, df: pd.DataFrame, target: str, 
                       method: str = 'mutual_info', k: int = 50) -> List[str]:
        """
        Select most important features using various methods
        """
        X = df.drop(columns=[target])
        y = df[target]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info(f"Top 10 features by {method}:")
        logger.info(feature_scores.head(10))
        
        return selected_features
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for non-linear relationships
        """
        poly_data = self.poly_features.fit_transform(df[features])
        poly_feature_names = self.poly_features.get_feature_names_out(features)
        
        poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
        
        # Merge with original dataframe
        result = pd.concat([df, poly_df], axis=1)
        
        return result