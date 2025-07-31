"""
Data Aggregator - Combines all free data sources into unified datasets
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.sleeper_client import SleeperAPIClient
from data.sources.nfl_data_py_client import NFLDataPyClient
from data.sources.weather_client import WeatherClient
from data.sources.espn_public_client import ESPNPublicClient

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Combines data from all free sources into unified datasets for ML training
    
    Sources:
    - Sleeper API: Players, stats, projections
    - NFL Data Py: Play-by-play, Next Gen Stats, advanced metrics
    - Open-Meteo: Weather data
    - ESPN Public: Game info, odds, schedules
    """
    
    def __init__(self):
        """Initialize all data source clients"""
        self.sleeper = SleeperAPIClient()
        self.nfl_py = NFLDataPyClient()
        self.weather = WeatherClient()
        self.espn = ESPNPublicClient()
        
        logger.info("Data Aggregator initialized with all data sources")
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.sleeper.__aenter__()
        await self.weather.__aenter__()
        await self.espn.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.sleeper.__aexit__(exc_type, exc_val, exc_tb)
        await self.weather.__aexit__(exc_type, exc_val, exc_tb)
        await self.espn.__aexit__(exc_type, exc_val, exc_tb)
        
    async def get_player_features(self, player_id: str, season: int, week: int) -> Dict[str, Any]:
        """
        Aggregate all available data for a player for a specific week
        
        Args:
            player_id: Player ID (Sleeper format)
            season: NFL season
            week: Week number
            
        Returns:
            Dict with comprehensive player features
        """
        features = {
            'player_id': player_id,
            'season': season,
            'week': week
        }
        
        # 1. Get base player info from Sleeper
        try:
            players = await self.sleeper.get_all_players()
            if player_id in players:
                player = players[player_id]
                features.update({
                    'name': player.full_name,
                    'position': player.position,
                    'team': player.team,
                    'age': player.age,
                    'years_exp': player.years_exp
                })
        except Exception as e:
            logger.warning(f"Error getting Sleeper data: {e}")
            
        # 2. Get weekly stats from NFL Data Py
        try:
            weekly_df = self.nfl_py.import_weekly_data(season)
            player_week = weekly_df[
                (weekly_df['player_id'] == player_id) & 
                (weekly_df['week'] == week)
            ]
            
            if not player_week.empty:
                # Add statistical features
                stats_cols = [
                    'completions', 'attempts', 'passing_yards', 'passing_tds',
                    'carries', 'rushing_yards', 'rushing_tds',
                    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                    'fantasy_points', 'fantasy_points_ppr'
                ]
                
                for col in stats_cols:
                    if col in player_week.columns:
                        features[f'stats_{col}'] = player_week[col].iloc[0]
                        
                # Calculate rolling averages
                player_season = weekly_df[
                    (weekly_df['player_id'] == player_id) & 
                    (weekly_df['week'] < week)
                ].sort_values('week')
                
                if len(player_season) > 0:
                    for window in [3, 5]:
                        if len(player_season) >= window:
                            features[f'fantasy_ppr_ma{window}'] = (
                                player_season['fantasy_points_ppr'].tail(window).mean()
                            )
                            
        except Exception as e:
            logger.warning(f"Error getting NFL Data Py stats: {e}")
            
        # 3. Get Next Gen Stats if available
        position = features.get('position', '')
        if position in ['QB', 'RB', 'WR', 'TE'] and season >= 2016:
            try:
                stat_type = {
                    'QB': 'passing',
                    'RB': 'rushing',
                    'WR': 'receiving',
                    'TE': 'receiving'
                }[position]
                
                ngs_df = self.nfl_py.import_ngs_data(stat_type, season)
                player_ngs = ngs_df[
                    (ngs_df['player_id'] == player_id) & 
                    (ngs_df['week'] == week)
                ]
                
                if not player_ngs.empty:
                    ngs_cols = {
                        'passing': ['avg_time_to_throw', 'avg_completed_air_yards'],
                        'rushing': ['efficiency', 'avg_time_to_los'],
                        'receiving': ['avg_separation', 'avg_intended_air_yards']
                    }
                    
                    for col in ngs_cols.get(stat_type, []):
                        if col in player_ngs.columns:
                            features[f'ngs_{col}'] = player_ngs[col].iloc[0]
                            
            except Exception as e:
                logger.warning(f"Error getting Next Gen Stats: {e}")
                
        # 4. Get weather data for the game
        if 'team' in features and features['team']:
            try:
                # Get schedule to find game info
                schedules_df = self.nfl_py.import_schedules(season)
                game = schedules_df[
                    ((schedules_df['home_team'] == features['team']) |
                     (schedules_df['away_team'] == features['team'])) &
                    (schedules_df['week'] == week)
                ]
                
                if not game.empty:
                    home_team = game['home_team'].iloc[0]
                    game_date = pd.to_datetime(game['gameday'].iloc[0])
                    
                    weather = await self.weather.get_game_weather(
                        home_team, game_date, 13
                    )
                    
                    features.update({
                        f'weather_{k}': v 
                        for k, v in weather.items()
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting weather data: {e}")
                
        # 5. Get opponent defensive metrics
        if 'team' in features:
            try:
                features['opponent_def_rank'] = self._calculate_opponent_rank(
                    season, week, features['team'], features.get('position', '')
                )
            except Exception as e:
                logger.warning(f"Error calculating opponent rank: {e}")
                
        return features
        
    def _calculate_opponent_rank(self, season: int, week: int, team: str, 
                                position: str) -> Optional[int]:
        """Calculate opponent defensive ranking vs position"""
        try:
            # Get schedule to find opponent
            schedules_df = self.nfl_py.import_schedules(season)
            game = schedules_df[
                ((schedules_df['home_team'] == team) |
                 (schedules_df['away_team'] == team)) &
                (schedules_df['week'] == week)
            ]
            
            if game.empty:
                return None
                
            # Find opponent
            if game['home_team'].iloc[0] == team:
                opponent = game['away_team'].iloc[0]
            else:
                opponent = game['home_team'].iloc[0]
                
            # Calculate defensive ranking
            weekly_df = self.nfl_py.import_weekly_data(season)
            
            # Get fantasy points allowed to position
            allowed = weekly_df[
                (weekly_df['opponent_team'] == opponent) &
                (weekly_df['position'] == position) &
                (weekly_df['week'] < week)
            ].groupby('week')['fantasy_points_ppr'].sum().mean()
            
            # Rank among all teams (1 = allows most points)
            all_teams_allowed = weekly_df[
                (weekly_df['position'] == position) &
                (weekly_df['week'] < week)
            ].groupby('opponent_team')['fantasy_points_ppr'].sum()
            
            if len(all_teams_allowed) > 0:
                rank = (all_teams_allowed >= allowed).sum() + 1
                return rank
                
        except Exception as e:
            logger.error(f"Error calculating opponent rank: {e}")
            
        return None
        
    async def build_training_dataset(self, seasons: List[int], 
                                   positions: List[str] = ['QB', 'RB', 'WR', 'TE'],
                                   min_points: float = 5.0) -> pd.DataFrame:
        """
        Build comprehensive dataset for ML training
        
        Args:
            seasons: List of seasons to include
            positions: List of positions to include
            min_points: Minimum fantasy points to include
            
        Returns:
            DataFrame with all features for ML training
        """
        logger.info(f"Building training dataset for seasons: {seasons}")
        
        all_data = []
        
        for season in seasons:
            logger.info(f"Processing season {season}...")
            
            # Get weekly data as base
            weekly_df = self.nfl_py.import_weekly_data(season)
            
            # Filter by position and minimum points
            weekly_df = weekly_df[
                (weekly_df['position'].isin(positions)) &
                (weekly_df['fantasy_points_ppr'] >= min_points)
            ]
            
            logger.info(f"Found {len(weekly_df)} player-weeks for season {season}")
            
            # Get unique player-week combinations
            player_weeks = weekly_df[['player_id', 'week']].drop_duplicates()
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(player_weeks), batch_size):
                batch = player_weeks.iloc[i:i+batch_size]
                
                batch_features = []
                for _, row in batch.iterrows():
                    features = await self.get_player_features(
                        row['player_id'], season, row['week']
                    )
                    batch_features.append(features)
                    
                all_data.extend(batch_features)
                
                if (i + batch_size) % 500 == 0:
                    logger.info(f"Processed {i + batch_size}/{len(player_weeks)} player-weeks")
                    
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Add engineered features
        df = self._engineer_features(df)
        
        logger.info(f"Built dataset with {len(df)} samples and {len(df.columns)} features")
        
        return df
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataset
        
        Args:
            df: Raw feature DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        # Sort by player and week for rolling calculations
        df = df.sort_values(['player_id', 'season', 'week'])
        
        # 1. Consistency metrics
        for stat in ['stats_fantasy_points_ppr', 'stats_passing_yards', 
                     'stats_rushing_yards', 'stats_receiving_yards']:
            if stat in df.columns:
                df[f'{stat}_std'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=5, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
                
        # 2. Trend features
        if 'stats_fantasy_points_ppr' in df.columns:
            df['fantasy_trend'] = (
                df.groupby('player_id')['stats_fantasy_points_ppr']
                .diff()
                .fillna(0)
            )
            
        # 3. Target share (for receivers)
        if 'stats_targets' in df.columns and 'stats_receptions' in df.columns:
            df['catch_rate'] = df['stats_receptions'] / df['stats_targets'].replace(0, 1)
            
        # 4. Efficiency metrics
        if 'stats_rushing_yards' in df.columns and 'stats_carries' in df.columns:
            df['yards_per_carry'] = (
                df['stats_rushing_yards'] / df['stats_carries'].replace(0, 1)
            )
            
        if 'stats_passing_yards' in df.columns and 'stats_attempts' in df.columns:
            df['yards_per_attempt'] = (
                df['stats_passing_yards'] / df['stats_attempts'].replace(0, 1)
            )
            
        # 5. Weather impact features
        if 'weather_temperature' in df.columns:
            df['extreme_cold'] = (df['weather_temperature'] < 32).astype(int)
            df['extreme_heat'] = (df['weather_temperature'] > 85).astype(int)
            
        if 'weather_wind_speed' in df.columns:
            df['high_wind'] = (df['weather_wind_speed'] > 15).astype(int)
            
        # 6. Rest days (simplified - assumes Sunday games)
        df['rest_days'] = 7  # Default to normal rest
        
        # 7. Home/away
        df['is_home'] = df.apply(
            lambda x: 1 if x.get('team') == x.get('home_team') else 0,
            axis=1
        )
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
        
    async def get_current_week_features(self, week: int, season: int = None) -> pd.DataFrame:
        """
        Get features for all players for the current/upcoming week
        
        Args:
            week: Week number
            season: Season (default: current)
            
        Returns:
            DataFrame with features for predictions
        """
        if season is None:
            season = datetime.now().year
            
        # Get all active players
        players = await self.sleeper.get_all_players()
        
        # Filter to skill positions
        skill_players = {
            pid: p for pid, p in players.items()
            if p.position in ['QB', 'RB', 'WR', 'TE'] and p.team
        }
        
        logger.info(f"Getting features for {len(skill_players)} players for week {week}")
        
        features_list = []
        
        for player_id, player in skill_players.items():
            features = await self.get_player_features(player_id, season, week)
            features_list.append(features)
            
        df = pd.DataFrame(features_list)
        df = self._engineer_features(df)
        
        return df


# Example usage
async def test_aggregator():
    """Test the data aggregator"""
    async with DataAggregator() as aggregator:
        # Test getting player features
        print("Testing player feature aggregation...")
        
        # Example: Josh Allen for Week 1, 2024
        features = await aggregator.get_player_features('4981', 2024, 1)
        print(f"\nFeatures for Josh Allen Week 1:")
        for k, v in features.items():
            if not k.startswith('weather_'):
                print(f"  {k}: {v}")
                
        # Test building small training dataset
        print("\nBuilding small training dataset...")
        df = await aggregator.build_training_dataset(
            seasons=[2023],
            positions=['QB'],
            min_points=10.0
        )
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")
        
        if not df.empty:
            print(f"\nTop QBs by average points:")
            top_qbs = df.groupby('name')['stats_fantasy_points_ppr'].agg(['mean', 'count'])
            top_qbs = top_qbs[top_qbs['count'] >= 5].sort_values('mean', ascending=False)
            print(top_qbs.head())


if __name__ == "__main__":
    asyncio.run(test_aggregator())