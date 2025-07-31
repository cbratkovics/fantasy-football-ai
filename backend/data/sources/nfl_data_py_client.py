"""
NFL Data Py Client - Comprehensive NFL Data from Free Sources
MIT License - Commercial Use Allowed
"""

import os
import redis
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import nfl_data_py as nfl
from functools import lru_cache

logger = logging.getLogger(__name__)


class NFLDataPyClient:
    """
    Client for nfl_data_py - comprehensive NFL data from multiple free sources
    Includes play-by-play, player stats, Next Gen Stats, and more
    """
    
    def __init__(self):
        """Initialize with Redis caching"""
        # Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Cache TTLs
        self.HISTORICAL_TTL = 7 * 24 * 60 * 60  # 7 days for historical data
        self.CURRENT_TTL = 60 * 60  # 1 hour for current season data
        
        logger.info("NFLDataPy client initialized")
        
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate cache key from data type and parameters"""
        params = "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return f"nfl_data_py:{data_type}:{params}"
        
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from Redis cache"""
        try:
            cached_json = self.redis_client.get(cache_key)
            if cached_json:
                return pd.read_json(cached_json)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None
        
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame, ttl: int):
        """Save DataFrame to Redis cache"""
        try:
            json_data = df.to_json()
            self.redis_client.setex(cache_key, ttl, json_data)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
            
    def import_pbp_data(self, years: Union[int, List[int]], 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Import play-by-play data (1999-present)
        
        Args:
            years: Single year or list of years
            columns: Specific columns to include (None for all)
            
        Returns:
            DataFrame with play-by-play data
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("pbp", years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            logger.info(f"Retrieved play-by-play data from cache for years: {years}")
            return cached_df
            
        try:
            # Fetch from nfl_data_py
            logger.info(f"Fetching play-by-play data for years: {years}")
            df = nfl.import_pbp_data(years, columns=columns)
            
            # Cache based on whether it's current season
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            logger.info(f"Retrieved {len(df)} play-by-play records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {e}")
            return pd.DataFrame()
            
    def import_weekly_data(self, years: Union[int, List[int]], 
                          position: Optional[str] = None) -> pd.DataFrame:
        """
        Import weekly player data
        
        Args:
            years: Single year or list of years
            position: Filter by position (QB, RB, WR, TE, etc.)
            
        Returns:
            DataFrame with weekly player stats
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("weekly", years=",".join(map(str, years)), 
                                       position=position or "all")
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            logger.info(f"Retrieved weekly data from cache")
            return cached_df
            
        try:
            logger.info(f"Fetching weekly data for years: {years}")
            df = nfl.import_weekly_data(years)
            
            # Filter by position if specified
            if position:
                df = df[df['position'] == position]
                
            # Cache based on current season
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            logger.info(f"Retrieved {len(df)} weekly player records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weekly data: {e}")
            return pd.DataFrame()
            
    def import_rosters(self, years: Union[int, List[int]]) -> pd.DataFrame:
        """
        Import seasonal rosters
        
        Args:
            years: Single year or list of years
            
        Returns:
            DataFrame with roster data
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("rosters", years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching roster data for years: {years}")
            df = nfl.import_rosters(years)
            
            # Cache
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching roster data: {e}")
            return pd.DataFrame()
            
    def import_schedules(self, years: Union[int, List[int]]) -> pd.DataFrame:
        """
        Import team schedules and game results
        
        Args:
            years: Single year or list of years
            
        Returns:
            DataFrame with schedule data
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("schedules", years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching schedule data for years: {years}")
            df = nfl.import_schedules(years)
            
            # Cache
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching schedule data: {e}")
            return pd.DataFrame()
            
    def import_ngs_data(self, stat_type: str, years: Union[int, List[int]]) -> pd.DataFrame:
        """
        Import Next Gen Stats data
        
        Args:
            stat_type: Type of NGS data ('passing', 'rushing', 'receiving')
            years: Single year or list of years (2016+)
            
        Returns:
            DataFrame with Next Gen Stats
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("ngs", stat_type=stat_type, 
                                       years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching Next Gen Stats ({stat_type}) for years: {years}")
            df = nfl.import_ngs_data(stat_type, years)
            
            # Cache
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching NGS data: {e}")
            return pd.DataFrame()
            
    def import_combine_data(self, years: Union[int, List[int]]) -> pd.DataFrame:
        """
        Import NFL Combine data
        
        Args:
            years: Single year or list of years
            
        Returns:
            DataFrame with combine measurements
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("combine", years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching combine data for years: {years}")
            df = nfl.import_combine_data(years)
            
            # Cache (historical data, use long TTL)
            self._save_to_cache(cache_key, df, self.HISTORICAL_TTL)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching combine data: {e}")
            return pd.DataFrame()
            
    def import_qbr(self, years: Union[int, List[int]], level: str = 'nfl', 
                   frequency: str = 'weekly') -> pd.DataFrame:
        """
        Import QBR (Quarterback Rating) data
        
        Args:
            years: Single year or list of years (2006+)
            level: 'nfl' or 'college'
            frequency: 'weekly' or 'season'
            
        Returns:
            DataFrame with QBR data
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("qbr", years=",".join(map(str, years)),
                                       level=level, frequency=frequency)
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching QBR data for years: {years}")
            df = nfl.import_qbr(years, level=level, frequency=frequency)
            
            # Cache
            current_year = datetime.now().year
            ttl = self.CURRENT_TTL if current_year in years else self.HISTORICAL_TTL
            self._save_to_cache(cache_key, df, ttl)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching QBR data: {e}")
            return pd.DataFrame()
            
    def import_win_totals(self, years: Union[int, List[int]]) -> pd.DataFrame:
        """
        Import team win totals and betting lines
        
        Args:
            years: Single year or list of years
            
        Returns:
            DataFrame with win totals
        """
        if isinstance(years, int):
            years = [years]
            
        cache_key = self._get_cache_key("win_totals", years=",".join(map(str, years)))
        
        # Check cache
        cached_df = self._get_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
            
        try:
            logger.info(f"Fetching win totals for years: {years}")
            df = nfl.import_win_totals(years)
            
            # Cache
            self._save_to_cache(cache_key, df, self.HISTORICAL_TTL)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching win totals: {e}")
            return pd.DataFrame()
            
    def get_player_stats_enriched(self, player_name: str, years: List[int]) -> pd.DataFrame:
        """
        Get enriched player statistics combining multiple data sources
        
        Args:
            player_name: Player name to search for
            years: List of years to include
            
        Returns:
            DataFrame with comprehensive player stats
        """
        # Get weekly data
        weekly_df = self.import_weekly_data(years)
        
        # Filter for player
        player_df = weekly_df[weekly_df['player_display_name'].str.contains(
            player_name, case=False, na=False
        )]
        
        if player_df.empty:
            logger.warning(f"No data found for player: {player_name}")
            return pd.DataFrame()
            
        # Get player position
        position = player_df['position'].iloc[0]
        
        # Add Next Gen Stats if available
        if position in ['QB', 'RB', 'WR', 'TE']:
            stat_type = 'passing' if position == 'QB' else 'rushing' if position == 'RB' else 'receiving'
            
            # Next Gen Stats available from 2016
            ngs_years = [y for y in years if y >= 2016]
            if ngs_years:
                ngs_df = self.import_ngs_data(stat_type, ngs_years)
                
                # Merge NGS data
                if not ngs_df.empty:
                    merge_cols = ['player_id', 'season', 'week'] if 'week' in ngs_df.columns else ['player_id', 'season']
                    player_df = player_df.merge(ngs_df, on=merge_cols, how='left')
                    
        return player_df
        
    def build_training_features(self, years: List[int]) -> pd.DataFrame:
        """
        Build comprehensive feature set for ML training
        
        Args:
            years: List of years to include
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Building training features for years: {years}")
        
        # Get base weekly data
        weekly_df = self.import_weekly_data(years)
        
        # Get schedules for opponent strength
        schedules_df = self.import_schedules(years)
        
        # Calculate team offensive/defensive rankings
        team_stats = weekly_df.groupby(['season', 'week', 'recent_team']).agg({
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'fantasy_points_ppr': 'sum'
        }).reset_index()
        
        # Add rolling averages
        for window in [3, 5, 10]:
            weekly_df[f'fantasy_points_ppr_ma{window}'] = (
                weekly_df.groupby('player_id')['fantasy_points_ppr']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
        # Add consistency metrics
        weekly_df['fantasy_points_ppr_std'] = (
            weekly_df.groupby('player_id')['fantasy_points_ppr']
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        logger.info(f"Created feature set with {len(weekly_df.columns)} features")
        return weekly_df


# Example usage
if __name__ == "__main__":
    client = NFLDataPyClient()
    
    # Test fetching weekly data
    print("Testing NFL Data Py client...")
    df = client.import_weekly_data(2023)
    print(f"Retrieved {len(df)} records for 2023 season")
    
    if not df.empty:
        print("\nSample columns available:")
        print(df.columns.tolist()[:20])
        
        print("\nTop 5 fantasy performers (PPR):")
        top_players = df.nlargest(5, 'fantasy_points_ppr')[
            ['player_display_name', 'position', 'week', 'fantasy_points_ppr']
        ]
        print(top_players)