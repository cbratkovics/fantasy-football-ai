"""
ESPN Fantasy Football API client for data collection and integration.

This module provides a professional interface to ESPN's Fantasy Football API,
integrating seamlessly with the database layer for data persistence.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

try:
    from espn_api.football import League
except ImportError:
    raise ImportError("espn_api package required. Install with: pip install espn_api")

from ..storage.models import DatabaseManager, Player, WeeklyStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ESPNConfig:
    """Configuration for ESPN API access."""
    league_id: int
    espn_s2: str
    swid: str
    seasons: List[int]
    max_weeks: int = 18
    excluded_positions: List[str] = None
    
    def __post_init__(self):
        if self.excluded_positions is None:
            self.excluded_positions = ["K", "D/ST"]


class ESPNDataCollector:
    """
    Professional ESPN Fantasy Football data collector.
    
    Handles data collection from ESPN API with robust error handling,
    data validation, and database integration.
    """
    
    def __init__(self, config: Optional[ESPNConfig] = None, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize ESPN data collector.
        
        Args:
            config: ESPN API configuration
            db_manager: Database manager for data persistence
        """
        # Load environment variables
        load_dotenv()
        
        # Set up configuration
        if config is None:
            config = self._load_default_config()
        self.config = config
        
        # Set up database manager
        self.db_manager = db_manager or DatabaseManager()
        
        # Validate credentials
        self._validate_credentials()
        
        logger.info(f"ESPN Data Collector initialized for league {self.config.league_id}")
        logger.info(f"Configured seasons: {self.config.seasons}")
    
    def _load_default_config(self) -> ESPNConfig:
        """Load default configuration from environment variables."""
        return ESPNConfig(
            league_id=int(os.getenv("LEAGUE_ID", "1099505687")),
            espn_s2=os.getenv("ESPN_S2"),
            swid=os.getenv("SWID"),
            seasons=[2022, 2023, 2024]
        )
    
    def _validate_credentials(self) -> None:
        """Validate ESPN API credentials."""
        if not self.config.espn_s2 or not self.config.swid:
            raise ValueError("ESPN_S2 and SWID credentials are required")
        
        # Log first few characters for verification (safely)
        logger.info(f"ESPN_S2: {self.config.espn_s2[:6]}...")
        logger.info(f"SWID: {self.config.swid[:6]}...")
    
    def collect_all_seasons(self, save_to_db: bool = True, save_to_csv: bool = False) -> pd.DataFrame:
        """
        Collect data for all configured seasons.
        
        Args:
            save_to_db: Whether to save data to database
            save_to_csv: Whether to save data to CSV files
            
        Returns:
            Combined DataFrame with all seasons data
        """
        all_seasons_data = []
        collection_summary = {
            'total_records': 0,
            'successful_seasons': 0,
            'failed_seasons': [],
            'players_processed': set()
        }
        
        logger.info(f"Starting data collection for {len(self.config.seasons)} seasons...")
        
        for season in self.config.seasons:
            try:
                season_data = self.collect_season_data(season)
                
                if not season_data.empty:
                    all_seasons_data.append(season_data)
                    collection_summary['total_records'] += len(season_data)
                    collection_summary['successful_seasons'] += 1
                    collection_summary['players_processed'].update(season_data['player_id'].unique())
                    
                    # Save individual season if requested
                    if save_to_csv:
                        self._save_season_csv(season_data, season)
                        
                    # Save to database if requested
                    if save_to_db:
                        self._save_season_to_db(season_data)
                        
                    logger.info(f"✅ Season {season}: {len(season_data)} records collected")
                else:
                    collection_summary['failed_seasons'].append(season)
                    logger.warning(f"❌ Season {season}: No data collected")
                    
            except Exception as e:
                collection_summary['failed_seasons'].append(season)
                logger.error(f"❌ Season {season} failed: {e}")
        
        # Combine all seasons
        if all_seasons_data:
            combined_df = pd.concat(all_seasons_data, ignore_index=True)
            
            # Save combined dataset
            if save_to_csv:
                self._save_combined_csv(combined_df)
            
            # Log collection summary
            self._log_collection_summary(collection_summary)
            
            return combined_df
        else:
            logger.error("No data collected for any season")
            return pd.DataFrame()
    
    def collect_season_data(self, season: int) -> pd.DataFrame:
        """
        Collect data for a specific season.
        
        Args:
            season: Year of the season to collect
            
        Returns:
            DataFrame with season data
        """
        logger.info(f"Collecting data for season {season}...")
        
        try:
            # Initialize ESPN API client
            league = League(
                league_id=self.config.league_id,
                year=season,
                espn_s2=self.config.espn_s2,
                swid=self.config.swid
            )
            
            weekly_stats = []
            
            # Collect data for each week
            for week in range(1, self.config.max_weeks):
                try:
                    week_data = self._collect_week_data(league, season, week)
                    weekly_stats.extend(week_data)
                    
                except Exception as e:
                    logger.warning(f"Week {week} failed for season {season}: {e}")
                    continue
            
            # Convert to DataFrame and clean
            if weekly_stats:
                df = pd.DataFrame(weekly_stats)
                df = self._clean_season_data(df)
                logger.info(f"Season {season}: Collected {len(df)} records")
                return df
            else:
                logger.warning(f"No data collected for season {season}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to initialize league for season {season}: {e}")
            return pd.DataFrame()
    
    def _collect_week_data(self, league: League, season: int, week: int) -> List[Dict[str, Any]]:
        """
        Collect data for a specific week.
        
        Args:
            league: ESPN League instance
            season: Season year
            week: Week number
            
        Returns:
            List of player statistics dictionaries
        """
        try:
            matchups = league.box_scores(week)
            week_stats = []
            
            for matchup in matchups:
                # Process both away and home lineups
                for player in matchup.away_lineup + matchup.home_lineup:
                    # Extract player data with safe attribute access
                    player_data = {
                        'season': season,
                        'week': week,
                        'player_id': getattr(player, 'playerId', None),
                        'player_name': getattr(player, 'name', 'Unknown'),
                        'team': getattr(player, 'proTeam', 'Unknown'),
                        'position': getattr(player, 'position', 'Unknown'),
                        'fantasy_points': getattr(player, 'points', 0.0),
                        'projected_points': getattr(player, 'projected_points', 0.0)
                    }
                    
                    # Validate required fields
                    if player_data['player_id'] and player_data['player_name'] != 'Unknown':
                        week_stats.append(player_data)
            
            logger.debug(f"Week {week}: {len(week_stats)} player records collected")
            return week_stats
            
        except Exception as e:
            logger.error(f"Error collecting week {week} data: {e}")
            return []
    
    def _clean_season_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate season data.
        
        Args:
            df: Raw season DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove excluded positions
        initial_count = len(df)
        df = df[~df['position'].isin(self.config.excluded_positions)]
        logger.info(f"Filtered out {initial_count - len(df)} records from excluded positions: {self.config.excluded_positions}")
        
        # Remove records with missing critical data
        df = df.dropna(subset=['player_id', 'player_name'])
        
        # Convert data types
        df['player_id'] = df['player_id'].astype(int)
        df['fantasy_points'] = pd.to_numeric(df['fantasy_points'], errors='coerce').fillna(0.0)
        df['projected_points'] = pd.to_numeric(df['projected_points'], errors='coerce').fillna(0.0)
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['player_id', 'season', 'week'])
        after_dedup = len(df)
        
        if before_dedup != after_dedup:
            logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        # Log unique positions for verification
        unique_positions = sorted(df['position'].unique())
        logger.info(f"Unique positions after cleaning: {unique_positions}")
        
        return df
    
    def _save_season_to_db(self, season_df: pd.DataFrame) -> None:
        """Save season data to database."""
        if season_df.empty:
            return
        
        logger.info(f"Saving {len(season_df)} records to database...")
        
        # Insert/update players
        players_saved = 0
        for _, row in season_df.drop_duplicates('player_id').iterrows():
            player = Player(
                player_id=int(row['player_id']),
                name=row['player_name'],
                position=row['position'],
                team=row['team']
            )
            if self.db_manager.insert_player(player):
                players_saved += 1
        
        logger.info(f"Processed {players_saved} unique players")
        
        # Bulk insert weekly stats
        weekly_stats = []
        for _, row in season_df.iterrows():
            stats = WeeklyStats(
                player_id=int(row['player_id']),
                season=int(row['season']),
                week=int(row['week']),
                fantasy_points=float(row['fantasy_points']),
                projected_points=float(row['projected_points'])
            )
            weekly_stats.append(stats)
        
        stats_saved = self.db_manager.bulk_insert_weekly_stats(weekly_stats)
        logger.info(f"Saved {stats_saved} weekly statistics records")
    
    def _save_season_csv(self, season_df: pd.DataFrame, season: int) -> None:
        """Save season data to CSV file."""
        if season_df.empty:
            return
            
        filename = f"data/fantasy_weekly_stats_{season}.csv"
        os.makedirs("data", exist_ok=True)
        season_df.to_csv(filename, index=False)
        logger.info(f"Season {season} data saved to {filename}")
    
    def _save_combined_csv(self, combined_df: pd.DataFrame) -> None:
        """Save combined seasons data to CSV file."""
        if combined_df.empty:
            return
            
        filename = "data/fantasy_weekly_stats_combined.csv"
        combined_df.to_csv(filename, index=False)
        logger.info(f"Combined data saved to {filename}")
    
    def _log_collection_summary(self, summary: Dict[str, Any]) -> None:
        """Log collection summary statistics."""
        logger.info("=== DATA COLLECTION SUMMARY ===")
        logger.info(f"Total records collected: {summary['total_records']}")
        logger.info(f"Successful seasons: {summary['successful_seasons']}/{len(self.config.seasons)}")
        logger.info(f"Unique players processed: {len(summary['players_processed'])}")
        
        if summary['failed_seasons']:
            logger.warning(f"Failed seasons: {summary['failed_seasons']}")
    
    def update_recent_data(self, weeks_back: int = 4) -> pd.DataFrame:
        """
        Update only recent weeks of data for current season.
        
        Args:
            weeks_back: Number of recent weeks to update
            
        Returns:
            DataFrame with updated data
        """
        current_season = max(self.config.seasons)
        logger.info(f"Updating last {weeks_back} weeks for season {current_season}")
        
        # This could be enhanced to determine current week automatically
        # For now, we'll update the specified number of recent weeks
        try:
            league = League(
                league_id=self.config.league_id,
                year=current_season,
                espn_s2=self.config.espn_s2,
                swid=self.config.swid
            )
            
            # Get current week (simplified approach)
            current_week = min(18, datetime.now().isocalendar()[1])  # Rough estimate
            start_week = max(1, current_week - weeks_back)
            
            weekly_stats = []
            for week in range(start_week, current_week + 1):
                try:
                    week_data = self._collect_week_data(league, current_season, week)
                    weekly_stats.extend(week_data)
                except Exception as e:
                    logger.warning(f"Failed to update week {week}: {e}")
            
            if weekly_stats:
                df = pd.DataFrame(weekly_stats)
                df = self._clean_season_data(df)
                self._save_season_to_db(df)
                logger.info(f"Updated {len(df)} recent records")
                return df
            else:
                logger.warning("No recent data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to update recent data: {e}")
            return pd.DataFrame()
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary of data in database."""
        return self.db_manager.get_database_stats()


def create_espn_collector(
    league_id: Optional[int] = None,
    seasons: Optional[List[int]] = None,
    db_path: Optional[str] = None
) -> ESPNDataCollector:
    """
    Factory function to create ESPN data collector with custom configuration.
    
    Args:
        league_id: ESPN league ID (uses default if not provided)
        seasons: List of seasons to collect (uses default if not provided)
        db_path: Path to database file
        
    Returns:
        Configured ESPNDataCollector instance
    """
    config = None
    if league_id or seasons:
        # Load environment variables for credentials
        load_dotenv()
        config = ESPNConfig(
            league_id=league_id or int(os.getenv("LEAGUE_ID", "1099505687")),
            espn_s2=os.getenv("ESPN_S2"),
            swid=os.getenv("SWID"),
            seasons=seasons or [2022, 2023, 2024]
        )
    
    db_manager = DatabaseManager(db_path) if db_path else None
    return ESPNDataCollector(config, db_manager)


if __name__ == "__main__":
    # Example usage
    try:
        # Create collector with default configuration
        collector = create_espn_collector()
        
        # Collect all seasons data
        data = collector.collect_all_seasons(save_to_db=True, save_to_csv=True)
        
        # Print summary
        if not data.empty:
            print(f"\n✅ Successfully collected {len(data)} total records")
            print(f"Seasons: {sorted(data['season'].unique())}")
            print(f"Unique players: {data['player_id'].nunique()}")
            print(f"Positions: {sorted(data['position'].unique())}")
            
            # Show database summary
            summary = collector.get_database_summary()
            print("\n📊 Database Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No data collected")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise