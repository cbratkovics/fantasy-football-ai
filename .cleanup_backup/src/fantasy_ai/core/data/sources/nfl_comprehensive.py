"""
Comprehensive NFL API integration for Fantasy Football AI Assistant.

This module integrates with API-Sports NFL API to get comprehensive 
player and team data for all NFL players, not just those in your ESPN league.
"""

import logging
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class NFLPlayerStats:
    """NFL player statistics from comprehensive API."""
    player_id: str
    name: str
    position: str
    team: str
    season: int
    week: int
    games_played: int
    passing_yards: Optional[int] = None
    passing_tds: Optional[int] = None
    rushing_yards: Optional[int] = None
    rushing_tds: Optional[int] = None
    receiving_yards: Optional[int] = None
    receiving_tds: Optional[int] = None
    receptions: Optional[int] = None
    fantasy_points: Optional[float] = None


@dataclass 
class NFLTeam:
    """NFL team information."""
    team_id: str
    name: str
    code: str
    city: str
    conference: str
    division: str


class NFLAPIClient:
    """
    Client for comprehensive NFL data from API-Sports.
    
    Provides access to all NFL players, teams, and statistics
    beyond what's available in your ESPN league.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NFL API client.
        
        Args:
            api_key: API-Sports API key (gets from env if not provided)
        """
        self.api_key = api_key or os.getenv('NFL_API_KEY')
        if not self.api_key:
            raise ValueError("NFL_API_KEY is required. Get one from https://api-sports.io/")
        
        self.base_url = "https://v1.american-football.api-sports.io"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'v1.american-football.api-sports.io'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Get the correct league IDs
        self.nfl_league_id = None
        self._initialize_league_ids()
        
        logger.info("NFL API Client initialized")
    
    def _initialize_league_ids(self):
        """Get the correct league ID for NFL."""
        try:
            leagues = self.get_leagues()
            for league_data in leagues:
                league = league_data.get('league', {})
                if league.get('name', '').upper() == 'NFL':
                    self.nfl_league_id = league.get('id')
                    logger.info(f"Found NFL league with ID: {self.nfl_league_id}")
                    break
            
            if not self.nfl_league_id:
                logger.warning("Could not find NFL league ID, defaulting to 2")
                self.nfl_league_id = 2  # Based on debug output
        except Exception as e:
            logger.warning(f"Could not initialize league IDs: {e}, defaulting to 2")
            self.nfl_league_id = 2
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params or {})
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)  # Retry
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {}
    
    def get_leagues(self) -> List[Dict[str, Any]]:
        """Get available NFL leagues/seasons."""
        response = self._make_request("leagues")
        return response.get('response', [])
    
    def get_teams(self, season: int = 2024) -> List[NFLTeam]:
        """
        Get all NFL teams.
        
        Args:
            season: Season year
            
        Returns:
            List of NFL teams
        """
        if not self.nfl_league_id:
            logger.error("NFL league ID not found")
            return []
        
        params = {'league': self.nfl_league_id, 'season': season}
        logger.info(f"Getting teams with params: {params}")
        response = self._make_request("teams", params)
        
        teams = []
        if response and 'response' in response:
            for team_data in response.get('response', []):
                team = team_data.get('team', {})
                teams.append(NFLTeam(
                    team_id=str(team.get('id', '')),
                    name=team.get('name', ''),
                    code=team.get('code', ''),
                    city=team.get('city', ''),
                    conference=team_data.get('conference', ''),
                    division=team_data.get('division', '')
                ))
        
        logger.info(f"Retrieved {len(teams)} NFL teams")
        return teams
    
    def get_players(self, team_id: Optional[str] = None, season: int = 2024) -> List[Dict[str, Any]]:
        """
        Get NFL players, optionally filtered by team.
        
        Args:
            team_id: Team ID to filter by (None for all teams)
            season: Season year
            
        Returns:
            List of player data
        """
        if not self.nfl_league_id:
            logger.error("NFL league ID not found")
            return []
        
        params = {'league': self.nfl_league_id, 'season': season}
        if team_id:
            params['team'] = team_id
        
        response = self._make_request("players", params)
        players = response.get('response', []) if response else []
        
        logger.info(f"Retrieved {len(players)} players" + (f" for team {team_id}" if team_id else ""))
        return players
    
    def get_all_players(self, season: int = 2024) -> pd.DataFrame:
        """
        Get all NFL players from all teams.
        
        Args:
            season: Season year
            
        Returns:
            DataFrame with all player information
        """
        logger.info(f"Collecting all NFL players for {season} season...")
        
        # Get all teams first
        teams = self.get_teams(season=season)
        if not teams:
            logger.error("No teams found, cannot get players")
            return pd.DataFrame()
        
        all_players = []
        
        for team in teams:
            logger.info(f"Getting players for {team.name}...")
            team_players = self.get_players(team.team_id, season)
            
            for player_data in team_players:
                player = player_data.get('player', {})
                all_players.append({
                    'player_id': str(player.get('id', '')),
                    'name': player.get('name', ''),
                    'position': player.get('position', ''),
                    'team_id': team.team_id,
                    'team_name': team.name,
                    'team_code': team.code,
                    'age': player.get('age'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'season': season
                })
            
            # Small delay to respect rate limits
            time.sleep(0.5)
        
        df = pd.DataFrame(all_players)
        logger.info(f"Collected {len(df)} total players from {len(teams)} teams")
        return df
    
    def get_player_statistics(self, player_id: str, season: int = 2024) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific player.
        
        Args:
            player_id: Player ID
            season: Season year
            
        Returns:
            Player statistics
        """
        if not self.nfl_league_id:
            logger.error("NFL league ID not found")
            return {}
        
        params = {'league': self.nfl_league_id, 'season': season, 'player': player_id}
        response = self._make_request("players/statistics", params)
        
        stats = response.get('response', []) if response else []
        return stats[0] if stats else {}
    
    def get_games(self, season: int = 2024, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get NFL games for a season/week.
        
        Args:
            season: Season year
            week: Specific week (None for all weeks)
            
        Returns:
            List of game data
        """
        if not self.nfl_league_id:
            logger.error("NFL league ID not found")
            return []
        
        params = {'league': self.nfl_league_id, 'season': season}
        if week:
            params['week'] = week
        
        response = self._make_request("games", params)
        games = response.get('response', []) if response else []
        
        logger.info(f"Retrieved {len(games)} games" + (f" for week {week}" if week else f" for {season}"))
        return games
    
    def get_comprehensive_player_stats(self, season: int = 2024) -> pd.DataFrame:
        """
        Get comprehensive statistics for all players in a season.
        
        Args:
            season: Season year
            
        Returns:
            DataFrame with detailed player statistics
        """
        logger.info(f"Collecting comprehensive player statistics for {season}...")
        
        # First get all players
        players_df = self.get_all_players(season)
        
        # Then get detailed stats for each player
        all_stats = []
        
        for _, player in players_df.iterrows():
            player_id = player['player_id']
            logger.info(f"Getting stats for {player['name']} ({player_id})...")
            
            try:
                stats = self.get_player_statistics(player_id, season)
                
                if stats:
                    # Parse statistics based on position
                    parsed_stats = self._parse_player_stats(stats, player)
                    all_stats.append(parsed_stats)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to get stats for {player['name']}: {e}")
                continue
        
        stats_df = pd.DataFrame(all_stats)
        logger.info(f"Collected statistics for {len(stats_df)} players")
        return stats_df
    
    def _parse_player_stats(self, stats_data: Dict[str, Any], player_info: pd.Series) -> Dict[str, Any]:
        """
        Parse player statistics from API response.
        
        Args:
            stats_data: Raw statistics from API
            player_info: Player information
            
        Returns:
            Parsed statistics dictionary
        """
        statistics = stats_data.get('statistics', [{}])[0] if stats_data.get('statistics') else {}
        
        parsed = {
            'player_id': player_info['player_id'],
            'name': player_info['name'],
            'position': player_info['position'],
            'team_name': player_info['team_name'],
            'season': player_info['season'],
            'games_played': statistics.get('games', {}).get('played', 0)
        }
        
        # Position-specific stats
        if player_info['position'] == 'QB':
            passing = statistics.get('passing', {})
            parsed.update({
                'passing_attempts': passing.get('attempts', 0),
                'passing_completions': passing.get('completions', 0),
                'passing_yards': passing.get('yards', 0),
                'passing_tds': passing.get('touchdowns', 0),
                'passing_interceptions': passing.get('interceptions', 0)
            })
        
        # Rushing stats (RB, some QB, WR)
        rushing = statistics.get('rushing', {})
        if rushing:
            parsed.update({
                'rushing_attempts': rushing.get('attempts', 0),
                'rushing_yards': rushing.get('yards', 0),
                'rushing_tds': rushing.get('touchdowns', 0)
            })
        
        # Receiving stats (WR, TE, RB)
        receiving = statistics.get('receiving', {})
        if receiving:
            parsed.update({
                'receptions': receiving.get('receptions', 0),
                'receiving_yards': receiving.get('yards', 0),
                'receiving_tds': receiving.get('touchdowns', 0),
                'targets': receiving.get('targets', 0)
            })
        
        # Calculate fantasy points (standard scoring)
        parsed['fantasy_points'] = self._calculate_fantasy_points(parsed)
        
        return parsed
    
    def _calculate_fantasy_points(self, stats: Dict[str, Any]) -> float:
        """
        Calculate standard fantasy points from statistics.
        
        Args:
            stats: Player statistics
            
        Returns:
            Fantasy points
        """
        points = 0.0
        
        # Passing points (4 points per TD, 1 point per 25 yards, -2 per INT)
        points += stats.get('passing_tds', 0) * 4
        points += stats.get('passing_yards', 0) * 0.04  # 1 point per 25 yards
        points -= stats.get('passing_interceptions', 0) * 2
        
        # Rushing points (6 points per TD, 1 point per 10 yards)
        points += stats.get('rushing_tds', 0) * 6
        points += stats.get('rushing_yards', 0) * 0.1  # 1 point per 10 yards
        
        # Receiving points (6 points per TD, 1 point per 10 yards, 1 per reception)
        points += stats.get('receiving_tds', 0) * 6
        points += stats.get('receiving_yards', 0) * 0.1  # 1 point per 10 yards
        points += stats.get('receptions', 0) * 1  # PPR scoring
        
        return round(points, 2)


class ComprehensiveNFLData:
    """
    Combines ESPN league data with comprehensive NFL API data.
    
    This class merges your specific league data with comprehensive 
    NFL player data to provide complete coverage for ML training.
    """
    
    def __init__(self, nfl_api_client: NFLAPIClient):
        """
        Initialize comprehensive data manager.
        
        Args:
            nfl_api_client: NFL API client instance
        """
        self.nfl_client = nfl_api_client
        logger.info("Comprehensive NFL Data manager initialized")
    
    def merge_espn_with_comprehensive_data(
        self, 
        espn_data: pd.DataFrame, 
        season: int = 2024
    ) -> pd.DataFrame:
        """
        Merge ESPN league data with comprehensive NFL data.
        
        Args:
            espn_data: DataFrame from ESPN API
            season: Season year
            
        Returns:
            Combined DataFrame with comprehensive player coverage
        """
        logger.info("Merging ESPN data with comprehensive NFL data...")
        
        # Get comprehensive NFL data
        comprehensive_data = self.nfl_client.get_comprehensive_player_stats(season)
        
        # Merge based on player name and position (ESPN doesn't have same player IDs)
        merged_data = pd.merge(
            espn_data,
            comprehensive_data,
            on=['name', 'position'],
            how='outer',
            suffixes=('_espn', '_nfl')
        )
        
        # Fill missing values and combine data
        merged_data = self._combine_data_sources(merged_data)
        
        logger.info(f"Merged data contains {len(merged_data)} players")
        return merged_data
    
    def _combine_data_sources(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently combine data from ESPN and NFL API sources.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Combined DataFrame with best data from both sources
        """
        # Use ESPN data for fantasy points when available (more accurate for fantasy)
        merged_df['fantasy_points'] = merged_df['fantasy_points_espn'].fillna(
            merged_df['fantasy_points_nfl']
        )
        
        # Use NFL API data for comprehensive stats
        stat_columns = [
            'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
            'receiving_yards', 'receiving_tds', 'receptions', 'games_played'
        ]
        
        for col in stat_columns:
            if f'{col}_nfl' in merged_df.columns:
                merged_df[col] = merged_df[f'{col}_nfl'].fillna(0)
        
        # Clean up duplicate columns
        columns_to_keep = [
            'player_id', 'name', 'position', 'team_name', 'season', 'week',
            'fantasy_points'
        ] + stat_columns
        
        return merged_df[columns_to_keep + [col for col in merged_df.columns if col not in columns_to_keep]]
    
    def get_missing_players_analysis(self, espn_data: pd.DataFrame, season: int = 2024) -> Dict[str, Any]:
        """
        Analyze what players are missing from ESPN data vs comprehensive data.
        
        Args:
            espn_data: ESPN league data
            season: Season year
            
        Returns:
            Analysis of missing players and coverage gaps
        """
        comprehensive_data = self.nfl_client.get_all_players(season)
        
        espn_players = set(espn_data['name'].unique())
        all_nfl_players = set(comprehensive_data['name'].unique())
        
        missing_from_espn = all_nfl_players - espn_players
        only_in_espn = espn_players - all_nfl_players
        
        analysis = {
            'total_nfl_players': len(all_nfl_players),
            'espn_league_players': len(espn_players),
            'missing_from_espn': len(missing_from_espn),
            'only_in_espn': len(only_in_espn),
            'coverage_percentage': len(espn_players) / len(all_nfl_players) * 100,
            'missing_players_sample': list(missing_from_espn)[:20],  # First 20
            'positions_missing': comprehensive_data[
                comprehensive_data['name'].isin(missing_from_espn)
            ]['position'].value_counts().to_dict()
        }
        
        logger.info(f"ESPN league covers {analysis['coverage_percentage']:.1f}% of NFL players")
        return analysis


def create_comprehensive_nfl_client(api_key: Optional[str] = None) -> ComprehensiveNFLData:
    """
    Factory function to create comprehensive NFL data client.
    
    Args:
        api_key: NFL API key (optional, gets from env)
        
    Returns:
        Configured comprehensive NFL data client
    """
    nfl_client = NFLAPIClient(api_key)
    return ComprehensiveNFLData(nfl_client)


if __name__ == "__main__":
    # Example usage
    try:
        # Create comprehensive NFL data client
        comprehensive_client = create_comprehensive_nfl_client()
        
        # Get comprehensive player data
        logger.info("Testing comprehensive NFL data collection...")
        
        # Get all teams
        teams = comprehensive_client.nfl_client.get_teams(season=2024)
        print(f"Found {len(teams)} NFL teams")
        
        # Get sample of players from one team
        if teams:
            sample_team = teams[0]
            players = comprehensive_client.nfl_client.get_players(sample_team.team_id, 2024)
            print(f"Found {len(players)} players for {sample_team.name}")
        
        # Note: Getting all player stats would use many API calls
        # In production, you'd want to cache this data
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print("Make sure to set NFL_API_KEY in your .env file")
        print("Get a free API key from https://api-sports.io/")