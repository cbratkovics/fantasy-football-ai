"""
ESPN Fantasy Football API Client
Uses ESPN_S2 and SWID for authentication
"""

import os
import requests
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ESPNClient:
    """
    ESPN Fantasy Football API Client
    
    Requires ESPN_S2 and ESPN_SWID cookies for private league access
    """
    
    BASE_URL = "https://fantasy.espn.com/apis/v3/games/ffl"
    
    def __init__(self):
        """Initialize with ESPN credentials from environment"""
        self.s2 = os.getenv('ESPN_S2')
        self.swid = os.getenv('ESPN_SWID')
        
        if not self.s2 or not self.swid:
            logger.warning("ESPN credentials not found in environment")
            
        # Set up session with cookies
        self.session = requests.Session()
        if self.s2 and self.swid:
            self.session.cookies.update({
                'espn_s2': self.s2,
                'SWID': self.swid
            })
            
    def get_league(self, league_id: int, year: int) -> Optional[Dict[str, Any]]:
        """
        Get league information
        
        Args:
            league_id: ESPN league ID
            year: Season year
            
        Returns:
            League data or None if error
        """
        url = f"{self.BASE_URL}/seasons/{year}/segments/0/leagues/{league_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching league {league_id}: {str(e)}")
            return None
            
    def get_teams(self, league_id: int, year: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get all teams in a league
        
        Args:
            league_id: ESPN league ID
            year: Season year
            
        Returns:
            List of teams or None if error
        """
        league_data = self.get_league(league_id, year)
        if league_data:
            return league_data.get('teams', [])
        return None
        
    def get_players(self, year: int) -> Optional[Dict[str, Any]]:
        """
        Get all NFL players with ESPN data
        
        Args:
            year: Season year
            
        Returns:
            Player data or None if error
        """
        url = f"{self.BASE_URL}/seasons/{year}/players"
        params = {
            'view': 'players_wl'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching players: {str(e)}")
            return None
            
    def get_player_stats(self, league_id: int, year: int, scoring_period: int) -> Optional[Dict[str, Any]]:
        """
        Get player stats for a specific week
        
        Args:
            league_id: ESPN league ID
            year: Season year
            scoring_period: Week number
            
        Returns:
            Player stats or None if error
        """
        url = f"{self.BASE_URL}/seasons/{year}/segments/0/leagues/{league_id}"
        params = {
            'view': 'kona_player_info',
            'scoringPeriodId': scoring_period
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            return None
            
    def get_matchups(self, league_id: int, year: int, week: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get matchups for a specific week
        
        Args:
            league_id: ESPN league ID
            year: Season year
            week: Week number
            
        Returns:
            List of matchups or None if error
        """
        url = f"{self.BASE_URL}/seasons/{year}/segments/0/leagues/{league_id}"
        params = {
            'view': 'mMatchupScore',
            'scoringPeriodId': week
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('schedule', [])
        except Exception as e:
            logger.error(f"Error fetching matchups: {str(e)}")
            return None
            
    def get_free_agents(self, league_id: int, year: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get available free agents
        
        Args:
            league_id: ESPN league ID
            year: Season year
            
        Returns:
            List of free agents or None if error
        """
        url = f"{self.BASE_URL}/seasons/{year}/segments/0/leagues/{league_id}"
        params = {
            'view': 'kona_player_info'
        }
        
        headers = {
            'x-fantasy-filter': '{"players":{"filterStatus":{"value":["FREEAGENT","WAIVERS"]}}}'
        }
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get('players', [])
        except Exception as e:
            logger.error(f"Error fetching free agents: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    client = ESPNClient()
    
    # Example: Get league info (replace with your league ID)
    # league_data = client.get_league(league_id=123456, year=2024)
    # if league_data:
    #     print(f"League: {league_data.get('settings', {}).get('name', 'Unknown')}")
    
    print("ESPN Client configured with credentials" if client.s2 else "No ESPN credentials found")