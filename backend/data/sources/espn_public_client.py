"""
ESPN Public API Client - No Authentication Required
Note: Check ESPN Terms of Service for commercial use restrictions
"""

import os
import redis
import json
import logging
import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class ESPNPublicClient:
    """
    ESPN Public API Client for accessing public NFL data
    
    Features:
    - Player information and stats
    - Team schedules and scores
    - Game scores and odds
    - Rate limiting (1 request/second)
    - Redis caching
    
    Note: This uses public endpoints. Verify ESPN ToS for commercial use.
    """
    
    BASE_URL = "https://site.api.espn.com/apis/"
    RATE_LIMIT = 1  # Requests per second
    
    def __init__(self):
        """Initialize ESPN public client with rate limiting"""
        # Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Cache TTLs
        self.CACHE_TTL = 3600  # 1 hour for most data
        self.STATIC_TTL = 86400  # 24 hours for static data
        
        # Rate limiting
        self.rate_limit = float(os.getenv('ESPN_RATE_LIMIT', '1'))
        self.last_request_time = 0
        
        # HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'FantasyFootballAI/1.0 (Educational/Research)'
            }
        )
        
        logger.info(f"ESPN Public client initialized (rate limit: {self.rate_limit} req/s)")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        
    async def _rate_limit_wait(self):
        """Enforce rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.rate_limit):
            wait_time = (1.0 / self.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)
            
        self.last_request_time = asyncio.get_event_loop().time()
        
    def _get_cache_key(self, endpoint: str, **params) -> str:
        """Generate cache key from endpoint and parameters"""
        param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"espn_public:{endpoint}:{param_str}"
        
    async def _make_request(self, endpoint: str, **params) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to ESPN API"""
        # Check cache first
        cache_key = self._get_cache_key(endpoint, **params)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            logger.debug(f"Cache hit for {endpoint}")
            return json.loads(cached_data)
            
        # Rate limit
        await self._rate_limit_wait()
        
        # Make request
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.redis_client.setex(cache_key, self.CACHE_TTL, json.dumps(data))
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
            
    async def get_player_info(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get player information from ESPN
        
        Args:
            player_id: ESPN player ID
            
        Returns:
            Player data dict or None
        """
        endpoint = f"common/v3/sports/football/nfl/athletes/{player_id}"
        data = await self._make_request(endpoint)
        
        if data and 'athlete' in data:
            return data['athlete']
        return None
        
    async def get_team_schedule(self, team_id: int, season: int = None) -> Optional[Dict[str, Any]]:
        """
        Get team schedule and scores
        
        Args:
            team_id: ESPN team ID
            season: Season year (default: current)
            
        Returns:
            Schedule data or None
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"site/v2/sports/football/nfl/teams/{team_id}/schedule"
        params = {'season': season}
        
        return await self._make_request(endpoint, **params)
        
    async def get_scoreboard(self, week: int, season: int = None) -> Optional[Dict[str, Any]]:
        """
        Get scores and game information for a week
        
        Args:
            week: Week number
            season: Season year (default: current)
            
        Returns:
            Scoreboard data or None
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"
        return await self._make_request(endpoint)
        
    async def get_team_roster(self, team_id: int) -> Optional[Dict[str, Any]]:
        """
        Get current team roster
        
        Args:
            team_id: ESPN team ID
            
        Returns:
            Roster data or None
        """
        endpoint = f"site/v2/sports/football/nfl/teams/{team_id}/roster"
        return await self._make_request(endpoint)
        
    async def get_player_gamelog(self, player_id: int, season: int = None) -> Optional[Dict[str, Any]]:
        """
        Get player game log for a season
        
        Args:
            player_id: ESPN player ID
            season: Season year (default: current)
            
        Returns:
            Game log data or None
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"common/v3/sports/football/nfl/athletes/{player_id}/gamelog"
        params = {'season': season}
        
        return await self._make_request(endpoint, **params)
        
    async def get_team_stats(self, team_id: int, season: int = None) -> Optional[Dict[str, Any]]:
        """
        Get team statistics
        
        Args:
            team_id: ESPN team ID
            season: Season year (default: current)
            
        Returns:
            Team stats or None
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"site/v2/sports/football/nfl/teams/{team_id}/statistics"
        params = {'season': season}
        
        return await self._make_request(endpoint, **params)
        
    def get_espn_team_id_mapping(self) -> Dict[str, int]:
        """
        Get mapping of team abbreviations to ESPN team IDs
        
        Returns:
            Dict mapping team abbreviations to ESPN IDs
        """
        return {
            'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2,
            'CAR': 29, 'CHI': 3, 'CIN': 4, 'CLE': 5,
            'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
            'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12,
            'LA': 14, 'LAC': 24, 'LV': 13, 'MIA': 15,
            'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
            'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SEA': 26,
            'SF': 25, 'TB': 27, 'TEN': 10, 'WAS': 28
        }
        
    async def get_week_games_with_odds(self, week: int, season: int = None) -> pd.DataFrame:
        """
        Get all games for a week with betting odds if available
        
        Args:
            week: Week number
            season: Season year
            
        Returns:
            DataFrame with game information and odds
        """
        scoreboard = await self.get_scoreboard(week, season)
        
        if not scoreboard or 'events' not in scoreboard:
            return pd.DataFrame()
            
        games_data = []
        
        for event in scoreboard['events']:
            game_info = {
                'game_id': event.get('id'),
                'date': event.get('date'),
                'name': event.get('name'),
                'short_name': event.get('shortName'),
                'status': event.get('status', {}).get('type', {}).get('name'),
            }
            
            # Get teams
            competitions = event.get('competitions', [])
            if competitions:
                competition = competitions[0]
                
                # Get competitors
                for competitor in competition.get('competitors', []):
                    team_type = 'home' if competitor.get('homeAway') == 'home' else 'away'
                    game_info[f'{team_type}_team'] = competitor.get('team', {}).get('abbreviation')
                    game_info[f'{team_type}_score'] = competitor.get('score')
                    
                # Get odds if available
                odds = competition.get('odds', [])
                if odds:
                    game_info['spread'] = odds[0].get('details')
                    game_info['over_under'] = odds[0].get('overUnder')
                    
            games_data.append(game_info)
            
        return pd.DataFrame(games_data)
        
    async def get_player_projections(self, week: int, season: int = None, 
                                   position: str = None) -> pd.DataFrame:
        """
        Get player projections from ESPN (if available in public API)
        
        Args:
            week: Week number
            season: Season year
            position: Filter by position
            
        Returns:
            DataFrame with projections
        """
        # Note: ESPN projections might not be available in public API
        # This is a placeholder for if/when they become available
        logger.warning("ESPN projections may not be available in public API")
        return pd.DataFrame()


# Example usage
async def test_espn_client():
    """Test the ESPN public client"""
    async with ESPNPublicClient() as client:
        # Get team mapping
        team_mapping = client.get_espn_team_id_mapping()
        
        # Test getting team schedule
        print("Testing ESPN Public API...")
        
        # Get Packers schedule
        schedule = await client.get_team_schedule(team_mapping['GB'], 2024)
        if schedule:
            print(f"Retrieved Green Bay Packers 2024 schedule")
            
        # Get week 1 scoreboard
        scoreboard = await client.get_scoreboard(1, 2024)
        if scoreboard:
            games_df = await client.get_week_games_with_odds(1, 2024)
            print(f"\nWeek 1 games: {len(games_df)}")
            if not games_df.empty:
                print(games_df[['home_team', 'away_team', 'spread', 'over_under']].head())
                
        # Get player info (example: Aaron Rodgers)
        player_info = await client.get_player_info(3118)  # ESPN ID for Aaron Rodgers
        if player_info:
            print(f"\nPlayer info retrieved for: {player_info.get('displayName')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_espn_client())