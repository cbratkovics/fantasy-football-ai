"""
Sleeper API Client with Production-Ready Features
- Async/await for performance
- Comprehensive error handling
- Rate limiting (1000 requests/minute)
- Redis caching
- Type hints for better code quality
- Logging for debugging
"""

import asyncio
import aiohttp
import redis
import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry
import backoff
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Player:
    """Player data model with type safety"""
    player_id: str
    first_name: str
    last_name: str
    position: str
    team: Optional[str]
    fantasy_positions: List[str]
    age: Optional[int]
    years_exp: Optional[int]
    status: str
    metadata: Dict[str, Any]

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @classmethod
    def from_api(cls, player_id: str, data: Dict[str, Any]) -> 'Player':
        """Factory method to create Player from API response"""
        return cls(
            player_id=player_id,
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            position=data.get('position', ''),
            team=data.get('team'),
            fantasy_positions=data.get('fantasy_positions', []),
            age=data.get('age'),
            years_exp=data.get('years_exp'),
            status=data.get('status', 'Unknown'),
            metadata=data
        )


class SleeperAPIClient:
    """
    Production-ready Sleeper API client with caching and rate limiting
    
    Features:
    - Async HTTP requests for performance
    - Redis caching to minimize API calls
    - Rate limiting (1000 requests/minute)
    - Exponential backoff for reliability
    - Comprehensive error handling
    """
    
    BASE_URL = "https://api.sleeper.app/v1"
    RATE_LIMIT = 900  # Stay under 1000/minute limit
    
    def __init__(self, redis_host: Optional[str] = None, redis_port: Optional[int] = None):
        """Initialize API client with Redis connection"""
        # Get Redis connection from environment variable or use defaults
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        if redis_host is None or redis_port is None:
            # Parse the Redis URL
            parsed_url = urlparse(redis_url)
            redis_host = redis_host or parsed_url.hostname or 'localhost'
            redis_port = redis_port or parsed_url.port or 6379
        
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True
        )
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @sleep_and_retry
    @limits(calls=RATE_LIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and retry logic
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            JSON response as dictionary
            
        Raises:
            aiohttp.ClientError: On HTTP errors
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            async with self.session.get(url, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                logger.info(f"Successfully fetched {endpoint}")
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching {endpoint}: {str(e)}")
            raise
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        parts = [prefix] + [str(arg) for arg in args]
        return ":".join(parts)
    
    async def _get_cached_or_fetch(
        self, 
        cache_key: str, 
        endpoint: str, 
        ttl: int = 86400
    ) -> Dict[str, Any]:
        """
        Get data from cache or fetch from API
        
        Args:
            cache_key: Redis cache key
            endpoint: API endpoint if cache miss
            ttl: Time to live in seconds (default 24 hours)
            
        Returns:
            Cached or fresh data
        """
        # Check cache first
        cached = self.redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached)
        
        # Fetch from API on cache miss
        data = await self._make_request(endpoint)
        
        # Cache the result
        self.redis_client.setex(
            cache_key, 
            ttl, 
            json.dumps(data)
        )
        logger.info(f"Cached {cache_key} for {ttl} seconds")
        
        return data
    
    async def get_all_players(self, sport: str = 'nfl') -> Dict[str, Player]:
        """
        Fetch all players with 24-hour caching
        
        Args:
            sport: Sport type (default 'nfl')
            
        Returns:
            Dictionary mapping player_id to Player objects
        """
        cache_key = self._get_cache_key('players', sport)
        data = await self._get_cached_or_fetch(
            cache_key, 
            f"players/{sport}",
            ttl=86400  # 24 hours
        )
        
        # Convert to Player objects
        players = {}
        for player_id, player_data in data.items():
            try:
                players[player_id] = Player.from_api(player_id, player_data)
            except Exception as e:
                logger.warning(f"Error parsing player {player_id}: {str(e)}")
                
        return players
    
    async def get_nfl_state(self) -> Dict[str, Any]:
        """
        Get current NFL state (week, season, etc.)
        
        Returns:
            NFL state information
        """
        cache_key = self._get_cache_key('state', 'nfl')
        return await self._get_cached_or_fetch(
            cache_key,
            'state/nfl',
            ttl=3600  # 1 hour cache for state
        )
    
    async def get_league(self, league_id: str) -> Dict[str, Any]:
        """
        Fetch league information
        
        Args:
            league_id: Sleeper league ID
            
        Returns:
            League data
        """
        cache_key = self._get_cache_key('league', league_id)
        return await self._get_cached_or_fetch(
            cache_key,
            f"league/{league_id}",
            ttl=3600  # 1 hour cache
        )
    
    async def get_rosters(self, league_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all rosters in a league
        
        Args:
            league_id: Sleeper league ID
            
        Returns:
            List of roster data
        """
        cache_key = self._get_cache_key('rosters', league_id)
        return await self._get_cached_or_fetch(
            cache_key,
            f"league/{league_id}/rosters",
            ttl=1800  # 30 minute cache for active data
        )
    
    async def get_matchups(
        self, 
        league_id: str, 
        week: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch matchups for a specific week
        
        Args:
            league_id: Sleeper league ID
            week: Week number
            
        Returns:
            List of matchup data
        """
        cache_key = self._get_cache_key('matchups', league_id, week)
        
        # Cache completed weeks longer
        nfl_state = await self.get_nfl_state()
        current_week = nfl_state.get('week', 1)
        ttl = 86400 if week < current_week else 1800
        
        return await self._get_cached_or_fetch(
            cache_key,
            f"league/{league_id}/matchups/{week}",
            ttl=ttl
        )
    
    async def get_trending_players(
        self, 
        sport: str = 'nfl',
        type: str = 'add',
        lookback_hours: int = 24,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Get trending players based on add/drop activity
        
        Args:
            sport: Sport type
            type: 'add' or 'drop'
            lookback_hours: Hours to look back
            limit: Number of results
            
        Returns:
            List of trending players with counts
        """
        cache_key = self._get_cache_key(
            'trending', 
            sport, 
            type, 
            lookback_hours
        )
        
        endpoint = (
            f"players/{sport}/trending/{type}"
            f"?lookback_hours={lookback_hours}&limit={limit}"
        )
        
        return await self._get_cached_or_fetch(
            cache_key,
            endpoint,
            ttl=300  # 5 minute cache for trending
        )
    
    async def get_stats(
        self,
        sport: str = 'nfl',
        season_type: str = 'regular',
        season: str = '2023',
        week: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get player stats for a specific season/week
        
        Args:
            sport: Sport type (default 'nfl')
            season_type: 'regular', 'post', or 'pre'
            season: Year as string (e.g., '2023')
            week: Week number (optional, returns all weeks if not specified)
            
        Returns:
            Dict of player stats keyed by player_id
        """
        if week:
            endpoint = f"stats/{sport}/{season_type}/{season}/{week}"
        else:
            endpoint = f"stats/{sport}/{season_type}/{season}"
            
        cache_key = self._get_cache_key('stats', f"{sport}_{season_type}_{season}_{week or 'all'}")
        return await self._get_cached_or_fetch(
            cache_key,
            endpoint,
            ttl=86400  # 24 hour cache for historical stats
        )
    
    async def get_projections(
        self,
        sport: str = 'nfl',
        season_type: str = 'regular',
        season: str = '2023',
        week: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get player projections for a specific week
        
        Args:
            sport: Sport type
            season_type: Season type
            season: Year as string
            week: Week number
            
        Returns:
            Dict of player projections
        """
        endpoint = f"projections/{sport}/{season_type}/{season}/{week}"
        cache_key = self._get_cache_key('projections', f"{sport}_{season_type}_{season}_{week}")
        return await self._get_cached_or_fetch(
            cache_key,
            endpoint,
            ttl=3600  # 1 hour cache for projections
        )
    
    async def get_week_stats(
        self,
        season: str = '2023',
        week: int = 1,
        season_type: str = 'regular'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Alias for get_stats with week parameter for compatibility
        
        Args:
            season: Year as string
            week: Week number
            season_type: Season type
            
        Returns:
            Dict of player stats for the specified week
        """
        return await self.get_stats(
            sport='nfl',
            season_type=season_type,
            season=season,
            week=week
        )

    async def invalidate_cache(self, pattern: str = '*'):
        """
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Redis key pattern (default all)
        """
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries")


# Example usage and testing
async def main():
    """Example usage of the Sleeper API client"""
    async with SleeperAPIClient() as client:
        # Get all players
        players = await client.get_all_players()
        print(f"Loaded {len(players)} players")
        
        # Show a few examples
        qbs = [p for p in players.values() if p.position == 'QB'][:5]
        for qb in qbs:
            print(f"- {qb.full_name} ({qb.team})")
        
        # Get NFL state
        state = await client.get_nfl_state()
        print(f"\nNFL State: Week {state['week']} of {state['season']}")
        
        # Get trending players
        trending = await client.get_trending_players(limit=10)
        print(f"\nTop 10 trending adds:")
        for i, player in enumerate(trending, 1):
            player_id = player['player_id']
            if player_id in players:
                p = players[player_id]
                print(f"{i}. {p.full_name} - {player['count']} adds")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())