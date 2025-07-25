"""
Enhanced NFL API Client with Integrated Rate Limiting and Advanced Error Handling.
Location: src/fantasy_ai/core/data/sources/nfl_comprehensive.py
"""

import aiohttp
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..rate_limiter import AdaptiveRateLimiter, RateLimitedContext, create_nfl_api_rate_limiter
from ..storage.models import Team, Player, WeeklyStats
from ..storage.simple_database import get_simple_session

logger = logging.getLogger(__name__)

@dataclass 
class APIResponse:
    """Structured API response with metadata."""
    data: Any
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    cached: bool = False

class NFLAPIError(Exception):
    """Custom exception for NFL API errors."""
    pass

class NFLAPIClient:
    """
    Enhanced NFL API client with intelligent rate limiting, caching,
    error handling, and performance monitoring.
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limiter: Optional[AdaptiveRateLimiter] = None):
        """Initialize NFL API client with configuration."""
        
        # API Configuration
        self.api_key = api_key or os.getenv('NFL_API_KEY')
        if not self.api_key:
            raise ValueError("NFL_API_KEY environment variable is required")
        
        self.base_url = "https://v1.american-football.api-sports.io"
        # Note: API-Sports may not use league_id in the same way
        self.league_identifier = "nfl"  # Changed from league_id to string identifier
        self.available_seasons = [2021, 2022, 2023, 2024]  # Updated available seasons
        
        # Rate limiting
        self.rate_limiter = rate_limiter or create_nfl_api_rate_limiter()
        
        # HTTP client configuration - FIXED: Use correct header for API-Sports
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.headers = {
            'x-apisports-key': self.api_key,  # FIXED: Correct header name
            'User-Agent': 'FantasyFootballAI/1.0'
        }
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Caching
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cached_responses': 0,
            'avg_response_time': 0.0,
            'last_request_time': None
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure HTTP session exists."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
                connector=connector
            )

    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            
        # Log final statistics
        logger.info(f"NFL API Client Statistics: {self.get_stats()}")

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        sorted_params = sorted(params.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        return f"{endpoint}?{param_str}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        age = time.time() - cache_entry['timestamp']
        return age < self.cache_ttl

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> APIResponse:
        """Make HTTP request with rate limiting and error handling."""
        
        await self._ensure_session()
        
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            logger.debug(f"Cache hit for {endpoint}")
            self.stats['cached_responses'] += 1
            cached_data = self._cache[cache_key]['data']
            return APIResponse(
                data=cached_data,
                status_code=200,
                response_time=0.0,
                success=True,
                cached=True
            )
        
        # Use rate-limited context for the request
        async with RateLimitedContext(self.rate_limiter, f"NFL API {endpoint}"):
            start_time = time.time()
            
            try:
                url = f"{self.base_url}/{endpoint}"
                
                logger.debug(f"Making request to {endpoint} with params: {params}")
                
                async with self._session.get(url, params=params) as response:
                    response_time = time.time() - start_time
                    response_text = await response.text()
                    
                    # Update statistics
                    self.stats['total_requests'] += 1
                    self.stats['last_request_time'] = datetime.now(timezone.utc)
                    
                    # Calculate rolling average response time
                    total_requests = self.stats['total_requests']
                    current_avg = self.stats['avg_response_time']
                    self.stats['avg_response_time'] = (
                        (current_avg * (total_requests - 1) + response_time) / total_requests
                    )
                    
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            
                            # Validate API response structure
                            if not self._validate_api_response(data):
                                logger.warning(f"Invalid API response structure: {data}")
                                # Don't raise error, just log and continue
                            
                            # Cache successful response
                            self._cache[cache_key] = {
                                'data': data,
                                'timestamp': time.time()
                            }
                            
                            self.stats['successful_requests'] += 1
                            
                            return APIResponse(
                                data=data,
                                status_code=response.status,
                                response_time=response_time,
                                success=True
                            )
                            
                        except json.JSONDecodeError as e:
                            error_msg = f"JSON decode error: {e}"
                            logger.error(f"{error_msg}. Response: {response_text[:500]}")
                            raise NFLAPIError(error_msg)
                    
                    else:
                        error_msg = f"HTTP {response.status}: {response_text}"
                        logger.error(f"API request failed: {error_msg}")
                        
                        return APIResponse(
                            data=None,
                            status_code=response.status,
                            response_time=response_time,
                            success=False,
                            error_message=error_msg
                        )
            
            except asyncio.TimeoutError:
                error_msg = "Request timeout"
                logger.error(error_msg)
                return APIResponse(
                    data=None,
                    status_code=408,
                    response_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )
            
            except aiohttp.ClientError as e:
                error_msg = f"HTTP client error: {e}"
                logger.error(error_msg)
                return APIResponse(
                    data=None,
                    status_code=0,
                    response_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )

    def _validate_api_response(self, data: Dict[str, Any]) -> bool:
        """Validate API response has expected structure."""
        
        # Check for required fields
        if not isinstance(data, dict):
            return False
        
        # API-Sports standard response structure
        required_fields = ['get', 'response']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field in API response: {field}")
                return False
        
        return True

    async def get_teams(self) -> List[Dict[str, Any]]:
        """Get all NFL teams."""
        
        logger.info("Fetching NFL teams")
        
        # FIXED: API-Sports requires both league and season parameters
        response = await self._make_request('teams', {
            'league': 1,
            'season': 2023  # Use 2023 as it has data
        })
        
        if not response.success:
            raise NFLAPIError(f"Failed to fetch teams: {response.error_message}")
        
        teams_data = response.data.get('response', [])
        logger.info(f"Retrieved {len(teams_data)} NFL teams")
        
        return teams_data

    async def get_team_info(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific team."""
        
        logger.debug(f"Fetching team info for team ID: {team_id}")
        
        response = await self._make_request('teams', {'id': team_id})
        
        if not response.success:
            logger.error(f"Failed to fetch team info for {team_id}: {response.error_message}")
            return None
        
        teams_data = response.data.get('response', [])
        return teams_data[0] if teams_data else None

    async def get_players_by_team(self, team_id: int, season: int) -> List[Dict[str, Any]]:
        """Get all players for a specific team and season."""
        
        if season not in self.available_seasons:
            logger.warning(f"Season {season} may not be available. Available: {self.available_seasons}")
        
        logger.info(f"Fetching players for team {team_id}, season {season}")
        
        response = await self._make_request('players', {
            'team': team_id,
            'season': season
        })
        
        if not response.success:
            logger.error(f"Failed to fetch players for team {team_id}, season {season}: {response.error_message}")
            return []
        
        players_data = response.data.get('response', [])
        logger.info(f"Retrieved {len(players_data)} players for team {team_id}, season {season}")
        
        return players_data

    async def get_player_info(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific player."""
        
        logger.debug(f"Fetching player info for player ID: {player_id}")
        
        response = await self._make_request('players', {'id': player_id})
        
        if not response.success:
            logger.error(f"Failed to fetch player info for {player_id}: {response.error_message}")
            return None
        
        players_data = response.data.get('response', [])
        return players_data[0] if players_data else None

    async def get_player_stats(self, player_id: int, season: int, week: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get player statistics for a specific season/week."""
        
        if season not in self.available_seasons:
            logger.warning(f"Season {season} may not be available")
        
        params = {
            'player': player_id,
            'season': season
        }
        
        if week:
            params['week'] = week
        
        logger.debug(f"Fetching player stats for player {player_id}, season {season}, week {week}")
        
        response = await self._make_request('players/statistics', params)
        
        if not response.success:
            logger.error(f"Failed to fetch player stats: {response.error_message}")
            return None
        
        stats_data = response.data.get('response', [])
        
        if not stats_data:
            logger.debug(f"No stats found for player {player_id}, season {season}, week {week}")
            return None
        
        # Return the first (and usually only) result
        return stats_data[0]

    async def get_games_by_week(self, season: int, week: int) -> List[Dict[str, Any]]:
        """Get all games for a specific season and week."""
        
        if season not in self.available_seasons:
            logger.warning(f"Season {season} may not be available")
        
        logger.info(f"Fetching games for season {season}, week {week}")
        
        response = await self._make_request('games', {
            'season': season,
            'week': week
        })
        
        if not response.success:
            logger.error(f"Failed to fetch games for season {season}, week {week}: {response.error_message}")
            return []
        
        games_data = response.data.get('response', [])
        logger.info(f"Retrieved {len(games_data)} games for season {season}, week {week}")
        
        return games_data

    async def get_standings(self, season: int) -> List[Dict[str, Any]]:
        """Get league standings for a season."""
        
        if season not in self.available_seasons:
            logger.warning(f"Season {season} may not be available")
        
        logger.info(f"Fetching standings for season {season}")
        
        response = await self._make_request('standings', {'season': season})
        
        if not response.success:
            logger.error(f"Failed to fetch standings for season {season}: {response.error_message}")
            return []
        
        standings_data = response.data.get('response', [])
        logger.info(f"Retrieved standings for season {season}")
        
        return standings_data

    async def validate_api_access(self) -> bool:
        """Validate API access and check rate limits."""
        
        logger.info("Validating NFL API access")
        
        try:
            # FIXED: Test with teams endpoint using correct parameters
            response = await self._make_request('teams', {
                'league': 1,
                'season': 2023  # Use 2023 as it has data
            })
            
            if response.success:
                # Check if we got team data (which means API is working)
                teams = response.data.get('response', [])
                
                if teams and len(teams) > 0:
                    logger.info("NFL API access validated successfully")
                    logger.info(f"Available seasons: {self.available_seasons}")
                    logger.info(f"Found {len(teams)} teams in response")
                    return True
                else:
                    logger.error("No teams found in API response")
                    return False
            else:
                logger.error(f"API validation failed: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating API access: {e}")
            return False

    async def bulk_collect_team_players(self, team_ids: List[int], season: int) -> Dict[int, List[Dict[str, Any]]]:
        """Efficiently collect players for multiple teams."""
        
        logger.info(f"Bulk collecting players for {len(team_ids)} teams, season {season}")
        
        results = {}
        
        # Process teams in small batches to respect rate limits
        batch_size = 3  # Conservative batch size
        
        for i in range(0, len(team_ids), batch_size):
            batch = team_ids[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.get_players_by_team(team_id, season) for team_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for team_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error collecting players for team {team_id}: {result}")
                    results[team_id] = []
                else:
                    results[team_id] = result
            
            # Brief pause between batches
            if i + batch_size < len(team_ids):
                await asyncio.sleep(1)
        
        total_players = sum(len(players) for players in results.values())
        logger.info(f"Bulk collection completed: {total_players} players from {len(team_ids)} teams")
        
        return results

    async def collect_weekly_stats_batch(self, player_ids: List[int], season: int, week: int) -> Dict[int, Optional[Dict[str, Any]]]:
        """Efficiently collect weekly stats for multiple players."""
        
        logger.info(f"Collecting weekly stats for {len(player_ids)} players, season {season}, week {week}")
        
        results = {}
        
        # Process in small batches to respect rate limits
        batch_size = 2  # Very conservative for stats collection
        
        for i in range(0, len(player_ids), batch_size):
            batch = player_ids[i:i + batch_size]
            
            # Process batch with delay between requests
            for player_id in batch:
                try:
                    stats = await self.get_player_stats(player_id, season, week)
                    results[player_id] = stats
                    
                    # Small delay between individual requests in batch
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error collecting stats for player {player_id}: {e}")
                    results[player_id] = None
            
            # Longer pause between batches
            if i + batch_size < len(player_ids):
                logger.debug(f"Completed batch {i//batch_size + 1}/{(len(player_ids) + batch_size - 1)//batch_size}")
                await asyncio.sleep(2)
        
        successful_collections = sum(1 for result in results.values() if result is not None)
        logger.info(f"Weekly stats collection completed: {successful_collections}/{len(player_ids)} successful")
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        
        stats = self.stats.copy()
        
        # Add rate limiter stats
        try:
            stats['rate_limiter'] = self.rate_limiter.get_status()
        except Exception as e:
            logger.error(f"Error getting rate limiter stats: {e}")
            stats['rate_limiter'] = {"error": str(e)}
        
        # Calculate success rate
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['cache_hit_rate'] = stats['cached_responses'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("API response cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        start_time = time.time()
        
        health_status = {
            'api_accessible': False,
            'response_time': 0.0,
            'rate_limit_status': {},
            'cache_size': len(self._cache),
            'session_status': 'unknown',
            'last_error': None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Get rate limiter status safely
            try:
                health_status['rate_limit_status'] = self.rate_limiter.get_status()
            except Exception as e:
                health_status['rate_limit_status'] = {"error": str(e)}
            
            # Test API accessibility
            health_status['api_accessible'] = await self.validate_api_access()
            health_status['response_time'] = time.time() - start_time
            
            # Check session status
            if self._session and not self._session.closed:
                health_status['session_status'] = 'active'
            else:
                health_status['session_status'] = 'inactive'
            
        except Exception as e:
            health_status['last_error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status

# Utility functions
async def create_nfl_client(api_key: Optional[str] = None) -> NFLAPIClient:
    """Create and initialize NFL API client."""
    
    client = NFLAPIClient(api_key)
    await client._ensure_session()
    
    # Validate access
    if not await client.validate_api_access():
        await client.close()
        raise NFLAPIError("Failed to validate NFL API access")
    
    return client

# FIXED: Updated sync_teams_to_database function with proper NULL handling and filtering
async def sync_teams_to_database(nfl_client) -> int:
    """Sync NFL teams from API to database."""
    
    logger.info("Syncing NFL teams to database")
    
    try:
        # Get teams from API
        logger.info("Fetching NFL teams")
        teams_data = await nfl_client.get_teams()
        logger.info(f"Retrieved {len(teams_data)} NFL teams")
        
        # DEBUG: Print what we actually get (remove this after fixing)
        if teams_data and len(teams_data) > 0:
            logger.info(f"DEBUG: First team structure: {teams_data[0]}")
        
        teams_synced = 0
        
        async with get_simple_session() as session:
            from sqlalchemy import select
            
            for team_data in teams_data:
                try:
                    # FIXED: Map API fields correctly and handle NULL values
                    api_id = team_data.get('id')
                    name = team_data.get('name', '')
                    code = team_data.get('code') or 'UNK'  # Provide default for NULL codes
                    city = team_data.get('city') or ''
                    logo = team_data.get('logo', '')
                    coach = team_data.get('coach') or ''
                    stadium = team_data.get('stadium') or ''
                    
                    # Handle conference/division (not in current API response)
                    conference = None  # Not available in current API
                    division = None    # Not available in current API
                    
                    # FIXED: Skip teams that are clearly not actual NFL teams
                    # AFC/NFC are conference entities, not teams
                    if name in ['AFC', 'NFC'] or not api_id or not name:
                        logger.debug(f"Skipping non-team entity: {name} (id: {api_id})")
                        continue
                    
                    # Additional validation - skip if it looks like a conference/all-star team
                    if any(keyword in name.upper() for keyword in ['CONFERENCE', 'ALL-STAR', 'PRO BOWL']):
                        logger.debug(f"Skipping conference/special team: {name}")
                        continue
                    
                    # Check if team already exists
                    result = await session.execute(
                        select(Team).where(Team.api_id == api_id)
                    )
                    existing_team = result.scalar_one_or_none()
                    
                    if existing_team:
                        # Update existing team
                        existing_team.name = name
                        existing_team.code = code
                        existing_team.city = city
                        existing_team.logo = logo
                        existing_team.conference = conference
                        existing_team.division = division
                        existing_team.stadium = stadium
                        existing_team.coach = coach
                        existing_team.updated_at = datetime.now(timezone.utc)
                        
                        logger.debug(f"Updated team: {name}")
                    else:
                        # Create new team with all required fields
                        new_team = Team(
                            api_id=api_id,
                            name=name,
                            code=code,  # Now guaranteed to not be None
                            city=city,
                            logo=logo,
                            conference=conference,
                            division=division,
                            stadium=stadium,
                            coach=coach
                        )
                        session.add(new_team)
                        logger.debug(f"Added new team: {name}")
                    
                    teams_synced += 1
                    
                except Exception as e:
                    logger.error(f"Error processing team {team_data}: {e}")
                    # Continue processing other teams instead of failing completely
                    continue
            
            # Commit all changes
            try:
                await session.commit()
                logger.info(f"Successfully synced {teams_synced} teams to database")
            except Exception as commit_error:
                logger.error(f"Error committing team changes: {commit_error}")
                await session.rollback()
                raise
            
    except Exception as e:
        logger.error(f"Error syncing teams to database: {e}")
        raise
    
    return teams_synced

# Helper function for player data processing
async def process_player_data(player_data):
    """Process individual player data from API response."""
    
    # The API returns player data directly (not nested under 'player' key)
    # Based on debug output: {"id": 3, "name": "Ameer Abdullah", "position": "RB", ...}
    
    return {
        'api_id': player_data.get('id'),  # API field is 'id'
        'name': player_data.get('name', ''),
        'position': player_data.get('position', 'OTHER'),
        'number': player_data.get('number'),
        'age': player_data.get('age'),
        'height': player_data.get('height', ''),
        'weight': player_data.get('weight', ''),
        'college': player_data.get('college', ''),
        'experience': player_data.get('experience'),
        # Add any other fields you need
    }

# Configuration helper
def get_api_config() -> Dict[str, Any]:
    """Get NFL API configuration from environment."""
    
    return {
        'api_key': os.getenv('NFL_API_KEY'),
        'rate_limit_requests_per_day': int(os.getenv('NFL_API_RATE_LIMIT', '100')),
        'cache_ttl': int(os.getenv('NFL_API_CACHE_TTL', '3600')),
        'timeout_seconds': int(os.getenv('NFL_API_TIMEOUT', '30')),
        'available_seasons': [2021, 2022, 2023, 2024]
    }