# src/data/nfl_api_manager.py
"""
Production-ready NFL Data API Manager
Handles real-time data ingestion from multiple sources with caching,
error handling, and data validation for fantasy football predictions.
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
import redis
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import time
from contextlib import asynccontextmanager
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerStats:
    """Structured player statistics"""
    player_id: str
    name: str
    position: str
    team: str
    week: int
    season: int
    fantasy_points: float
    fantasy_points_ppr: float
    passing_yards: Optional[int] = None
    passing_tds: Optional[int] = None
    rushing_yards: Optional[int] = None
    rushing_tds: Optional[int] = None
    receiving_yards: Optional[int] = None
    receiving_tds: Optional[int] = None
    receptions: Optional[int] = None
    targets: Optional[int] = None
    snap_count: Optional[int] = None
    snap_percentage: Optional[float] = None
    red_zone_targets: Optional[int] = None
    air_yards: Optional[int] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class TeamMatchup:
    """Team matchup information"""
    team: str
    opponent: str
    week: int
    season: int
    home_away: str
    spread: Optional[float] = None
    total: Optional[float] = None
    weather: Optional[Dict] = None
    injury_report: Optional[List] = None

@dataclass
class PlayerNews:
    """Player news and updates"""
    player_id: str
    headline: str
    description: str
    impact: str  # HIGH, MEDIUM, LOW
    timestamp: datetime
    source: str

class DataSourceInterface(ABC):
    """Abstract interface for data sources"""
    
    @abstractmethod
    async def get_players(self) -> List[Dict]:
        pass
    
    @abstractmethod
    async def get_player_stats(self, player_id: str, week: Optional[int] = None) -> Dict:
        pass
    
    @abstractmethod
    async def get_matchups(self, week: int) -> List[Dict]:
        pass

class SleeperAPI(DataSourceInterface):
    """Sleeper API implementation - Primary source for player data"""
    
    def __init__(self):
        self.base_url = "https://api.sleeper.app/v1"
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Sleeper API error {response.status}: {url}")
                    return {}
        except Exception as e:
            logger.error(f"Sleeper API request failed: {e}")
            return {}
    
    async def get_current_state(self) -> Dict:
        """Get current NFL season state"""
        return await self._make_request("state/nfl")
    
    async def get_players(self) -> List[Dict]:
        """Get all NFL players"""
        players_data = await self._make_request("players/nfl")
        
        # Convert to list format
        players = []
        for player_id, player_info in players_data.items():
            if player_info.get('position') in ['QB', 'RB', 'WR', 'TE']:
                player_info['player_id'] = player_id
                players.append(player_info)
        
        return players
    
    async def get_player_stats(self, player_id: str, week: Optional[int] = None) -> Dict:
        """Get player stats for specific week or season"""
        state = await self.get_current_state()
        season = state.get('season', '2024')
        
        if week:
            endpoint = f"stats/nfl/regular/{season}/{week}"
        else:
            endpoint = f"stats/nfl/regular/{season}"
        
        stats_data = await self._make_request(endpoint)
        return stats_data.get(player_id, {})
    
    async def get_matchups(self, week: int) -> List[Dict]:
        """Get matchup data for specific week"""
        state = await self.get_current_state()
        season = state.get('season', '2024')
        
        matchups = await self._make_request(f"state/nfl/regular/{season}/{week}")
        return matchups

class ESPNAPI(DataSourceInterface):
    """ESPN API implementation - Secondary source for enhanced data"""
    
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str) -> Dict:
        """Make ESPN API request"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"ESPN API error {response.status}: {url}")
                    return {}
        except Exception as e:
            logger.error(f"ESPN API request failed: {e}")
            return {}
    
    async def get_players(self) -> List[Dict]:
        """ESPN doesn't provide comprehensive player list"""
        return []
    
    async def get_player_stats(self, player_id: str, week: Optional[int] = None) -> Dict:
        """Get player stats from ESPN"""
        # ESPN player stats require different approach
        return {}
    
    async def get_matchups(self, week: int) -> List[Dict]:
        """Get matchup data from ESPN"""
        scoreboard = await self._make_request("scoreboard")
        matchups = []
        
        for event in scoreboard.get('events', []):
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            if len(competitors) == 2:
                matchup = {
                    'home_team': competitors[0].get('team', {}).get('abbreviation'),
                    'away_team': competitors[1].get('team', {}).get('abbreviation'),
                    'week': week,
                    'odds': competition.get('odds', [{}])[0] if competition.get('odds') else {}
                }
                matchups.append(matchup)
        
        return matchups

class NFLDataManager:
    """
    Main NFL Data Manager
    Orchestrates multiple data sources with caching and validation
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 cache_ttl: Dict[str, int] = None):
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Default cache TTL values
        self.cache_ttl = cache_ttl or {
            'players': 3600,      # 1 hour
            'stats': 1800,        # 30 minutes
            'matchups': 3600,     # 1 hour
            'news': 900,          # 15 minutes
            'current_week': 3600  # 1 hour
        }
        
        # Data sources
        self.sleeper_api = SleeperAPI()
        self.espn_api = ESPNAPI()
        
        # Validation thresholds
        self.min_players_threshold = 1000
        self.max_fantasy_points = 60.0
        
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache key"""
        key_data = f"{prefix}:{':'.join(map(str, args))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get data from Redis cache"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read failed for key {cache_key}: {e}")
        return None
    
    async def _set_cached_data(self, cache_key: str, data: Dict, ttl: int):
        """Set data in Redis cache"""
        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache write failed for key {cache_key}: {e}")
    
    def _validate_player_stats(self, stats: Dict) -> bool:
        """Validate player statistics data"""
        try:
            # Check for required fields
            required_fields = ['fantasy_points', 'week', 'season']
            for field in required_fields:
                if field not in stats:
                    return False
            
            # Check reasonable ranges
            fantasy_points = stats.get('fantasy_points', 0)
            if fantasy_points < 0 or fantasy_points > self.max_fantasy_points:
                return False
            
            return True
        except Exception:
            return False
    
    async def get_current_week(self) -> int:
        """Get current NFL week with caching"""
        cache_key = self._generate_cache_key('current_week')
        
        # Check cache
        cached_week = await self._get_cached_data(cache_key)
        if cached_week:
            return cached_week.get('week', 1)
        
        try:
            async with self.sleeper_api as sleeper:
                state = await sleeper.get_current_state()
                current_week = state.get('week', 1)
                
                # Cache the result
                await self._set_cached_data(
                    cache_key,
                    {'week': current_week, 'updated': datetime.now().isoformat()},
                    self.cache_ttl['current_week']
                )
                
                return current_week
        except Exception as e:
            logger.error(f"Failed to get current week: {e}")
            return 1  # Fallback
    
    async def get_all_players(self) -> pd.DataFrame:
        """Get all NFL players with caching"""
        cache_key = self._generate_cache_key('all_players')
        
        # Check cache
        cached_players = await self._get_cached_data(cache_key)
        if cached_players:
            return pd.DataFrame(cached_players)
        
        try:
            async with self.sleeper_api as sleeper:
                players_data = await sleeper.get_players()
                
                # Validate data
                if len(players_data) < self.min_players_threshold:
                    logger.warning(f"Insufficient players data: {len(players_data)}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                players_df = pd.DataFrame(players_data)
                
                # Cache the result
                await self._set_cached_data(
                    cache_key,
                    players_data,
                    self.cache_ttl['players']
                )
                
                logger.info(f"Loaded {len(players_df)} players from Sleeper API")
                return players_df
                
        except Exception as e:
            logger.error(f"Failed to get players: {e}")
            return pd.DataFrame()
    
    async def get_player_stats(self, 
                             player_id: str, 
                             week: Optional[int] = None,
                             season: int = 2024) -> PlayerStats:
        """Get comprehensive player statistics"""
        cache_key = self._generate_cache_key('player_stats', player_id, week or 'season', season)
        
        # Check cache
        cached_stats = await self._get_cached_data(cache_key)
        if cached_stats:
            cached_stats['last_updated'] = datetime.fromisoformat(cached_stats['last_updated'])
            return PlayerStats(**cached_stats)
        
        try:
            async with self.sleeper_api as sleeper:
                # Get player info
                players_data = await sleeper.get_players()
                player_info = None
                for pid, info in players_data.items():
                    if pid == player_id:
                        player_info = info
                        break
                
                if not player_info:
                    raise ValueError(f"Player {player_id} not found")
                
                # Get stats
                stats_data = await sleeper.get_player_stats(player_id, week)
                
                # Create PlayerStats object
                player_stats = PlayerStats(
                    player_id=player_id,
                    name=f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}",
                    position=player_info.get('position', ''),
                    team=player_info.get('team', ''),
                    week=week or 0,
                    season=season,
                    fantasy_points=stats_data.get('pts_std', 0.0),
                    fantasy_points_ppr=stats_data.get('pts_ppr', 0.0),
                    passing_yards=stats_data.get('pass_yd'),
                    passing_tds=stats_data.get('pass_td'),
                    rushing_yards=stats_data.get('rush_yd'),
                    rushing_tds=stats_data.get('rush_td'),
                    receiving_yards=stats_data.get('rec_yd'),
                    receiving_tds=stats_data.get('rec_td'),
                    receptions=stats_data.get('rec'),
                    targets=stats_data.get('rec_tgt')
                )
                
                # Cache the result
                stats_dict = asdict(player_stats)
                stats_dict['last_updated'] = stats_dict['last_updated'].isoformat()
                await self._set_cached_data(
                    cache_key,
                    stats_dict,
                    self.cache_ttl['stats']
                )
                
                return player_stats
                
        except Exception as e:
            logger.error(f"Failed to get player stats for {player_id}: {e}")
            # Return default stats
            return PlayerStats(
                player_id=player_id,
                name="Unknown Player",
                position="",
                team="",
                week=week or 0,
                season=season,
                fantasy_points=0.0,
                fantasy_points_ppr=0.0
            )
    
    async def get_multiple_player_stats(self, 
                                      player_ids: List[str], 
                                      week: Optional[int] = None) -> List[PlayerStats]:
        """Get stats for multiple players efficiently"""
        tasks = [self.get_player_stats(pid, week) for pid in player_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, PlayerStats):
                valid_results.append(result)
            else:
                logger.error(f"Player stats error: {result}")
        
        return valid_results
    
    async def get_season_stats_summary(self, player_id: str) -> Dict[str, float]:
        """Get season-long statistical summary"""
        cache_key = self._generate_cache_key('season_summary', player_id)
        
        cached_summary = await self._get_cached_data(cache_key)
        if cached_summary:
            return cached_summary
        
        try:
            current_week = await self.get_current_week()
            
            # Get stats for all weeks
            weekly_stats = []
            for week in range(1, current_week):
                stats = await self.get_player_stats(player_id, week)
                if stats.fantasy_points_ppr > 0:  # Only include weeks with data
                    weekly_stats.append(stats.fantasy_points_ppr)
            
            if not weekly_stats:
                return {}
            
            # Calculate summary statistics
            summary = {
                'games_played': len(weekly_stats),
                'total_points': sum(weekly_stats),
                'avg_points': np.mean(weekly_stats),
                'std_points': np.std(weekly_stats),
                'max_points': max(weekly_stats),
                'min_points': min(weekly_stats),
                'consistency_score': np.mean(weekly_stats) / max(np.std(weekly_stats), 0.1),
                'boom_weeks': sum(1 for points in weekly_stats if points > (np.mean(weekly_stats) + np.std(weekly_stats))),
                'bust_weeks': sum(1 for points in weekly_stats if points < (np.mean(weekly_stats) - np.std(weekly_stats) * 0.5)),
                'recent_form': np.mean(weekly_stats[-4:]) if len(weekly_stats) >= 4 else np.mean(weekly_stats)
            }
            
            # Cache the summary
            await self._set_cached_data(cache_key, summary, self.cache_ttl['stats'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get season summary for {player_id}: {e}")
            return {}
    
    async def get_matchup_data(self, team: str, week: int) -> TeamMatchup:
        """Get team matchup information"""
        cache_key = self._generate_cache_key('matchup', team, week)
        
        cached_matchup = await self._get_cached_data(cache_key)
        if cached_matchup:
            return TeamMatchup(**cached_matchup)
        
        try:
            # Get matchups from multiple sources
            async with self.espn_api as espn:
                espn_matchups = await espn.get_matchups(week)
            
            # Find team's matchup
            opponent = None
            home_away = 'home'
            
            for matchup in espn_matchups:
                if matchup.get('home_team') == team:
                    opponent = matchup.get('away_team')
                    home_away = 'home'
                    break
                elif matchup.get('away_team') == team:
                    opponent = matchup.get('home_team')
                    home_away = 'away'
                    break
            
            if not opponent:
                raise ValueError(f"No matchup found for {team} in week {week}")
            
            team_matchup = TeamMatchup(
                team=team,
                opponent=opponent,
                week=week,
                season=2024,
                home_away=home_away
            )
            
            # Cache the result
            matchup_dict = asdict(team_matchup)
            await self._set_cached_data(
                cache_key,
                matchup_dict,
                self.cache_ttl['matchups']
            )
            
            return team_matchup
            
        except Exception as e:
            logger.error(f"Failed to get matchup for {team}: {e}")
            return TeamMatchup(
                team=team,
                opponent="TBD",
                week=week,
                season=2024,
                home_away='home'
            )
    
    async def get_trending_players(self, 
                                 position: Optional[str] = None,
                                 limit: int = 20) -> List[Dict]:
        """Get trending players based on recent performance"""
        cache_key = self._generate_cache_key('trending', position or 'all', limit)
        
        cached_trending = await self._get_cached_data(cache_key)
        if cached_trending:
            return cached_trending
        
        try:
            # Get all players
            players_df = await self.get_all_players()
            
            if position:
                players_df = players_df[players_df['position'] == position]
            
            # Calculate trending scores
            trending_scores = []
            current_week = await self.get_current_week()
            
            for _, player in players_df.head(100).iterrows():  # Limit for performance
                try:
                    # Get recent stats
                    recent_weeks = []
                    for week in range(max(1, current_week - 4), current_week):
                        stats = await self.get_player_stats(player['player_id'], week)
                        if stats.fantasy_points_ppr > 0:
                            recent_weeks.append(stats.fantasy_points_ppr)
                    
                    if len(recent_weeks) >= 2:
                        # Calculate trend (simple linear regression slope)
                        x = np.arange(len(recent_weeks))
                        trend = np.polyfit(x, recent_weeks, 1)[0]
                        avg_points = np.mean(recent_weeks)
                        
                        trending_scores.append({
                            'player_id': player['player_id'],
                            'name': f"{player.get('first_name', '')} {player.get('last_name', '')}",
                            'position': player.get('position', ''),
                            'team': player.get('team', ''),
                            'trend_score': trend,
                            'avg_points': avg_points,
                            'recent_weeks': len(recent_weeks)
                        })
                except Exception as e:
                    continue
            
            # Sort by trend score and return top performers
            trending_scores.sort(key=lambda x: x['trend_score'], reverse=True)
            top_trending = trending_scores[:limit]
            
            # Cache the result
            await self._set_cached_data(
                cache_key,
                top_trending,
                self.cache_ttl['stats']
            )
            
            return top_trending
            
        except Exception as e:
            logger.error(f"Failed to get trending players: {e}")
            return []
    
    async def health_check(self) -> Dict[str, any]:
        """Comprehensive health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'redis_connection': False,
            'sleeper_api': False,
            'espn_api': False,
            'data_freshness': {},
            'cache_stats': {}
        }
        
        # Test Redis connection
        try:
            health_status['redis_connection'] = self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Test Sleeper API
        try:
            async with self.sleeper_api as sleeper:
                state = await sleeper.get_current_state()
                health_status['sleeper_api'] = bool(state)
        except Exception as e:
            logger.error(f"Sleeper API health check failed: {e}")
        
        # Test ESPN API
        try:
            async with self.espn_api as espn:
                scoreboard = await espn._make_request("scoreboard")
                health_status['espn_api'] = bool(scoreboard)
        except Exception as e:
            logger.error(f"ESPN API health check failed: {e}")
        
        # Cache statistics
        try:
            cache_info = self.redis_client.info('memory')
            health_status['cache_stats'] = {
                'used_memory': cache_info.get('used_memory_human'),
                'keyspace_hits': cache_info.get('keyspace_hits', 0),
                'keyspace_misses': cache_info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
        
        return health_status
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        try:
            # This is a simplified cleanup - in production you'd want more sophisticated logic
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Get all keys (be careful in production with large key spaces)
            keys = self.redis_client.keys("*")
            
            deleted_count = 0
            for key in keys:
                try:
                    # Try to get the data and check timestamp
                    data = await self._get_cached_data(key)
                    if data and 'last_updated' in data:
                        update_time = datetime.fromisoformat(data['last_updated'])
                        if update_time < cutoff_time:
                            self.redis_client.delete(key)
                            deleted_count += 1
                except Exception:
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old cache entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0