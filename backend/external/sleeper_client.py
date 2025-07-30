"""
Optimized Sleeper API Client with Intelligent Batching
High-performance client with request coalescing, caching, and rate limiting
"""
import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib

from backend.core.cache import cache, cached, CACHE_TTL_LONG, CACHE_TTL_SHORT

logger = logging.getLogger(__name__)

# Sleeper API constants
SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
SLEEPER_RATE_LIMIT = 1000  # requests per minute
SLEEPER_BATCH_SIZE = 100   # max players per batch request

# Request types for batching
class RequestType:
    PLAYER = "player"
    STATS = "stats"
    PROJECTIONS = "projections"
    LEAGUE = "league"
    ROSTER = "roster"
    MATCHUP = "matchup"


@dataclass
class BatchRequest:
    """Represents a batched request"""
    request_type: str
    params: Dict[str, Any]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher priority = processed first


class RateLimiter:
    """Token bucket rate limiter for Sleeper API"""
    
    def __init__(self, rate: int, burst: int = None):
        self.rate = rate  # requests per minute
        self.burst = burst or rate // 10  # burst capacity
        self.tokens = self.burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Acquire tokens for API request"""
        async with self.lock:
            now = time.time()
            # Add tokens based on time passed
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + (time_passed * self.rate / 60)
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # Calculate wait time
            wait_time = (tokens - self.tokens) * 60 / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = max(0, self.tokens - tokens)
            return True


class SleeperClient:
    """
    Optimized Sleeper API client with intelligent batching and caching
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(SLEEPER_RATE_LIMIT)
        
        # Batching queues
        self.batch_queues: Dict[str, List[BatchRequest]] = {
            RequestType.PLAYER: [],
            RequestType.STATS: [],
            RequestType.PROJECTIONS: [],
            RequestType.LEAGUE: [],
            RequestType.ROSTER: [],
            RequestType.MATCHUP: []
        }
        
        # Batch processing tasks
        self.batch_tasks: Dict[str, Optional[asyncio.Task]] = {}
        
        # Request deduplication
        self.pending_requests: Dict[str, List[asyncio.Future]] = {}
        
        # Performance metrics
        self.metrics = {
            "requests_sent": 0,
            "requests_cached": 0,
            "requests_batched": 0,
            "requests_deduplicated": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50, ttl_dns_cache=300),
            headers={"User-Agent": "FantasyFootballAI/1.0"}
        )
        
        # Start batch processors
        for request_type in self.batch_queues.keys():
            self.batch_tasks[request_type] = asyncio.create_task(
                self._process_batch_queue(request_type)
            )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cancel batch tasks
        for task in self.batch_tasks.values():
            if task:
                task.cancel()
        
        # Close session
        if self.session:
            await self.session.close()
    
    async def _process_batch_queue(self, request_type: str):
        """Process batched requests for a specific type"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                queue = self.batch_queues[request_type]
                if not queue:
                    continue
                
                # Determine batch size based on request type
                if request_type == RequestType.PLAYER:
                    batch_size = SLEEPER_BATCH_SIZE
                elif request_type == RequestType.STATS:
                    batch_size = 50  # Stats endpoint more expensive
                else:
                    batch_size = 20
                
                # Process batch if queue is full or oldest request is old enough
                should_process = (
                    len(queue) >= batch_size or
                    (queue and time.time() - queue[0].timestamp > 0.5)
                )
                
                if should_process:
                    # Take batch from queue
                    batch = queue[:batch_size]
                    del queue[:batch_size]
                    
                    # Sort by priority
                    batch.sort(key=lambda x: x.priority, reverse=True)
                    
                    # Process batch
                    await self._execute_batch(request_type, batch)
                    
            except Exception as e:
                logger.error(f"Batch processing error for {request_type}: {str(e)}")
    
    async def _execute_batch(self, request_type: str, batch: List[BatchRequest]):
        """Execute a batch of requests"""
        if not batch:
            return
        
        try:
            if request_type == RequestType.PLAYER:
                await self._execute_player_batch(batch)
            elif request_type == RequestType.STATS:
                await self._execute_stats_batch(batch)
            elif request_type == RequestType.PROJECTIONS:
                await self._execute_projections_batch(batch)
            else:
                # For non-batchable requests, execute individually
                for request in batch:
                    await self._execute_single_request(request)
                    
        except Exception as e:
            logger.error(f"Batch execution error: {str(e)}")
            # Set error for all futures in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _execute_player_batch(self, batch: List[BatchRequest]):
        """Execute batch player requests"""
        # Collect all player IDs
        player_ids = set()
        for request in batch:
            if "player_ids" in request.params:
                player_ids.update(request.params["player_ids"])
            elif "player_id" in request.params:
                player_ids.add(request.params["player_id"])
        
        if not player_ids:
            return
        
        # Fetch all players in one request
        url = f"{SLEEPER_BASE_URL}/players/nfl"
        
        await self.rate_limiter.acquire()
        async with self.session.get(url) as response:
            if response.status == 200:
                all_players = await response.json()
                self.metrics["requests_sent"] += 1
                
                # Filter to requested players
                requested_players = {
                    pid: player for pid, player in all_players.items()
                    if pid in player_ids
                }
                
                # Resolve all futures
                for request in batch:
                    try:
                        if "player_ids" in request.params:
                            # Multiple players requested
                            result = {
                                pid: requested_players.get(pid)
                                for pid in request.params["player_ids"]
                            }
                        else:
                            # Single player requested
                            result = requested_players.get(request.params["player_id"])
                        
                        request.future.set_result(result)
                        self.metrics["requests_batched"] += 1
                        
                    except Exception as e:
                        request.future.set_exception(e)
            else:
                # Handle error
                error = f"Sleeper API error: {response.status}"
                for request in batch:
                    request.future.set_exception(Exception(error))
    
    async def _execute_stats_batch(self, batch: List[BatchRequest]):
        """Execute batch stats requests"""
        # Group by season/week
        season_week_groups = {}
        
        for request in batch:
            season = request.params.get("season", 2024)
            week = request.params.get("week")
            key = f"{season}_{week}"
            
            if key not in season_week_groups:
                season_week_groups[key] = []
            season_week_groups[key].append(request)
        
        # Process each season/week group
        for key, requests in season_week_groups.items():
            season, week = key.split("_")
            week = int(week) if week != "None" else None
            
            # Build URL
            if week:
                url = f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}/{week}"
            else:
                url = f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}"
            
            await self.rate_limiter.acquire()
            async with self.session.get(url) as response:
                if response.status == 200:
                    stats_data = await response.json()
                    self.metrics["requests_sent"] += 1
                    
                    # Resolve futures for this group
                    for request in requests:
                        try:
                            player_ids = request.params.get("player_ids", [])
                            if player_ids:
                                # Filter to requested players
                                result = {
                                    pid: stats_data.get(pid, {})
                                    for pid in player_ids
                                }
                            else:
                                result = stats_data
                            
                            request.future.set_result(result)
                            self.metrics["requests_batched"] += 1
                            
                        except Exception as e:
                            request.future.set_exception(e)
                else:
                    error = f"Sleeper API error: {response.status}"
                    for request in requests:
                        request.future.set_exception(Exception(error))
    
    async def _execute_projections_batch(self, batch: List[BatchRequest]):
        """Execute batch projections requests"""
        # Similar to stats but for projections endpoint
        season_week_groups = {}
        
        for request in batch:
            season = request.params.get("season", 2024)
            week = request.params.get("week")
            key = f"{season}_{week}"
            
            if key not in season_week_groups:
                season_week_groups[key] = []
            season_week_groups[key].append(request)
        
        for key, requests in season_week_groups.items():
            season, week = key.split("_")
            week = int(week) if week != "None" else None
            
            if week:
                url = f"{SLEEPER_BASE_URL}/projections/nfl/regular/{season}/{week}"
            else:
                url = f"{SLEEPER_BASE_URL}/projections/nfl/regular/{season}"
            
            await self.rate_limiter.acquire()
            async with self.session.get(url) as response:
                if response.status == 200:
                    proj_data = await response.json()
                    self.metrics["requests_sent"] += 1
                    
                    for request in requests:
                        try:
                            player_ids = request.params.get("player_ids", [])
                            if player_ids:
                                result = {
                                    pid: proj_data.get(pid, {})
                                    for pid in player_ids
                                }
                            else:
                                result = proj_data
                            
                            request.future.set_result(result)
                            self.metrics["requests_batched"] += 1
                            
                        except Exception as e:
                            request.future.set_exception(e)
                else:
                    error = f"Sleeper API error: {response.status}"
                    for request in requests:
                        request.future.set_exception(Exception(error))
    
    async def _execute_single_request(self, request: BatchRequest):
        """Execute a single request that can't be batched"""
        try:
            # Build URL based on request type and params
            if request.request_type == RequestType.LEAGUE:
                league_id = request.params["league_id"]
                url = f"{SLEEPER_BASE_URL}/league/{league_id}"
            elif request.request_type == RequestType.ROSTER:
                league_id = request.params["league_id"]
                url = f"{SLEEPER_BASE_URL}/league/{league_id}/rosters"
            elif request.request_type == RequestType.MATCHUP:
                league_id = request.params["league_id"]
                week = request.params["week"]
                url = f"{SLEEPER_BASE_URL}/league/{league_id}/matchups/{week}"
            else:
                request.future.set_exception(Exception(f"Unknown request type: {request.request_type}"))
                return
            
            await self.rate_limiter.acquire()
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    request.future.set_result(data)
                    self.metrics["requests_sent"] += 1
                else:
                    error = f"Sleeper API error: {response.status}"
                    request.future.set_exception(Exception(error))
                    
        except Exception as e:
            request.future.set_exception(e)
    
    def _create_request_key(self, request_type: str, params: Dict[str, Any]) -> str:
        """Create unique key for request deduplication"""
        key_data = f"{request_type}:{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _queue_request(
        self,
        request_type: str,
        params: Dict[str, Any],
        priority: int = 0
    ) -> Any:
        """Queue a request for batching"""
        # Check for request deduplication
        request_key = self._create_request_key(request_type, params)
        
        if request_key in self.pending_requests:
            # Duplicate request - wait for existing one
            future = asyncio.Future()
            self.pending_requests[request_key].append(future)
            self.metrics["requests_deduplicated"] += 1
            return await future
        
        # Create new request
        future = asyncio.Future()
        self.pending_requests[request_key] = [future]
        
        request = BatchRequest(
            request_type=request_type,
            params=params,
            future=future,
            priority=priority
        )
        
        # Add to appropriate queue
        self.batch_queues[request_type].append(request)
        
        try:
            result = await future
            
            # Resolve any duplicate requests
            for dup_future in self.pending_requests[request_key]:
                if not dup_future.done():
                    dup_future.set_result(result)
            
            del self.pending_requests[request_key]
            return result
            
        except Exception as e:
            # Propagate error to duplicates
            for dup_future in self.pending_requests[request_key]:
                if not dup_future.done():
                    dup_future.set_exception(e)
            
            del self.pending_requests[request_key]
            raise
    
    # Public API methods
    
    @cached(prefix="sleeper:player:", ttl=CACHE_TTL_LONG)
    async def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Get single player data"""
        return await self._queue_request(
            RequestType.PLAYER,
            {"player_id": player_id},
            priority=1
        )
    
    @cached(prefix="sleeper:players:", ttl=CACHE_TTL_LONG)
    async def get_players(self, player_ids: List[str]) -> Dict[str, Any]:
        """Get multiple players data"""
        return await self._queue_request(
            RequestType.PLAYER,
            {"player_ids": player_ids},
            priority=2
        )
    
    @cached(prefix="sleeper:stats:", ttl=CACHE_TTL_SHORT)
    async def get_stats(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        player_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get player stats"""
        params = {"season": season, "week": week}
        if player_ids:
            params["player_ids"] = player_ids
        
        return await self._queue_request(
            RequestType.STATS,
            params,
            priority=3
        )
    
    @cached(prefix="sleeper:proj:", ttl=CACHE_TTL_SHORT)
    async def get_projections(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        player_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get player projections"""
        params = {"season": season, "week": week}
        if player_ids:
            params["player_ids"] = player_ids
        
        return await self._queue_request(
            RequestType.PROJECTIONS,
            params,
            priority=3
        )
    
    @cached(prefix="sleeper:league:", ttl=CACHE_TTL_SHORT)
    async def get_league(self, league_id: str) -> Dict[str, Any]:
        """Get league information"""
        return await self._queue_request(
            RequestType.LEAGUE,
            {"league_id": league_id}
        )
    
    @cached(prefix="sleeper:rosters:", ttl=CACHE_TTL_SHORT)
    async def get_rosters(self, league_id: str) -> List[Dict[str, Any]]:
        """Get league rosters"""
        return await self._queue_request(
            RequestType.ROSTER,
            {"league_id": league_id}
        )
    
    @cached(prefix="sleeper:matchups:", ttl=CACHE_TTL_SHORT)
    async def get_matchups(self, league_id: str, week: int) -> List[Dict[str, Any]]:
        """Get league matchups for a week"""
        return await self._queue_request(
            RequestType.MATCHUP,
            {"league_id": league_id, "week": week}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "queue_sizes": {
                req_type: len(queue)
                for req_type, queue in self.batch_queues.items()
            },
            "pending_requests": len(self.pending_requests)
        }