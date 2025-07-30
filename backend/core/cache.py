"""
Redis Cache Configuration and Utilities
Optimized for sub-200ms response times
"""
import redis
import json
import pickle
import hashlib
from typing import Optional, Any, Callable
from functools import wraps
from datetime import timedelta
import logging
import os

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_DEFAULT = 3600  # 1 hour default
CACHE_TTL_SHORT = 300  # 5 minutes for volatile data
CACHE_TTL_LONG = 86400  # 24 hours for stable data

# Cache key prefixes
CACHE_PREFIX_PREDICTION = "pred:"
CACHE_PREFIX_PLAYER = "player:"
CACHE_PREFIX_STATS = "stats:"
CACHE_PREFIX_TIER = "tier:"
CACHE_PREFIX_EFFICIENCY = "eff:"
CACHE_PREFIX_MOMENTUM = "mom:"


class RedisCache:
    """High-performance Redis cache manager"""
    
    def __init__(self):
        self.redis_client = None
        self.connect()
    
    def connect(self):
        """Establish Redis connection with retry logic"""
        try:
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=False,  # Use binary for pickle
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic deserialization"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                # Try JSON first, then pickle
                try:
                    return json.loads(value)
                except:
                    return pickle.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = CACHE_TTL_DEFAULT
    ) -> bool:
        """Set value in cache with automatic serialization"""
        if not self.redis_client:
            return False
        
        try:
            # Try JSON first for better interoperability
            try:
                serialized = json.dumps(value)
            except:
                serialized = pickle.dumps(value)
            
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {str(e)}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except:
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key"""
        if not self.redis_client:
            return -1
        
        try:
            return self.redis_client.ttl(key)
        except:
            return -1
    
    def batch_get(self, keys: list) -> dict:
        """Get multiple values in a single operation"""
        if not self.redis_client or not keys:
            return {}
        
        try:
            values = self.redis_client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except:
                        result[key] = pickle.loads(value)
            return result
        except Exception as e:
            logger.error(f"Batch get error: {str(e)}")
            return {}
    
    def pipeline_set(self, data: dict, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        """Set multiple values using pipeline for performance"""
        if not self.redis_client or not data:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            for key, value in data.items():
                try:
                    serialized = json.dumps(value)
                except:
                    serialized = pickle.dumps(value)
                pipe.setex(key, ttl, serialized)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Pipeline set error: {str(e)}")
            return False


# Global cache instance
cache = RedisCache()


def cache_key(*args, **kwargs) -> str:
    """Generate consistent cache key from arguments"""
    key_data = f"{args}:{sorted(kwargs.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    prefix: str = "",
    ttl: int = CACHE_TTL_DEFAULT,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_func: Custom function to generate cache key
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = cache_key(*args, **kwargs)
            
            full_key = f"{prefix}{cache_key_str}"
            
            # Try to get from cache
            cached_value = cache.get(full_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(full_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = cache_key(*args, **kwargs)
            
            full_key = f"{prefix}{cache_key_str}"
            
            # Try to get from cache
            cached_value = cache.get(full_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(full_key, result, ttl)
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def invalidate_player_cache(player_id: str):
    """Invalidate all cache entries for a player"""
    patterns = [
        f"{CACHE_PREFIX_PREDICTION}{player_id}*",
        f"{CACHE_PREFIX_PLAYER}{player_id}*",
        f"{CACHE_PREFIX_STATS}{player_id}*",
        f"{CACHE_PREFIX_EFFICIENCY}{player_id}*",
        f"{CACHE_PREFIX_MOMENTUM}{player_id}*"
    ]
    
    total_deleted = 0
    for pattern in patterns:
        deleted = cache.delete_pattern(pattern)
        total_deleted += deleted
    
    logger.info(f"Invalidated {total_deleted} cache entries for player {player_id}")
    return total_deleted


def warm_cache_for_week(season: int, week: int):
    """Pre-warm cache for commonly accessed data"""
    # This would be called by a background task
    # to pre-populate cache before peak usage
    pass


# Cache warming strategies
class CacheWarmer:
    """Background cache warming for optimal performance"""
    
    @staticmethod
    async def warm_predictions(player_ids: list, season: int, week: int):
        """Pre-calculate and cache predictions for multiple players"""
        from backend.ml.ensemble_predictions import EnsemblePredictionEngine
        
        engine = EnsemblePredictionEngine()
        cache_data = {}
        
        for player_id in player_ids:
            key = f"{CACHE_PREFIX_PREDICTION}{player_id}:{season}:{week}"
            if not cache.exists(key):
                try:
                    prediction = engine.predict_player_week(
                        player_id, season, week, include_explanations=False
                    )
                    cache_data[key] = prediction
                except Exception as e:
                    logger.error(f"Failed to warm cache for {player_id}: {str(e)}")
        
        if cache_data:
            cache.pipeline_set(cache_data, ttl=CACHE_TTL_DEFAULT)
            logger.info(f"Warmed cache for {len(cache_data)} predictions")
    
    @staticmethod
    async def warm_player_stats(player_ids: list, season: int):
        """Pre-cache player statistics"""
        from backend.models.database import SessionLocal, PlayerStats
        
        cache_data = {}
        
        with SessionLocal() as db:
            for player_id in player_ids:
                key = f"{CACHE_PREFIX_STATS}{player_id}:{season}"
                if not cache.exists(key):
                    stats = db.query(PlayerStats).filter(
                        PlayerStats.player_id == player_id,
                        PlayerStats.season == season
                    ).all()
                    
                    if stats:
                        cache_data[key] = [
                            {
                                'week': s.week,
                                'points_ppr': float(s.fantasy_points_ppr) if s.fantasy_points_ppr else 0,
                                'points_std': float(s.fantasy_points_std) if s.fantasy_points_std else 0
                            }
                            for s in stats
                        ]
        
        if cache_data:
            cache.pipeline_set(cache_data, ttl=CACHE_TTL_LONG)
            logger.info(f"Warmed cache for {len(cache_data)} player stats")