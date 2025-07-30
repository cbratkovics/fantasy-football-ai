"""
Rate Limiting System for Subscription Tiers
Implements Redis-based rate limiting with sliding window algorithm
"""
import time
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
import hashlib

from fastapi import HTTPException, Request, Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from backend.core.cache import cache

logger = logging.getLogger(__name__)

# Rate limit configurations per tier
RATE_LIMITS = {
    "free": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
        "burst_size": 10,  # Allow short bursts
        "concurrent_requests": 2
    },
    "pro": {
        "requests_per_minute": 300,
        "requests_per_hour": 5000,
        "requests_per_day": 50000,
        "burst_size": 50,
        "concurrent_requests": 5
    },
    "premium": {
        "requests_per_minute": 1000,
        "requests_per_hour": 20000,
        "requests_per_day": 200000,
        "burst_size": 100,
        "concurrent_requests": 10
    },
    "unlimited": {  # For internal services
        "requests_per_minute": float('inf'),
        "requests_per_hour": float('inf'),
        "requests_per_day": float('inf'),
        "burst_size": float('inf'),
        "concurrent_requests": float('inf')
    }
}

# Endpoint-specific multipliers (some endpoints cost more)
ENDPOINT_COSTS = {
    "/api/v1/predictions/batch": 5,  # Batch predictions cost 5x
    "/api/v1/ml/train": 100,  # Training endpoints very expensive
    "/api/v1/analysis/full": 10,  # Full analysis costs 10x
    "/api/v1/export": 20,  # Data export costs 20x
}


class RateLimiter:
    """
    Sliding window rate limiter with Redis backend
    Supports multiple time windows and burst protection
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client or cache.redis_client
        self.enabled = self.redis is not None
    
    def _get_key(self, identifier: str, window: str) -> str:
        """Generate Redis key for rate limit tracking"""
        return f"rate_limit:{identifier}:{window}"
    
    def _get_identifier(self, user_id: Optional[str], ip_address: str) -> str:
        """Get unique identifier for rate limiting"""
        if user_id:
            return f"user:{user_id}"
        else:
            # Hash IP for privacy
            return f"ip:{hashlib.sha256(ip_address.encode()).hexdigest()[:16]}"
    
    def check_rate_limit(
        self,
        identifier: str,
        tier: str,
        endpoint: str = "/",
        increment: bool = True
    ) -> Tuple[bool, dict]:
        """
        Check if request is within rate limits
        
        Returns:
            Tuple of (allowed, metadata)
        """
        if not self.enabled:
            return True, {"limit": "unlimited", "remaining": "unlimited"}
        
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
        
        # Calculate endpoint cost
        cost = ENDPOINT_COSTS.get(endpoint, 1)
        
        # Check all time windows
        current_time = time.time()
        windows = {
            "minute": (60, limits["requests_per_minute"]),
            "hour": (3600, limits["requests_per_hour"]),
            "day": (86400, limits["requests_per_day"])
        }
        
        for window_name, (window_seconds, limit) in windows.items():
            if limit == float('inf'):
                continue
            
            key = self._get_key(identifier, window_name)
            
            # Sliding window implementation
            try:
                pipe = self.redis.pipeline()
                
                # Remove old entries
                pipe.zremrangebyscore(key, 0, current_time - window_seconds)
                
                # Count current requests
                pipe.zcard(key)
                
                # Add current request if incrementing
                if increment:
                    pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiry
                pipe.expire(key, window_seconds + 60)
                
                results = pipe.execute()
                current_count = results[1]
                
                # Check if over limit (accounting for cost)
                if current_count + cost > limit:
                    return False, {
                        "limit": limit,
                        "window": window_name,
                        "current": current_count,
                        "reset_at": current_time + window_seconds,
                        "retry_after": window_seconds
                    }
                    
            except Exception as e:
                logger.error(f"Rate limit check failed: {str(e)}")
                # Fail open - allow request if Redis is down
                return True, {"error": "rate_limit_check_failed"}
        
        # Check burst protection
        burst_key = f"{identifier}:burst"
        burst_count = self._check_burst(burst_key, limits["burst_size"])
        
        if not burst_count[0]:
            return False, {
                "limit": limits["burst_size"],
                "window": "burst",
                "message": "Burst limit exceeded, please slow down"
            }
        
        # All checks passed
        remaining = min(
            limits["requests_per_minute"] - self._get_count(identifier, "minute"),
            limits["requests_per_hour"] - self._get_count(identifier, "hour"),
            limits["requests_per_day"] - self._get_count(identifier, "day")
        )
        
        return True, {
            "limit": limits["requests_per_minute"],
            "remaining": max(0, remaining),
            "tier": tier,
            "reset_at": current_time + 60
        }
    
    def _check_burst(self, key: str, burst_limit: int) -> Tuple[bool, int]:
        """Check burst limit using token bucket algorithm"""
        if burst_limit == float('inf'):
            return True, burst_limit
        
        try:
            current_time = time.time()
            
            # Get current tokens
            pipe = self.redis.pipeline()
            pipe.get(f"{key}:tokens")
            pipe.get(f"{key}:last_refill")
            tokens_data = pipe.execute()
            
            tokens = float(tokens_data[0]) if tokens_data[0] else burst_limit
            last_refill = float(tokens_data[1]) if tokens_data[1] else current_time
            
            # Refill tokens (1 token per second up to burst_limit)
            time_passed = current_time - last_refill
            tokens = min(burst_limit, tokens + time_passed)
            
            # Check if we have tokens
            if tokens < 1:
                return False, 0
            
            # Consume a token
            tokens -= 1
            
            # Update Redis
            pipe = self.redis.pipeline()
            pipe.setex(f"{key}:tokens", 300, tokens)
            pipe.setex(f"{key}:last_refill", 300, current_time)
            pipe.execute()
            
            return True, int(tokens)
            
        except Exception as e:
            logger.error(f"Burst check failed: {str(e)}")
            return True, burst_limit
    
    def _get_count(self, identifier: str, window: str) -> int:
        """Get current request count for a window"""
        try:
            key = self._get_key(identifier, window)
            window_seconds = {"minute": 60, "hour": 3600, "day": 86400}[window]
            
            self.redis.zremrangebyscore(key, 0, time.time() - window_seconds)
            return self.redis.zcard(key)
        except:
            return 0
    
    def get_rate_limit_status(self, identifier: str, tier: str) -> dict:
        """Get current rate limit status for user"""
        if not self.enabled:
            return {"status": "unlimited"}
        
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
        current_time = time.time()
        
        status = {
            "tier": tier,
            "limits": limits,
            "current_usage": {
                "minute": self._get_count(identifier, "minute"),
                "hour": self._get_count(identifier, "hour"),
                "day": self._get_count(identifier, "day")
            },
            "reset_times": {
                "minute": current_time + 60,
                "hour": current_time + 3600,
                "day": current_time + 86400
            }
        }
        
        return status


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(tier_getter=None):
    """
    Decorator for rate limiting endpoints
    
    Args:
        tier_getter: Function to get user tier from request
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(request: Request, *args, **kwargs):
            # Get user tier
            if tier_getter:
                tier = await tier_getter(request)
            else:
                # Default tier detection
                tier = getattr(request.state, "user_tier", "free")
            
            # Get identifier
            user_id = getattr(request.state, "user_id", None)
            ip_address = request.client.host
            identifier = rate_limiter._get_identifier(user_id, ip_address)
            
            # Check rate limit
            allowed, metadata = rate_limiter.check_rate_limit(
                identifier,
                tier,
                request.url.path
            )
            
            if not allowed:
                # Add rate limit headers
                response = Response(
                    content="Rate limit exceeded",
                    status_code=HTTP_429_TOO_MANY_REQUESTS
                )
                response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", 0))
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset_at", 0)))
                response.headers["Retry-After"] = str(metadata.get("retry_after", 60))
                
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. {metadata.get('message', 'Please try again later.')}"
                )
            
            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset_at", 0)))
            
            return response
        
        @wraps(func)
        def sync_wrapper(request: Request, *args, **kwargs):
            # Similar logic for sync functions
            if tier_getter:
                tier = tier_getter(request)
            else:
                tier = getattr(request.state, "user_tier", "free")
            
            user_id = getattr(request.state, "user_id", None)
            ip_address = request.client.host
            identifier = rate_limiter._get_identifier(user_id, ip_address)
            
            allowed, metadata = rate_limiter.check_rate_limit(
                identifier,
                tier,
                request.url.path
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. {metadata.get('message', 'Please try again later.')}"
                )
            
            return func(request, *args, **kwargs)
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Middleware for global rate limiting
async def rate_limit_middleware(request: Request, call_next):
    """Global rate limiting middleware"""
    # Skip rate limiting for health checks and metrics
    if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
        return await call_next(request)
    
    # Get user tier from auth
    tier = getattr(request.state, "user_tier", "free")
    user_id = getattr(request.state, "user_id", None)
    ip_address = request.client.host
    
    identifier = rate_limiter._get_identifier(user_id, ip_address)
    
    # Check rate limit
    allowed, metadata = rate_limiter.check_rate_limit(
        identifier,
        tier,
        request.url.path
    )
    
    if not allowed:
        return Response(
            content=f"Rate limit exceeded. {metadata.get('message', 'Please try again later.')}",
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "X-RateLimit-Limit": str(metadata.get("limit", 0)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(metadata.get("reset_at", 0))),
                "Retry-After": str(metadata.get("retry_after", 60))
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", 0))
    response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
    response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset_at", 0)))
    
    return response


# Admin endpoints for rate limit management
async def reset_rate_limit(user_id: str):
    """Reset rate limits for a user (admin only)"""
    identifier = f"user:{user_id}"
    
    for window in ["minute", "hour", "day"]:
        key = rate_limiter._get_key(identifier, window)
        cache.delete(key)
    
    # Reset burst tokens
    cache.delete(f"{identifier}:burst:tokens")
    cache.delete(f"{identifier}:burst:last_refill")
    
    return {"status": "reset", "user_id": user_id}