"""
Advanced Adaptive Rate Limiter with Exponential Backoff and Performance Monitoring.
Location: src/fantasy_ai/core/data/rate_limiter.py
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import json

from sqlalchemy.orm import Session
from .storage.models import ApiRateLimit
from .storage.database import get_db_session

logger = logging.getLogger(__name__)

class RateLimitTier(Enum):
    """Rate limiting performance tiers."""
    OPTIMAL = "optimal"       # Fast response times, no errors
    CAUTIOUS = "cautious"     # Moderate response times, few errors  
    CONSERVATIVE = "conservative"  # Slow response times, some errors
    RESTRICTED = "restricted"  # Very slow, many errors

@dataclass
class RateLimitConfig:
    """Configuration for adaptive rate limiting."""
    requests_per_day: int = 100
    requests_per_hour: int = 10
    requests_per_minute: int = 5
    
    # Adaptive parameters
    min_request_interval: float = 1.0    # Minimum seconds between requests
    max_request_interval: float = 60.0   # Maximum seconds between requests
    backoff_multiplier: float = 1.5      # Exponential backoff multiplier
    recovery_factor: float = 0.9         # Recovery rate for successful requests
    
    # Performance thresholds
    slow_response_threshold: float = 5.0  # Seconds
    error_rate_threshold: float = 0.1     # 10% error rate triggers restrictions
    consecutive_error_threshold: int = 3   # Consecutive errors trigger backoff
    
    # Monitoring
    performance_window_size: int = 20     # Number of recent requests to analyze
    tier_adjustment_interval: int = 300   # Seconds between tier adjustments

@dataclass
class RequestMetrics:
    """Metrics for a single API request."""
    timestamp: datetime
    response_time: float
    success: bool
    status_code: Optional[int] = None
    error_message: Optional[str] = None

class AdaptiveRateLimiter:
    """
    Advanced rate limiter that adapts to API performance and implements
    intelligent backoff strategies with real-time monitoring.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.current_tier = RateLimitTier.OPTIMAL
        
        # Request tracking
        self.request_history: List[RequestMetrics] = []
        self.last_request_time: Optional[float] = None
        self.current_interval: float = config.min_request_interval
        
        # Error tracking
        self.consecutive_errors = 0
        self.total_requests = 0
        self.total_errors = 0
        
        # Daily/hourly counters
        self.daily_requests = 0
        self.hourly_requests = 0
        self.minute_requests = 0
        self.last_reset_day = datetime.now(timezone.utc).date()
        self.last_reset_hour = datetime.now(timezone.utc).hour
        self.last_reset_minute = datetime.now(timezone.utc).minute
        
        # Performance monitoring
        self.last_tier_adjustment = time.time()
        
        # Asyncio synchronization
        self._request_lock = asyncio.Lock()

    async def can_make_request(self) -> bool:
        """Check if a request can be made without waiting."""
        async with self._request_lock:
            await self._update_counters()
            
            # Check daily limit
            if self.daily_requests >= self.config.requests_per_day:
                return False
            
            # Check hourly limit
            if self.hourly_requests >= self.config.requests_per_hour:
                return False
            
            # Check minute limit
            if self.minute_requests >= self.config.requests_per_minute:
                return False
            
            # Check time interval
            if self.last_request_time:
                time_since_last = time.time() - self.last_request_time
                if time_since_last < self.current_interval:
                    return False
            
            return True

    async def wait_for_request(self) -> None:
        """Wait until a request can be made, implementing adaptive delays."""
        
        while not await self.can_make_request():
            wait_time = await self.get_wait_time()
            
            logger.debug(f"Rate limit wait: {wait_time:.1f}s (tier: {self.current_tier.value})")
            await asyncio.sleep(min(wait_time, 60))  # Cap wait time at 1 minute
        
        # Update request tracking
        async with self._request_lock:
            self.last_request_time = time.time()
            await self._increment_counters()

    async def get_wait_time(self) -> float:
        """Calculate how long to wait before next request."""
        
        await self._update_counters()
        
        wait_times = []
        
        # Time-based waits
        if self.last_request_time:
            time_since_last = time.time() - self.last_request_time
            interval_wait = max(0, self.current_interval - time_since_last)
            wait_times.append(interval_wait)
        
        # Daily limit wait
        if self.daily_requests >= self.config.requests_per_day:
            tomorrow = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            daily_wait = (tomorrow - datetime.now(timezone.utc)).total_seconds()
            wait_times.append(daily_wait)
        
        # Hourly limit wait
        if self.hourly_requests >= self.config.requests_per_hour:
            next_hour = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
            hourly_wait = (next_hour - datetime.now(timezone.utc)).total_seconds()
            wait_times.append(hourly_wait)
        
        # Minute limit wait
        if self.minute_requests >= self.config.requests_per_minute:
            next_minute = datetime.now(timezone.utc).replace(
                second=0, microsecond=0
            ) + timedelta(minutes=1)
            minute_wait = (next_minute - datetime.now(timezone.utc)).total_seconds()
            wait_times.append(minute_wait)
        
        return max(wait_times) if wait_times else 0

    async def record_request(self, response_time: float, success: bool, 
                           status_code: Optional[int] = None, 
                           error_message: Optional[str] = None) -> None:
        """Record the result of an API request for adaptive learning."""
        
        async with self._request_lock:
            # Record metrics
            metrics = RequestMetrics(
                timestamp=datetime.now(timezone.utc),
                response_time=response_time,
                success=success,
                status_code=status_code,
                error_message=error_message
            )
            
            self.request_history.append(metrics)
            
            # Limit history size
            if len(self.request_history) > self.config.performance_window_size * 2:
                self.request_history = self.request_history[-self.config.performance_window_size:]
            
            # Update counters
            self.total_requests += 1
            if not success:
                self.total_errors += 1
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 0
            
            # Adapt request interval based on performance
            await self._adapt_request_interval(metrics)
            
            # Check if tier adjustment is needed
            if time.time() - self.last_tier_adjustment > self.config.tier_adjustment_interval:
                await self._adjust_performance_tier()
                self.last_tier_adjustment = time.time()
            
            # Persist to database
            await self._update_database_metrics()

    async def _adapt_request_interval(self, metrics: RequestMetrics) -> None:
        """Adapt request interval based on latest request performance."""
        
        if not metrics.success:
            # Increase interval on errors (exponential backoff)
            self.current_interval = min(
                self.current_interval * self.config.backoff_multiplier,
                self.config.max_request_interval
            )
            
            # Additional backoff for consecutive errors
            if self.consecutive_errors >= self.config.consecutive_error_threshold:
                self.current_interval = min(
                    self.current_interval * (1 + self.consecutive_errors * 0.1),
                    self.config.max_request_interval
                )
            
            logger.warning(f"Request failed, increased interval to {self.current_interval:.1f}s "
                          f"(consecutive errors: {self.consecutive_errors})")
        
        elif metrics.response_time > self.config.slow_response_threshold:
            # Increase interval for slow responses
            self.current_interval = min(
                self.current_interval * 1.2,
                self.config.max_request_interval
            )
            
            logger.info(f"Slow response ({metrics.response_time:.1f}s), "
                       f"increased interval to {self.current_interval:.1f}s")
        
        else:
            # Decrease interval for fast, successful responses (recovery)
            self.current_interval = max(
                self.current_interval * self.config.recovery_factor,
                self.config.min_request_interval
            )

    async def _adjust_performance_tier(self) -> None:
        """Adjust performance tier based on recent request history."""
        
        if len(self.request_history) < 10:
            return
        
        recent_requests = self.request_history[-self.config.performance_window_size:]
        
        # Calculate performance metrics
        success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests)
        avg_response_time = statistics.mean(r.response_time for r in recent_requests)
        error_rate = 1 - success_rate
        
        # Determine appropriate tier
        old_tier = self.current_tier
        
        if error_rate <= 0.05 and avg_response_time <= 2.0:
            self.current_tier = RateLimitTier.OPTIMAL
        elif error_rate <= 0.1 and avg_response_time <= self.config.slow_response_threshold:
            self.current_tier = RateLimitTier.CAUTIOUS
        elif error_rate <= 0.2 and avg_response_time <= self.config.slow_response_threshold * 2:
            self.current_tier = RateLimitTier.CONSERVATIVE
        else:
            self.current_tier = RateLimitTier.RESTRICTED
        
        # Adjust base parameters based on tier
        tier_configs = {
            RateLimitTier.OPTIMAL: {
                'min_interval': self.config.min_request_interval,
                'backoff_multiplier': 1.3
            },
            RateLimitTier.CAUTIOUS: {
                'min_interval': self.config.min_request_interval * 1.5,
                'backoff_multiplier': 1.5
            },
            RateLimitTier.CONSERVATIVE: {
                'min_interval': self.config.min_request_interval * 2.0,
                'backoff_multiplier': 2.0
            },
            RateLimitTier.RESTRICTED: {
                'min_interval': self.config.min_request_interval * 3.0,
                'backoff_multiplier': 2.5
            }
        }
        
        tier_config = tier_configs[self.current_tier]
        self.current_interval = max(self.current_interval, tier_config['min_interval'])
        
        if old_tier != self.current_tier:
            logger.info(f"Performance tier changed: {old_tier.value} -> {self.current_tier.value} "
                       f"(error_rate: {error_rate:.1%}, avg_response_time: {avg_response_time:.1f}s)")

    async def _update_counters(self) -> None:
        """Update time-based request counters."""
        
        now = datetime.now(timezone.utc)
        
        # Reset daily counter
        if now.date() > self.last_reset_day:
            self.daily_requests = 0
            self.last_reset_day = now.date()
        
        # Reset hourly counter
        if now.hour != self.last_reset_hour:
            self.hourly_requests = 0
            self.last_reset_hour = now.hour
        
        # Reset minute counter
        if now.minute != self.last_reset_minute:
            self.minute_requests = 0
            self.last_reset_minute = now.minute

    async def _increment_counters(self) -> None:
        """Increment request counters."""
        self.daily_requests += 1
        self.hourly_requests += 1
        self.minute_requests += 1

    async def _update_database_metrics(self) -> None:
        """Update rate limit metrics in database."""
        
        try:
            async with get_db_session() as session:
                # Get or create rate limit record
                rate_limit = session.query(ApiRateLimit).filter(
                    ApiRateLimit.api_name == 'nfl_api',
                    ApiRateLimit.endpoint == 'default'
                ).first()
                
                if not rate_limit:
                    rate_limit = ApiRateLimit(
                        api_name='nfl_api',
                        endpoint='default',
                        requests_limit=self.config.requests_per_day,
                        reset_time=datetime.now(timezone.utc).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ) + timedelta(days=1)
                    )
                    session.add(rate_limit)
                
                # Update metrics
                rate_limit.requests_made = self.daily_requests
                rate_limit.last_request_time = datetime.now(timezone.utc)
                rate_limit.consecutive_errors = self.consecutive_errors
                
                # Calculate average response time from recent requests
                if self.request_history:
                    recent_requests = self.request_history[-10:]
                    rate_limit.avg_response_time = statistics.mean(
                        r.response_time for r in recent_requests
                    )
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error updating database metrics: {e}")

    def get_status(self) -> Dict[str, any]:
        """Get current rate limiter status and performance metrics."""
        
        recent_requests = self.request_history[-self.config.performance_window_size:] \
                         if len(self.request_history) >= self.config.performance_window_size \
                         else self.request_history
        
        status = {
            'current_tier': self.current_tier.value,
            'current_interval': self.current_interval,
            'requests_today': self.daily_requests,
            'requests_this_hour': self.hourly_requests,
            'requests_this_minute': self.minute_requests,
            'consecutive_errors': self.consecutive_errors,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
        }
        
        if recent_requests:
            success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests)
            avg_response_time = statistics.mean(r.response_time for r in recent_requests)
            
            status.update({
                'recent_success_rate': success_rate,
                'recent_avg_response_time': avg_response_time,
                'recent_requests_count': len(recent_requests)
            })
        
        # Rate limit status
        status.update({
            'daily_limit': self.config.requests_per_day,
            'hourly_limit': self.config.requests_per_hour,
            'minute_limit': self.config.requests_per_minute,
            'daily_remaining': max(0, self.config.requests_per_day - self.daily_requests),
            'hourly_remaining': max(0, self.config.requests_per_hour - self.hourly_requests),
            'minute_remaining': max(0, self.config.requests_per_minute - self.minute_requests)
        })
        
        return status

    async def reset_errors(self) -> None:
        """Manually reset error counters (useful for recovery after issues resolved)."""
        async with self._request_lock:
            self.consecutive_errors = 0
            self.current_interval = self.config.min_request_interval
            self.current_tier = RateLimitTier.OPTIMAL
            logger.info("Rate limiter error counters reset")

    async def set_conservative_mode(self, enabled: bool = True) -> None:
        """Manually set conservative mode for cautious operation."""
        async with self._request_lock:
            if enabled:
                self.current_tier = RateLimitTier.CONSERVATIVE
                self.current_interval = self.config.min_request_interval * 2.0
                logger.info("Conservative mode enabled")
            else:
                self.current_tier = RateLimitTier.OPTIMAL
                self.current_interval = self.config.min_request_interval
                logger.info("Conservative mode disabled")

class RateLimitDecorator:
    """Decorator for automatically handling rate limiting on API calls."""
    
    def __init__(self, rate_limiter: AdaptiveRateLimiter):
        self.rate_limiter = rate_limiter
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Wait for rate limit
            await self.rate_limiter.wait_for_request()
            
            start_time = time.time()
            success = True
            status_code = None
            error_message = None
            
            try:
                result = await func(*args, **kwargs)
                
                # Check if result indicates success (customize based on API)
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                    success = 200 <= status_code < 300
                elif result is None:
                    success = False
                    error_message = "No data returned"
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            
            finally:
                response_time = time.time() - start_time
                await self.rate_limiter.record_request(
                    response_time=response_time,
                    success=success,
                    status_code=status_code,
                    error_message=error_message
                )
        
        return wrapper

# Context manager for rate-limited operations
class RateLimitedContext:
    """Context manager for rate-limited API operations."""
    
    def __init__(self, rate_limiter: AdaptiveRateLimiter, operation_name: str = "API call"):
        self.rate_limiter = rate_limiter
        self.operation_name = operation_name
        self.start_time = None
    
    async def __aenter__(self):
        await self.rate_limiter.wait_for_request()
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            response_time = time.time() - self.start_time
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            
            await self.rate_limiter.record_request(
                response_time=response_time,
                success=success,
                error_message=error_message
            )
            
            if not success:
                logger.warning(f"{self.operation_name} failed: {error_message}")

# Factory function for creating configured rate limiters
def create_nfl_api_rate_limiter(requests_per_day: int = 100) -> AdaptiveRateLimiter:
    """Create a pre-configured rate limiter for NFL API."""
    
    config = RateLimitConfig(
        requests_per_day=requests_per_day,
        requests_per_hour=max(4, requests_per_day // 24),
        requests_per_minute=2,
        min_request_interval=2.0,  # Conservative for NFL API
        max_request_interval=300.0,  # 5 minutes max
        slow_response_threshold=10.0,  # NFL API can be slow
        error_rate_threshold=0.15,  # Slightly higher tolerance
        consecutive_error_threshold=2,  # Aggressive error handling
        performance_window_size=15
    )
    
    return AdaptiveRateLimiter(config)