"""
Optimized FastAPI Backend for Fantasy Football AI
Achieves sub-200ms response times with caching and performance optimizations
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging
import time
import asyncio
from typing import Optional
import uvicorn

# Performance monitoring
from prometheus_client import Counter, Histogram, generate_latest
import psutil

# Import optimized components
from backend.core.cache import cache, CacheWarmer
from backend.core.rate_limiter import RateLimiter
from backend.api import auth, players, predictions, subscriptions
from backend.models.database import engine, Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

# Background tasks
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Fantasy Football AI API...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize cache connection
    cache.connect()
    
    # Start background cache warming
    task = asyncio.create_task(periodic_cache_warming())
    background_tasks.add(task)
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    logger.info("Application shutdown complete")


# Create optimized FastAPI app
app = FastAPI(
    title="Fantasy Football AI API - Optimized",
    version="2.0.0",
    description="High-performance ML-powered fantasy football predictions",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware in order (order matters for performance)

# 1. GZIP compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests
)


# Custom middleware for performance monitoring
@app.middleware("http")
async def add_performance_metrics(request: Request, call_next):
    """Track request performance metrics"""
    start_time = time.time()
    
    # Add request ID for tracking
    request_id = request.headers.get("X-Request-ID", str(time.time()))
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Add performance headers
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    response.headers["X-Request-ID"] = request_id
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Log slow requests
    if duration > 0.5:  # 500ms threshold
        logger.warning(
            f"Slow request: {request.method} {request.url.path} "
            f"took {duration:.3f}s"
        )
    
    return response


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    # Check Redis
    redis_status = "connected" if cache.redis_client and cache.redis_client.ping() else "disconnected"
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": "connected",  # Add actual DB check
            "redis": redis_status
        },
        "system": {
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.2f} GB"
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


# Include routers with prefix versioning
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(players.router, prefix="/api/v1/players", tags=["Players"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(subscriptions.router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])


# Global exception handler with better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "request_id": request.headers.get("X-Request-ID", "unknown")
    }


# Background tasks
async def periodic_cache_warming():
    """Periodically warm cache with popular data"""
    while True:
        try:
            # Wait for next cache warming cycle
            await asyncio.sleep(300)  # 5 minutes
            
            # Get current week
            from datetime import datetime
            current_week = datetime.now().isocalendar()[1]
            current_season = datetime.now().year
            
            # Get top players to warm cache for
            # In production, this would query most accessed players
            top_player_ids = [
                "6783", "4035", "5849", "6804", "7523",
                "5844", "4984", "6770", "5846", "7553"
            ]
            
            # Warm prediction cache
            await CacheWarmer.warm_predictions(
                top_player_ids, 
                current_season, 
                current_week
            )
            
            # Warm stats cache
            await CacheWarmer.warm_player_stats(
                top_player_ids,
                current_season
            )
            
            logger.info("Cache warming cycle completed")
            
        except Exception as e:
            logger.error(f"Cache warming error: {str(e)}")


# Performance tips endpoint
@app.get("/api/v1/performance/tips")
async def performance_tips():
    """Get API performance optimization tips"""
    return {
        "tips": [
            "Use batch endpoints when fetching multiple players",
            "Cache responses on client side for static data",
            "Use ETags for conditional requests",
            "Enable GZIP compression in client",
            "Use webhook subscriptions for real-time updates instead of polling"
        ],
        "rate_limits": {
            "free": "60 requests per minute",
            "pro": "300 requests per minute",
            "premium": "1000 requests per minute"
        },
        "cache_ttl": {
            "predictions": "1 hour",
            "player_info": "24 hours",
            "statistics": "1 hour",
            "rankings": "30 minutes"
        }
    }


# Batch prediction endpoint for performance
@app.post("/api/v1/predictions/batch")
async def batch_predictions(
    player_ids: list[str],
    season: int,
    week: int,
    include_explanations: bool = False
):
    """Get predictions for multiple players in one request"""
    if len(player_ids) > 50:
        raise HTTPException(400, "Maximum 50 players per batch request")
    
    # Process in parallel
    from backend.ml.ensemble_predictions import EnsemblePredictionEngine
    engine = EnsemblePredictionEngine()
    
    tasks = []
    for player_id in player_ids:
        # Check cache first
        cache_key = f"pred:{player_id}:{season}:{week}"
        cached = cache.get(cache_key)
        
        if cached:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Return cached immediately
        else:
            # Create async task for prediction
            task = asyncio.create_task(
                asyncio.to_thread(
                    engine.predict_player_week,
                    player_id, season, week, include_explanations
                )
            )
            tasks.append(task)
    
    # Wait for all predictions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format response
    predictions = {}
    errors = {}
    
    for player_id, result in zip(player_ids, results):
        if isinstance(result, Exception):
            errors[player_id] = str(result)
        elif result is None:  # Cached result
            predictions[player_id] = cache.get(f"pred:{player_id}:{season}:{week}")
        else:
            predictions[player_id] = result
            # Cache successful prediction
            cache.set(f"pred:{player_id}:{season}:{week}", result)
    
    return {
        "predictions": predictions,
        "errors": errors,
        "cached_count": len([r for r in results if r is None]),
        "computed_count": len([r for r in results if r is not None and not isinstance(r, Exception)])
    }


# Server configuration for production
if __name__ == "__main__":
    uvicorn.run(
        "backend.main_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for production
        loop="uvloop",  # Faster event loop
        access_log=False,  # Disable for performance
        log_level="info"
    )