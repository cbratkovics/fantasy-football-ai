"""
Subscription Service for managing user tiers and usage tracking
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.models.database import get_db, User, Subscription, PredictionUsage
from backend.core.cache import get_redis_client

logger = logging.getLogger(__name__)

class SubscriptionService:
    """Service for managing user subscriptions and usage"""
    
    SUBSCRIPTION_TIERS = {
        "scout": {
            "name": "Scout",
            "price": 0,
            "ai_queries_per_week": 20,
            "leagues": 1,
            "features": ["basic_predictions", "player_rankings"]
        },
        "analyst": {
            "name": "Analyst", 
            "price": 7.99,
            "ai_queries_per_week": None,  # Unlimited
            "leagues": None,  # Unlimited
            "features": ["all_predictions", "ai_assistant", "advanced_analytics", "trade_analyzer"]
        },
        "gm": {
            "name": "GM",
            "price": 19.99,
            "ai_queries_per_week": None,  # Unlimited
            "leagues": None,  # Unlimited
            "features": ["all_analyst_features", "custom_ai_training", "white_label", "priority_support"]
        }
    }
    
    def __init__(self):
        self.redis_client = None
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await get_redis_client()
    
    async def get_user_tier(self, user_id: str) -> str:
        """Get user's current subscription tier"""
        try:
            # Try Redis cache first
            cache_key = f"user_tier:{user_id}"
            if self.redis_client:
                cached_tier = await self.redis_client.get(cache_key)
                if cached_tier:
                    return cached_tier.decode()
            
            # Query database
            with next(get_db()) as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return "scout"  # Default tier
                
                subscription = db.query(Subscription).filter(
                    Subscription.user_id == user_id,
                    Subscription.status == "active"
                ).first()
                
                if not subscription:
                    tier = "scout"
                else:
                    # Determine tier based on subscription
                    if subscription.stripe_subscription_id:
                        # This would normally check Stripe for the actual plan
                        # For now, we'll use a simple mapping
                        tier = user.subscription_tier or "scout"
                    else:
                        tier = "scout"
                
                # Cache the result
                if self.redis_client:
                    await self.redis_client.setex(cache_key, 300, tier)  # 5 min cache
                
                return tier
                
        except Exception as e:
            logger.error(f"Failed to get user tier for {user_id}: {e}")
            return "scout"  # Safe default
    
    async def get_user_usage(self, user_id: str, period: str = "week") -> Dict:
        """Get user's current usage statistics"""
        try:
            if period == "week":
                start_date = datetime.now() - timedelta(days=7)
            elif period == "month":
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = datetime.now() - timedelta(days=1)
            
            with next(get_db()) as db:
                usage = db.query(PredictionUsage).filter(
                    PredictionUsage.user_id == user_id,
                    PredictionUsage.created_at >= start_date
                ).all()
                
                total_queries = len(usage)
                total_predictions = sum(u.predictions_count for u in usage)
                
                return {
                    "period": period,
                    "total_queries": total_queries,
                    "total_predictions": total_predictions,
                    "start_date": start_date.isoformat(),
                    "end_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage for {user_id}: {e}")
            return {
                "period": period,
                "total_queries": 0,
                "total_predictions": 0,
                "error": str(e)
            }
    
    async def check_usage_limits(self, user_id: str, requested_queries: int = 1) -> Dict:
        """Check if user can make additional queries"""
        try:
            tier = await self.get_user_tier(user_id)
            tier_config = self.SUBSCRIPTION_TIERS[tier]
            
            # Unlimited for paid tiers
            if tier_config["ai_queries_per_week"] is None:
                return {
                    "allowed": True,
                    "remaining": None,
                    "tier": tier
                }
            
            # Check current usage for free tier
            usage = await self.get_user_usage(user_id, "week")
            current_queries = usage["total_queries"]
            limit = tier_config["ai_queries_per_week"]
            
            remaining = max(0, limit - current_queries)
            allowed = remaining >= requested_queries
            
            return {
                "allowed": allowed,
                "remaining": remaining,
                "limit": limit,
                "current": current_queries,
                "tier": tier
            }
            
        except Exception as e:
            logger.error(f"Failed to check usage limits for {user_id}: {e}")
            return {
                "allowed": False,
                "error": str(e)
            }
    
    async def record_usage(self, user_id: str, query_type: str, tokens_used: int = 0):
        """Record user query usage"""
        try:
            with next(get_db()) as db:
                # Find or create usage record for current week
                week_start = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=datetime.now().weekday())
                
                usage = db.query(PredictionUsage).filter(
                    PredictionUsage.user_id == user_id,
                    PredictionUsage.week_start == week_start
                ).first()
                
                if not usage:
                    usage = PredictionUsage(
                        user_id=user_id,
                        week_start=week_start,
                        predictions_count=1
                    )
                    db.add(usage)
                else:
                    usage.predictions_count += 1
                
                db.commit()
                
                # Invalidate cache
                if self.redis_client:
                    cache_key = f"user_tier:{user_id}"
                    await self.redis_client.delete(cache_key)
                
        except Exception as e:
            logger.error(f"Failed to record usage for {user_id}: {e}")
    
    async def upgrade_user_subscription(self, user_id: str, tier: str, stripe_subscription_id: str):
        """Upgrade user to paid subscription"""
        try:
            with next(get_db()) as db:
                # Update user tier
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    user.subscription_tier = tier
                
                # Update or create subscription record
                subscription = db.query(Subscription).filter(
                    Subscription.user_id == user_id
                ).first()
                
                if subscription:
                    subscription.status = "active"
                    subscription.stripe_subscription_id = stripe_subscription_id
                    subscription.current_period_start = datetime.now()
                    subscription.current_period_end = datetime.now() + timedelta(days=30)
                else:
                    subscription = Subscription(
                        user_id=user_id,
                        stripe_subscription_id=stripe_subscription_id,
                        status="active",
                        current_period_start=datetime.now(),
                        current_period_end=datetime.now() + timedelta(days=30)
                    )
                    db.add(subscription)
                
                db.commit()
                
                # Clear cache
                if self.redis_client:
                    cache_key = f"user_tier:{user_id}"
                    await self.redis_client.delete(cache_key)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to upgrade subscription for {user_id}: {e}")
            return False
    
    def get_tier_features(self, tier: str) -> Dict:
        """Get features available for a subscription tier"""
        return self.SUBSCRIPTION_TIERS.get(tier, self.SUBSCRIPTION_TIERS["scout"])
    
    async def get_subscription_analytics(self) -> Dict:
        """Get subscription analytics for admin dashboard"""
        try:
            with next(get_db()) as db:
                total_users = db.query(func.count(User.id)).scalar()
                
                active_subscriptions = db.query(func.count(Subscription.id)).filter(
                    Subscription.status == "active"
                ).scalar()
                
                tier_distribution = {}
                for tier in ["scout", "analyst", "gm"]:
                    count = db.query(func.count(User.id)).filter(
                        User.subscription_tier == tier
                    ).scalar()
                    tier_distribution[tier] = count
                
                return {
                    "total_users": total_users,
                    "active_subscriptions": active_subscriptions,
                    "tier_distribution": tier_distribution,
                    "conversion_rate": active_subscriptions / total_users if total_users > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get subscription analytics: {e}")
            return {
                "error": str(e)
            }