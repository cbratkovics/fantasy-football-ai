"""Subscriptions API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from backend.models.database import User, get_db
from backend.models.schemas import SubscriptionTier

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return {
        "plans": [
            {
                "tier": SubscriptionTier.FREE,
                "name": "Free Tier",
                "price": 0,
                "features": [
                    "Basic player rankings",
                    "Limited predictions (5 players)",
                    "100 API calls/hour"
                ]
            },
            {
                "tier": SubscriptionTier.PRO,
                "name": "Pro Tier",
                "price": 9.99,
                "features": [
                    "Full GMM draft tiers",
                    "Unlimited predictions",
                    "Draft assistant",
                    "Waiver wire AI",
                    "1,000 API calls/hour"
                ]
            },
            {
                "tier": SubscriptionTier.PREMIUM,
                "name": "Premium Tier",
                "price": 19.99,
                "features": [
                    "Everything in Pro",
                    "Custom league scoring",
                    "Historical analysis",
                    "Priority support",
                    "Beta features",
                    "10,000 API calls/hour"
                ]
            }
        ]
    }


@router.post("/upgrade")
async def upgrade_subscription(
    tier: SubscriptionTier,
    db: Session = Depends(get_db)
):
    """Upgrade user subscription"""
    # For MVP, return success
    return {
        "success": True,
        "message": f"Subscription upgraded to {tier}",
        "tier": tier,
        "expires_at": "2025-01-29T00:00:00Z"
    }


@router.post("/cancel")
async def cancel_subscription(db: Session = Depends(get_db)):
    """Cancel user subscription"""
    return {
        "success": True,
        "message": "Subscription cancelled",
        "tier": SubscriptionTier.FREE,
        "cancelled_at": datetime.utcnow().isoformat()
    }