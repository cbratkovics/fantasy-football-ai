"""
Payment and Subscription API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session
import stripe
import logging

from backend.models.database import get_db, User
from backend.models.schemas import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    SubscriptionStatus
)
from backend.services.stripe_service import StripeService
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
stripe_service = StripeService()


@router.post("/checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create Stripe checkout session for season pass
    - $20/season with 7-day free trial (August only)
    """
    
    try:
        # Check if user already has active subscription
        subscription_status = await stripe_service.check_subscription_status(
            current_user.id, db
        )
        
        if subscription_status['has_access']:
            raise HTTPException(
                status_code=400,
                detail="You already have an active subscription"
            )
        
        # Determine trial eligibility
        trial_eligible = stripe_service.calculate_trial_eligibility(
            current_user.created_at
        )
        
        # Create checkout session
        session = await stripe_service.create_checkout_session(
            user_id=current_user.id,
            user_email=current_user.email,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            trial_eligible=trial_eligible
        )
        
        return CheckoutSessionResponse(
            checkout_url=session['checkout_url'],
            session_id=session['session_id'],
            trial_days=session['trial_days']
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Payment processing error. Please try again."
        )
    except Exception as e:
        logger.error(f"Checkout session creation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session"
        )


@router.get("/subscription/status", response_model=SubscriptionStatus)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's subscription status"""
    
    try:
        status = await stripe_service.check_subscription_status(
            current_user.id, db
        )
        
        return SubscriptionStatus(**status)
        
    except Exception as e:
        logger.error(f"Failed to get subscription status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve subscription status"
        )


@router.post("/customer-portal")
async def create_customer_portal_session(
    return_url: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create Stripe customer portal session for subscription management
    Allows users to update payment method, cancel subscription, etc.
    """
    
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=400,
            detail="No active subscription found"
        )
    
    try:
        portal_url = await stripe_service.create_customer_portal_session(
            customer_id=current_user.stripe_customer_id,
            return_url=return_url
        )
        
        return {"portal_url": portal_url}
        
    except Exception as e:
        logger.error(f"Portal session creation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create portal session"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
    db: Session = Depends(get_db)
):
    """
    Handle Stripe webhook events
    - subscription.created
    - subscription.updated
    - subscription.deleted
    - invoice.payment_failed
    """
    
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing stripe signature")
    
    try:
        # Get raw body
        payload = await request.body()
        
        # Process webhook
        result = await stripe_service.handle_webhook(
            payload=payload,
            signature=stripe_signature,
            db=db
        )
        
        return result
        
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed"
        )


@router.get("/pricing")
async def get_pricing_info():
    """Get current pricing information"""
    
    return {
        "season_pass": {
            "price": 20.00,
            "currency": "USD",
            "period": "season",
            "trial_days": 7,
            "trial_available": "August only",
            "features": [
                "Unlimited predictions all season",
                "Advanced lineup optimizer",
                "Real-time injury updates",
                "Custom scoring support",
                "Email/SMS alerts",
                "API access"
            ]
        },
        "free_tier": {
            "price": 0,
            "features": [
                "5 predictions per week",
                "Basic player rankings",
                "Public accuracy reports"
            ]
        }
    }