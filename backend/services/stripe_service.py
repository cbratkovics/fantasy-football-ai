"""
Stripe Payment Integration Service
Handles subscriptions and payment processing
"""

import stripe
import os
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from backend.models.database import User, Subscription

logger = logging.getLogger(__name__)

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "price_fantasy_season_20")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")


class StripeService:
    """
    Handles Stripe payment integration
    - Season pass subscriptions ($20)
    - 7-day free trial
    - Webhook processing
    """
    
    def __init__(self):
        self.stripe = stripe
        self.price_id = STRIPE_PRICE_ID
        self.trial_days = 7
    
    async def create_checkout_session(
        self,
        user_id: str,
        user_email: str,
        success_url: str,
        cancel_url: str,
        trial_eligible: bool = True
    ) -> Dict[str, Any]:
        """Create Stripe checkout session for season pass"""
        
        try:
            # Determine if user gets trial
            trial_period_days = self.trial_days if trial_eligible else None
            
            # Create checkout session
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                customer_email=user_email,
                line_items=[{
                    'price': self.price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                subscription_data={
                    'trial_period_days': trial_period_days,
                    'metadata': {
                        'user_id': user_id,
                        'subscription_type': 'season_pass'
                    }
                },
                metadata={
                    'user_id': user_id
                }
            )
            
            return {
                'checkout_url': session.url,
                'session_id': session.id,
                'trial_days': trial_period_days
            }
            
        except Exception as e:
            logger.error(f"Failed to create checkout session: {str(e)}")
            raise
    
    async def create_customer_portal_session(
        self,
        customer_id: str,
        return_url: str
    ) -> str:
        """Create customer portal session for subscription management"""
        
        try:
            session = self.stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            return session.url
            
        except Exception as e:
            logger.error(f"Failed to create portal session: {str(e)}")
            raise
    
    async def handle_webhook(
        self,
        payload: bytes,
        signature: str,
        db: Session
    ) -> Dict[str, Any]:
        """Process Stripe webhook events"""
        
        try:
            # Verify webhook signature
            event = self.stripe.Webhook.construct_event(
                payload, signature, STRIPE_WEBHOOK_SECRET
            )
            
            # Handle different event types
            if event['type'] == 'checkout.session.completed':
                await self._handle_checkout_completed(event['data']['object'], db)
                
            elif event['type'] == 'customer.subscription.updated':
                await self._handle_subscription_updated(event['data']['object'], db)
                
            elif event['type'] == 'customer.subscription.deleted':
                await self._handle_subscription_deleted(event['data']['object'], db)
                
            elif event['type'] == 'invoice.payment_failed':
                await self._handle_payment_failed(event['data']['object'], db)
            
            return {'status': 'success', 'event_type': event['type']}
            
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise
        except Exception as e:
            logger.error(f"Webhook processing failed: {str(e)}")
            raise
    
    async def _handle_checkout_completed(
        self,
        session: Dict[str, Any],
        db: Session
    ):
        """Handle successful checkout"""
        
        user_id = session['metadata']['user_id']
        customer_id = session['customer']
        subscription_id = session['subscription']
        
        # Get subscription details
        subscription = self.stripe.Subscription.retrieve(subscription_id)
        
        # Update user record
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.stripe_customer_id = customer_id
            user.subscription_tier = 'pro'
            
            # Create or update subscription record
            sub_record = db.query(Subscription).filter(
                Subscription.user_id == user_id
            ).first()
            
            if not sub_record:
                sub_record = Subscription(user_id=user_id)
                db.add(sub_record)
            
            sub_record.stripe_subscription_id = subscription_id
            sub_record.status = subscription['status']
            sub_record.current_period_start = datetime.fromtimestamp(
                subscription['current_period_start']
            )
            sub_record.current_period_end = datetime.fromtimestamp(
                subscription['current_period_end']
            )
            sub_record.trial_end = datetime.fromtimestamp(
                subscription['trial_end']
            ) if subscription.get('trial_end') else None
            
            db.commit()
            
            logger.info(f"Subscription created for user {user_id}")
    
    async def _handle_subscription_updated(
        self,
        subscription: Dict[str, Any],
        db: Session
    ):
        """Handle subscription updates"""
        
        sub_record = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription['id']
        ).first()
        
        if sub_record:
            sub_record.status = subscription['status']
            sub_record.current_period_start = datetime.fromtimestamp(
                subscription['current_period_start']
            )
            sub_record.current_period_end = datetime.fromtimestamp(
                subscription['current_period_end']
            )
            
            # Update user tier based on status
            user = db.query(User).filter(User.id == sub_record.user_id).first()
            if user:
                if subscription['status'] in ['active', 'trialing']:
                    user.subscription_tier = 'pro'
                else:
                    user.subscription_tier = 'free'
            
            db.commit()
            
            logger.info(f"Subscription updated: {subscription['id']}")
    
    async def _handle_subscription_deleted(
        self,
        subscription: Dict[str, Any],
        db: Session
    ):
        """Handle subscription cancellation"""
        
        sub_record = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription['id']
        ).first()
        
        if sub_record:
            sub_record.status = 'canceled'
            sub_record.canceled_at = datetime.utcnow()
            
            # Downgrade user to free tier
            user = db.query(User).filter(User.id == sub_record.user_id).first()
            if user:
                user.subscription_tier = 'free'
            
            db.commit()
            
            logger.info(f"Subscription canceled: {subscription['id']}")
    
    async def _handle_payment_failed(
        self,
        invoice: Dict[str, Any],
        db: Session
    ):
        """Handle failed payments"""
        
        subscription_id = invoice['subscription']
        
        sub_record = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if sub_record:
            # Log payment failure
            logger.warning(f"Payment failed for subscription {subscription_id}")
            
            # Could send email notification here
            # For now, Stripe handles retry logic
    
    async def check_subscription_status(
        self,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Check current subscription status for a user"""
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {'status': 'no_user', 'has_access': False}
        
        if user.subscription_tier == 'free':
            return {
                'status': 'free',
                'has_access': False,
                'weekly_predictions_remaining': 5
            }
        
        sub_record = db.query(Subscription).filter(
            Subscription.user_id == user_id
        ).first()
        
        if not sub_record:
            return {'status': 'no_subscription', 'has_access': False}
        
        # Check if subscription is active
        now = datetime.utcnow()
        is_active = (
            sub_record.status in ['active', 'trialing'] and
            sub_record.current_period_end > now
        )
        
        return {
            'status': sub_record.status,
            'has_access': is_active,
            'is_trial': sub_record.status == 'trialing',
            'trial_ends': sub_record.trial_end.isoformat() if sub_record.trial_end else None,
            'current_period_end': sub_record.current_period_end.isoformat(),
            'canceled': sub_record.canceled_at is not None
        }
    
    def calculate_trial_eligibility(self, user_created_at: datetime) -> bool:
        """Determine if user is eligible for free trial"""
        
        # Trial available during August (draft season)
        now = datetime.utcnow()
        is_august = now.month == 8
        
        # User must be new (created within last 30 days)
        is_new_user = (now - user_created_at).days <= 30
        
        return is_august and is_new_user