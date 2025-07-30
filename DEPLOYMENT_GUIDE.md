# Fantasy Football AI - Production Deployment Guide

## Deployment Strategy for Subscription-Based Revenue

### Overview
This guide covers deploying Fantasy Football AI as a profitable SaaS platform with tiered subscriptions (Free, Pro, Premium).

## Prerequisites

### 1. Domain & SSL
```bash
# Register domain (recommended: fantasyfootballai.com)
# Setup Cloudflare for DNS, CDN, and DDoS protection
# Configure SSL certificates (automatic with Cloudflare)
```

### 2. Cloud Infrastructure
**Recommended: AWS (or DigitalOcean for cost-effectiveness)**

#### AWS Architecture:
- **Application**: ECS Fargate or EC2 Auto Scaling Group
- **Database**: RDS PostgreSQL (Multi-AZ for production)
- **Cache**: ElastiCache Redis (Cluster Mode)
- **Load Balancer**: Application Load Balancer
- **Storage**: S3 for model artifacts and static files
- **CDN**: CloudFront
- **Monitoring**: CloudWatch + Datadog

#### DigitalOcean Architecture (Cost-Effective):
- **Application**: App Platform or Kubernetes
- **Database**: Managed PostgreSQL
- **Cache**: Managed Redis
- **Load Balancer**: Load Balancer
- **Storage**: Spaces (S3-compatible)

## Subscription & Payment Integration

### Stripe Integration
```python
# Add to requirements.txt
stripe==5.5.0
python-dotenv==0.19.0

# Environment variables
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### Subscription Tiers
```python
SUBSCRIPTION_PLANS = {
    'free': {
        'price_id': None,
        'price': 0,
        'features': [
            '60 predictions per day',
            'Basic player analysis',
            'Tier rankings',
            'Standard scoring only'
        ],
        'limits': {
            'requests_per_day': 60,
            'batch_size': 5,
            'historical_data_weeks': 4
        }
    },
    'pro': {
        'price_id': 'price_1234567890',  # Stripe price ID
        'price': 9.99,
        'features': [
            '300 predictions per day',
            'Advanced analytics (Efficiency Ratio)',
            'Momentum detection alerts',
            'All scoring formats',
            'Weather adjustments',
            'Injury impact analysis',
            'Export capabilities'
        ],
        'limits': {
            'requests_per_day': 300,
            'batch_size': 20,
            'historical_data_weeks': 12
        }
    },
    'premium': {
        'price_id': 'price_0987654321',
        'price': 19.99,
        'features': [
            'Unlimited predictions',
            'All Pro features',
            'Real-time WebSocket updates',
            'Trade analyzer',
            'Custom league integration',
            'API access',
            'Priority support'
        ],
        'limits': {
            'requests_per_day': 'unlimited',
            'batch_size': 50,
            'historical_data_weeks': 52
        }
    }
}
```

## Infrastructure Setup

### 1. Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "backend.main_optimized:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football
      - REDIS_URL=redis://redis:6379/0
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fantasy_football
      POSTGRES_USER: fantasy_user
      POSTGRES_PASSWORD: fantasy_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  streamlit:
    build: 
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://app:8000
    depends_on:
      - app

volumes:
  postgres_data:
```

### 3. Kubernetes Deployment (Production)
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fantasy-ai

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fantasy-ai-api
  namespace: fantasy-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fantasy-ai-api
  template:
    metadata:
      labels:
        app: fantasy-ai-api
    spec:
      containers:
      - name: api
        image: your-registry/fantasy-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fantasy-ai-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: fantasy-ai-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fantasy-ai-service
  namespace: fantasy-ai
spec:
  selector:
    app: fantasy-ai-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fantasy-ai-ingress
  namespace: fantasy-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.fantasyfootballai.com
    secretName: fantasy-ai-tls
  rules:
  - host: api.fantasyfootballai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fantasy-ai-service
            port:
              number: 80
```

## Monetization Implementation

### 1. User Authentication & Subscriptions
```python
# backend/models/subscription.py
from sqlalchemy import Column, String, DateTime, Boolean, Float
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class Subscription(Base):
    __tablename__ = 'subscriptions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    stripe_subscription_id = Column(String, unique=True)
    plan_type = Column(String)  # free, pro, premium
    status = Column(String)  # active, canceled, past_due
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancel_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="subscription")
```

### 2. Stripe Webhook Handler
```python
# backend/api/webhooks.py
from fastapi import APIRouter, Request, HTTPException
import stripe
import os

router = APIRouter()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle subscription events
    if event['type'] == 'customer.subscription.created':
        await handle_subscription_created(event['data']['object'])
    elif event['type'] == 'customer.subscription.updated':
        await handle_subscription_updated(event['data']['object'])
    elif event['type'] == 'customer.subscription.deleted':
        await handle_subscription_canceled(event['data']['object'])
    elif event['type'] == 'invoice.payment_failed':
        await handle_payment_failed(event['data']['object'])
    
    return {"status": "success"}
```

### 3. Subscription Management API
```python
# backend/api/subscriptions.py
from fastapi import APIRouter, Depends, HTTPException
import stripe

router = APIRouter()

@router.post("/subscribe")
async def create_subscription(
    plan_type: str,
    current_user: User = Depends(get_current_user)
):
    """Create new subscription"""
    plan_config = SUBSCRIPTION_PLANS.get(plan_type)
    if not plan_config or not plan_config['price_id']:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    try:
        # Create Stripe customer if doesn't exist
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': str(current_user.id)}
            )
            current_user.stripe_customer_id = customer.id
            # Save to database
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=current_user.stripe_customer_id,
            items=[{'price': plan_config['price_id']}],
            metadata={'user_id': str(current_user.id)}
        )
        
        return {
            "subscription_id": subscription.id,
            "client_secret": subscription.latest_invoice.payment_intent.client_secret
        }
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cancel")
async def cancel_subscription(current_user: User = Depends(get_current_user)):
    """Cancel subscription"""
    if not current_user.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    
    stripe.Subscription.modify(
        current_user.stripe_subscription_id,
        cancel_at_period_end=True
    )
    
    return {"status": "subscription will cancel at period end"}
```

## Frontend Integration

### 1. Next.js Landing Page
```jsx
// pages/pricing.js
import { useState } from 'react'
import { loadStripe } from '@stripe/stripe-js'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY)

export default function Pricing() {
    const [loading, setLoading] = useState(false)
    
    const handleSubscribe = async (planType) => {
        setLoading(true)
        
        const response = await fetch('/api/subscribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userToken}`
            },
            body: JSON.stringify({ plan_type: planType })
        })
        
        const { client_secret } = await response.json()
        
        const stripe = await stripePromise
        const { error } = await stripe.confirmCardPayment(client_secret)
        
        if (error) {
            console.error('Payment failed:', error)
        } else {
            // Redirect to success page
            window.location.href = '/dashboard'
        }
        
        setLoading(false)
    }
    
    return (
        <div className="pricing-container">
            <h1>Choose Your Plan</h1>
            
            <div className="pricing-cards">
                {/* Free Plan */}
                <div className="card">
                    <h3>Free</h3>
                    <p className="price">$0/month</p>
                    <ul>
                        <li>60 predictions per day</li>
                        <li>Basic player analysis</li>
                        <li>Tier rankings</li>
                    </ul>
                    <button onClick={() => handleSubscribe('free')}>
                        Get Started Free
                    </button>
                </div>
                
                {/* Pro Plan */}
                <div className="card featured">
                    <h3>Pro</h3>
                    <p className="price">$9.99/month</p>
                    <ul>
                        <li>300 predictions per day</li>
                        <li>Advanced analytics</li>
                        <li>Momentum detection</li>
                        <li>Weather adjustments</li>
                    </ul>
                    <button onClick={() => handleSubscribe('pro')}>
                        Upgrade to Pro
                    </button>
                </div>
                
                {/* Premium Plan */}
                <div className="card">
                    <h3>Premium</h3>
                    <p className="price">$19.99/month</p>
                    <ul>
                        <li>Unlimited predictions</li>
                        <li>Real-time updates</li>
                        <li>Trade analyzer</li>
                        <li>API access</li>
                    </ul>
                    <button onClick={() => handleSubscribe('premium')}>
                        Go Premium
                    </button>
                </div>
            </div>
        </div>
    )
}
```

## Analytics & Monitoring

### 1. Revenue Tracking
```python
# backend/analytics/revenue.py
from sqlalchemy import func
from datetime import datetime, timedelta

class RevenueAnalytics:
    def get_mrr(self) -> float:
        """Calculate Monthly Recurring Revenue"""
        with SessionLocal() as db:
            active_subs = db.query(Subscription).filter(
                Subscription.status == 'active'
            ).all()
            
            mrr = 0
            for sub in active_subs:
                plan = SUBSCRIPTION_PLANS.get(sub.plan_type, {})
                mrr += plan.get('price', 0)
            
            return mrr
    
    def get_churn_rate(self, period_days: int = 30) -> float:
        """Calculate churn rate"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        with SessionLocal() as db:
            # Users who were active at start of period
            active_start = db.query(Subscription).filter(
                Subscription.status == 'active',
                Subscription.created_at <= start_date
            ).count()
            
            # Users who canceled during period
            canceled = db.query(Subscription).filter(
                Subscription.status == 'canceled',
                Subscription.cancel_at.between(start_date, end_date)
            ).count()
            
            return (canceled / active_start) * 100 if active_start > 0 else 0
```

### 2. Usage Analytics
```python
# backend/analytics/usage.py
class UsageAnalytics:
    def track_api_usage(self, user_id: str, endpoint: str, cost: int = 1):
        """Track API usage for billing and analytics"""
        # Store in Redis for real-time tracking
        today = datetime.utcnow().strftime('%Y-%m-%d')
        key = f"usage:{user_id}:{today}"
        
        cache.redis_client.hincrby(key, endpoint, cost)
        cache.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    def get_user_usage(self, user_id: str, days: int = 30) -> Dict:
        """Get user's usage statistics"""
        usage_data = {}
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
            key = f"usage:{user_id}:{date}"
            daily_usage = cache.redis_client.hgetall(key)
            usage_data[date] = daily_usage
        
        return usage_data
```

## Deployment Steps

### 1. Local Development
```bash
# 1. Clone and setup
git clone your-repo
cd fantasy-football-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run with Docker
docker-compose up -d

# 4. Initialize database
python scripts/init_database.py
python scripts/load_sample_data.py
```

### 2. Production Deployment

#### Option A: DigitalOcean App Platform (Easiest)
```yaml
# .do/app.yaml
name: fantasy-football-ai
services:
- name: api
  source_dir: /
  github:
    repo: your-username/fantasy-football-ai
    branch: main
  run_command: uvicorn backend.main_optimized:app --host 0.0.0.0 --port 8080
  environment_slug: python
  instance_count: 2
  instance_size_slug: professional-xs
  envs:
  - key: DATABASE_URL
    value: ${db.DATABASE_URL}
  - key: REDIS_URL
    value: ${redis.DATABASE_URL}
  - key: STRIPE_SECRET_KEY
    value: ${STRIPE_SECRET_KEY}
    type: SECRET

databases:
- name: db
  engine: PG
  version: "15"
  size: db-s-1vcpu-1gb

- name: redis
  engine: REDIS
  version: "7"
  size: db-s-1vcpu-1gb
```

#### Option B: AWS ECS (Scalable)
```bash
# 1. Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker build -t fantasy-ai .
docker tag fantasy-ai:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/fantasy-ai:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/fantasy-ai:latest

# 2. Deploy with Terraform or CDK
# See terraform/ directory for infrastructure as code
```

### 3. Domain & SSL Setup
```bash
# 1. Point domain to your deployment
# A record: api.fantasyfootballai.com -> your-server-ip
# A record: app.fantasyfootballai.com -> your-server-ip

# 2. Setup SSL with Let's Encrypt (if not using Cloudflare)
certbot --nginx -d api.fantasyfootballai.com -d app.fantasyfootballai.com
```

## Marketing & Growth Strategy

### 1. Free Tier Strategy
- **60 predictions/day**: Enough to evaluate 1-2 weekly lineups
- **Basic features**: Hook users with core value proposition
- **Upgrade prompts**: Show premium features in action

### 2. Content Marketing
- **Blog**: Weekly fantasy insights using your AI predictions
- **YouTube**: "AI vs Experts" prediction competitions  
- **Podcast**: Fantasy football with AI-powered insights
- **Social Media**: Daily player recommendations and alerts

### 3. SEO Strategy
- Target keywords: "fantasy football AI", "player predictions", "draft rankings"
- Create prediction pages for every NFL player
- Weekly content: "Week X AI Predictions"

### 4. Partnership Opportunities
- **Fantasy Football Podcasts**: Sponsor with free API access
- **Content Creators**: Affiliate program
- **Fantasy Platforms**: White-label API integration

## Revenue Projections

### Conservative Estimates (Year 1)
- **Free Users**: 10,000 (90%)
- **Pro Users**: 1,000 (9%) × $9.99 = $9,990/month
- **Premium Users**: 100 (1%) × $19.99 = $1,999/month
- **Total MRR**: ~$12,000/month
- **Annual Revenue**: ~$144,000

### Growth Targets (Year 2)
- **Free Users**: 50,000
- **Pro Users**: 8,000 × $9.99 = $79,920/month  
- **Premium Users**: 2,000 × $19.99 = $39,980/month
- **Total MRR**: ~$120,000/month
- **Annual Revenue**: ~$1.4M

## Launch Checklist

### Pre-Launch
- [ ] Complete feature development
- [ ] Set up production infrastructure  
- [ ] Configure Stripe payments
- [ ] Create landing page and pricing
- [ ] Legal: Terms of Service, Privacy Policy
- [ ] Set up analytics and monitoring
- [ ] Load test the system
- [ ] Create onboarding flow

### Launch Week
- [ ] Deploy to production
- [ ] Announce on social media
- [ ] Submit to Product Hunt
- [ ] Email marketing campaign
- [ ] Influencer outreach
- [ ] Monitor for issues

### Post-Launch
- [ ] Gather user feedback
- [ ] Optimize conversion funnel  
- [ ] A/B test pricing and features
- [ ] Scale infrastructure as needed
- [ ] Plan feature roadmap

## Maintenance & Scaling

### Weekly Tasks
- Monitor system health and performance
- Update player data and predictions
- Analyze user behavior and revenue metrics
- Content creation and social media

### Monthly Tasks  
- Review and optimize infrastructure costs
- Analyze churn and implement retention strategies
- Plan and develop new features
- Security audits and updates

### Scaling Considerations
- **Database**: Read replicas, connection pooling
- **Cache**: Redis Cluster, CDN optimization
- **API**: Auto-scaling, load balancing
- **ML Models**: Model serving optimization, A/B testing

This deployment guide provides a complete path from development to profitable SaaS operation. The key is starting with a solid technical foundation and focusing on user acquisition through the free tier while optimizing conversion to paid plans.