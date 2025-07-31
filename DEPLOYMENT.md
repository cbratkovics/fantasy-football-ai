# 🚀 Fantasy Football AI - Complete Deployment Guide

## Overview

This guide covers all deployment options for Fantasy Football AI, from local development to production deployment on various platforms.

## Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/cbratkovics/fantasy-football-ai.git
cd fantasy-football-ai

# Setup environment
cp backend/.env.example backend/.env
cp frontend-next/.env.example frontend-next/.env

# Start with Docker Compose
docker-compose up -d

# Or start services individually
cd backend && uvicorn main:app --reload
cd frontend-next && npm run dev
```

## Deployment Options

### 1. Railway Deployment (Recommended for Quick Start)

Railway provides the easiest deployment with automatic scaling and built-in PostgreSQL/Redis.

```bash
# Deploy both services
./deploy-to-railway.sh

# Or deploy manually
cd frontend-next && railway up
cd backend && railway up
```

**Key Features:**
- Automatic SSL certificates
- Built-in PostgreSQL and Redis
- GitHub integration for CI/CD
- Zero-downtime deployments

### 2. AWS Deployment (Recommended for Production)

#### Architecture:
- **Compute**: ECS Fargate or EC2 Auto Scaling
- **Database**: RDS PostgreSQL (Multi-AZ)
- **Cache**: ElastiCache Redis
- **Load Balancer**: Application Load Balancer
- **Storage**: S3 for models and static files
- **CDN**: CloudFront

#### Deployment Steps:
```bash
# Use Terraform configuration
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Deploy with AWS CLI
aws ecs update-service --cluster fantasy-football --service backend --force-new-deployment
```

### 3. DigitalOcean Deployment (Cost-Effective)

#### Architecture:
- **App Platform** for frontend and backend
- **Managed PostgreSQL** database
- **Managed Redis** for caching
- **Spaces** for object storage

```bash
# Deploy with doctl
doctl apps create --spec .do/app.yaml
```

### 4. Docker Deployment (Self-Hosted)

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Or use individual containers
docker build -t fantasy-backend ./backend
docker build -t fantasy-frontend ./frontend-next

docker run -d -p 8000:8000 fantasy-backend
docker run -d -p 3000:3000 fantasy-frontend
```

## Environment Configuration

### Backend Environment Variables
```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fantasy_football

# Redis
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# Stripe Payments
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Frontend URL (for CORS)
FRONTEND_URL=https://yourapp.com
```

### Frontend Environment Variables
```env
# API Configuration
NEXT_PUBLIC_API_URL=https://api.yourapp.com

# Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
CLERK_SECRET_KEY=sk_live_...

# Analytics (optional)
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
```

## Production Checklist

### Pre-Deployment
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database migrations run
- [ ] Redis cache configured
- [ ] Stripe webhooks configured
- [ ] CORS settings updated
- [ ] Rate limiting configured
- [ ] Monitoring setup (Datadog/New Relic)

### Security
- [ ] Secrets stored in secure vault (AWS Secrets Manager, etc.)
- [ ] Database backups configured
- [ ] WAF rules configured
- [ ] DDoS protection enabled
- [ ] API rate limiting active
- [ ] Input validation enabled

### Performance
- [ ] CDN configured for static assets
- [ ] Database connection pooling
- [ ] Redis caching active
- [ ] Auto-scaling configured
- [ ] Health checks configured

## Monitoring & Maintenance

### Health Checks
- Backend: `GET /health`
- Frontend: `GET /`

### Logs
```bash
# Railway
railway logs --service fantasy-football-backend

# Docker
docker logs fantasy-backend

# AWS
aws logs tail /ecs/fantasy-football
```

### Database Maintenance
```bash
# Run migrations
alembic upgrade head

# Backup database
pg_dump $DATABASE_URL > backup.sql
```

## Scaling Considerations

### Horizontal Scaling
- Backend: Add more API instances
- Frontend: Use CDN and edge functions
- Database: Read replicas for queries
- Cache: Redis cluster mode

### Vertical Scaling
- Increase instance sizes based on CPU/memory usage
- Optimize database queries with indexes
- Implement query result caching

## Cost Optimization

### Estimated Monthly Costs:
- **Railway**: $20-50/month
- **AWS**: $100-500/month (depends on scale)
- **DigitalOcean**: $50-200/month
- **Self-Hosted**: Variable (server costs)

### Tips:
1. Use spot instances for ML training
2. Implement aggressive caching
3. Optimize Docker images
4. Use CDN for static assets
5. Schedule non-critical tasks during off-peak

## Troubleshooting

### Common Issues:
1. **Port conflicts**: Ensure ports 3000, 8000 are available
2. **Memory issues**: Increase Docker memory limit
3. **Database connections**: Check connection pool settings
4. **CORS errors**: Verify FRONTEND_URL in backend
5. **SSL issues**: Check certificate configuration

### Debug Commands:
```bash
# Check service health
curl https://api.yourapp.com/health

# Test database connection
python scripts/test_db_connection.py

# Verify Redis
redis-cli ping
```

## Support

For deployment issues:
- GitHub Issues: https://github.com/cbratkovics/fantasy-football-ai/issues
- Documentation: This guide
- Logs: Check application and infrastructure logs

Remember to always test in staging before deploying to production!