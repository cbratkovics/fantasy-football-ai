#!/bin/bash
set -e

ENVIRONMENT=$1

if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: ./deploy.sh [production|staging]"
    exit 1
fi

echo "Deploying to $ENVIRONMENT..."

# Load environment variables
if [ "$ENVIRONMENT" == "production" ]; then
    source .env.production
else
    source .env.staging
fi

# Build and push Docker images to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -t fantasy-backend ./backend
docker tag fantasy-backend:latest $ECR_REGISTRY/fantasy-backend:latest
docker push $ECR_REGISTRY/fantasy-backend:latest

docker build -t fantasy-frontend ./frontend
docker tag fantasy-frontend:latest $ECR_REGISTRY/fantasy-frontend:latest
docker push $ECR_REGISTRY/fantasy-frontend:latest

# Deploy to ECS or EC2
if [ "$DEPLOYMENT_TYPE" == "ecs" ]; then
    # Update ECS service
    aws ecs update-service --cluster fantasy-cluster --service fantasy-backend --force-new-deployment
    aws ecs update-service --cluster fantasy-cluster --service fantasy-frontend --force-new-deployment
else
    # Deploy to EC2 using docker-compose
    ssh -i ~/.ssh/fantasy-key.pem ec2-user@$EC2_HOST << EOSSH
        cd /home/ec2-user/fantasy-football-ai
        git pull origin main
        docker-compose pull
        docker-compose up -d
        docker system prune -f
EOSSH
fi

echo "Deployment complete!"
