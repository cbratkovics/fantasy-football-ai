#!/bin/bash

# Fantasy Football AI - Project Setup Script
# This script creates all necessary files and directories for the project

set -e  # Exit on error

echo "🏈 Setting up Fantasy Football AI project structure..."

# Create additional directories
echo "📁 Creating directories..."
mkdir -p scripts
mkdir -p models
mkdir -p logs
mkdir -p ssl
mkdir -p infrastructure/terraform

# Create all __init__.py files
echo "📄 Creating Python package files..."
touch backend/__init__.py
touch backend/api/__init__.py
touch backend/ml/__init__.py
touch backend/data/__init__.py
touch backend/models/__init__.py
touch backend/tasks/__init__.py
touch frontend/__init__.py
touch frontend/pages/__init__.py
touch frontend/components/__init__.py

# Create docker-compose.yml
echo "🐳 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: fantasy_db
    environment:
      POSTGRES_USER: ${DB_USER:-fantasy_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-fantasy_pass}
      POSTGRES_DB: ${DB_NAME:-fantasy_football}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - fantasy_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-fantasy_user}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: fantasy_redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - fantasy_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # FastAPI Backend
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: fantasy_backend
    environment:
      DATABASE_URL: postgresql+asyncpg://${DB_USER:-fantasy_user}:${DB_PASSWORD:-fantasy_pass}@postgres:5432/${DB_NAME:-fantasy_football}
      REDIS_URL: redis://redis:6379
      SECRET_KEY: ${SECRET_KEY:-your-secret-key-change-in-production}
      PYTHONPATH: /app
    volumes:
      - ./backend:/app
      - ./models:/app/models
    ports:
      - "8000:8000"
    networks:
      - fantasy_network
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # Streamlit Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: fantasy_frontend
    environment:
      API_BASE_URL: http://backend:8000
      STREAMLIT_SERVER_PORT: 8501
      STREAMLIT_SERVER_ADDRESS: 0.0.0.0
      STREAMLIT_THEME_BASE: light
      STREAMLIT_THEME_PRIMARY_COLOR: "#3b82f6"
    volumes:
      - ./frontend:/app
    ports:
      - "8501:8501"
    networks:
      - fantasy_network
    depends_on:
      - backend
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0

  # Celery Worker for Background Tasks
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: fantasy_celery
    environment:
      DATABASE_URL: postgresql+asyncpg://${DB_USER:-fantasy_user}:${DB_PASSWORD:-fantasy_pass}@postgres:5432/${DB_NAME:-fantasy_football}
      REDIS_URL: redis://redis:6379
      PYTHONPATH: /app
    volumes:
      - ./backend:/app
      - ./models:/app/models
    networks:
      - fantasy_network
    depends_on:
      - postgres
      - redis
      - backend
    command: celery -A tasks worker --loglevel=info

  # Celery Beat for Scheduled Tasks
  celery_beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: fantasy_celery_beat
    environment:
      DATABASE_URL: postgresql+asyncpg://${DB_USER:-fantasy_user}:${DB_PASSWORD:-fantasy_pass}@postgres:5432/${DB_NAME:-fantasy_football}
      REDIS_URL: redis://redis:6379
      PYTHONPATH: /app
    volumes:
      - ./backend:/app
    networks:
      - fantasy_network
    depends_on:
      - postgres
      - redis
      - backend
    command: celery -A tasks beat --loglevel=info

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: fantasy_nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - fantasy_network
    depends_on:
      - backend
      - frontend

networks:
  fantasy_network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
EOF

# Create backend Dockerfile
echo "🐳 Creating backend/Dockerfile..."
cat > backend/Dockerfile << 'EOF'
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create frontend Dockerfile
echo "🐳 Creating frontend/Dockerfile..."
cat > frontend/Dockerfile << 'EOF'
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
EOF

# Create backend requirements.txt
echo "📦 Creating backend/requirements.txt..."
cat > backend/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
redis==5.0.1
aioredis==2.0.1
fastapi-limiter==0.1.5
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
httpx==0.25.2
celery==5.3.4
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
scipy==1.11.4
aiohttp==3.9.1
ratelimit==2.2.1
backoff==2.2.1
python-dotenv==1.0.0
EOF

# Create frontend requirements.txt
echo "📦 Creating frontend/requirements.txt..."
cat > frontend/requirements.txt << 'EOF'
streamlit==1.29.0
pandas==2.1.3
numpy==1.24.3
plotly==5.18.0
requests==2.31.0
python-dotenv==1.0.0
streamlit-authenticator==0.2.3
extra-streamlit-components==0.1.60
EOF

# Create nginx.conf
echo "🔧 Creating infrastructure/nginx.conf..."
cat > infrastructure/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml;
    gzip_disable "MSIE [1-6]\.";

    # Upstream servers
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:8501;
    }

    # HTTP server - redirect to HTTPS
    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL configuration (update paths as needed)
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: ws: wss: data: blob: 'unsafe-inline' 'unsafe-eval'" always;

        # API routes
        location /api/ {
            rewrite ^/api/(.*) /$1 break;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # WebSocket for API
        location /api/ws {
            rewrite ^/api/(.*) /$1 break;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # Streamlit WebSocket
        location /_stcore/stream {
            proxy_pass http://frontend/_stcore/stream;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

# Create .env.example
echo "🔐 Creating .env.example..."
cat > .env.example << 'EOF'
# Database
DB_USER=fantasy_user
DB_PASSWORD=your_secure_password_here
DB_NAME=fantasy_football

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-generate-with-openssl-rand-hex-32

# AWS (for production)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Stripe (for future payments)
STRIPE_SECRET_KEY=sk_test_your_stripe_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Email (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EOF

# Create Makefile
echo "🛠️ Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help build up down logs clean test

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs"
	@echo "  make clean    - Clean up volumes and images"
	@echo "  make test     - Run tests"
	@echo "  make migrate  - Run database migrations"
	@echo "  make shell    - Open shell in backend container"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

test:
	docker-compose run --rm backend pytest

migrate:
	docker-compose run --rm backend alembic upgrade head

shell:
	docker-compose exec backend /bin/bash

# Development
dev-backend:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && streamlit run app.py

# Production deployment
deploy-prod:
	./scripts/deploy.sh production

deploy-staging:
	./scripts/deploy.sh staging

# Database operations
db-backup:
	./scripts/backup-db.sh

db-restore:
	./scripts/restore-db.sh

# ML operations
train-models:
	docker-compose run --rm backend python -m ml.train

update-predictions:
	docker-compose run --rm backend python -m data.update_predictions
EOF

# Create deployment script
echo "🚀 Creating scripts/deploy.sh..."
cat > scripts/deploy.sh << 'EOF'
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
EOF

# Make deploy script executable
chmod +x scripts/deploy.sh

# Create Terraform configuration
echo "☁️ Creating infrastructure/terraform/main.tf..."
cat > infrastructure/terraform/main.tf << 'EOF'
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

variable "db_password" {
  sensitive = true
}

# VPC and basic infrastructure configuration...
# (Full terraform config is in the original artifact)
EOF

# Create README.md (copy from the artifact content)
echo "📝 Creating README.md..."
# Note: This would be the full README content from the project_readme artifact
# For brevity, creating a placeholder that should be replaced with the full content
cat > README.md << 'EOF'
# 🏈 Fantasy Football AI - Production System

> **Advanced Machine Learning System for Fantasy Football Draft Optimization and Weekly Predictions**

[Full README content from the project_readme artifact should be placed here]
EOF

# Create DEPLOYMENT.md (copy from the artifact content)
echo "📝 Creating DEPLOYMENT.md..."
# Note: This would be the full deployment checklist content
# For brevity, creating a placeholder that should be replaced with the full content
cat > DEPLOYMENT.md << 'EOF'
# 🚀 Fantasy Football AI - Deployment Checklist & Next Steps

[Full deployment checklist content from the deployment_checklist artifact should be placed here]
EOF

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
*.egg-info/
dist/
build/

# Environment variables
.env
.env.local
.env.production
.env.staging

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# ML Models
models/*.pkl
models/*.h5
models/*.pt

# Database
*.db
*.sqlite3

# Docker
.docker/

# SSL certificates
ssl/*.pem
ssl/*.key
ssl/*.crt

# OS
.DS_Store
Thumbs.db

# Terraform
*.tfstate
*.tfstate.*
.terraform/
.terraform.lock.hcl

# Testing
.coverage
htmlcov/
.pytest_cache/

# Temporary files
tmp/
temp/
EOF

# Final message
echo "✅ Project setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Review and update the .env file with your configuration"
echo "2. Replace README.md and DEPLOYMENT.md with full content from artifacts"
echo "3. Run 'make build' to build Docker images"
echo "4. Run 'make up' to start the application"
echo ""
echo "🎯 Your Fantasy Football AI project is ready for development!"
EOF

# Make the script executable
chmod +x setup_project.sh