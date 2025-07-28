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
