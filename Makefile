.PHONY: help install test format clean docker-up docker-down

help:
	@echo "Fantasy Football AI - Development Commands"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

format:  ## Format code
	black src tests

clean:  ## Clean temporary files
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down
