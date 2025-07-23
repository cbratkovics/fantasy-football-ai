# Project Structure - Fantasy Football AI Assistant


```
fantasy-football-ai/
├── 📄 README.md                    # Project overview & quick start
├── 📄 ARCHITECTURE.md              # Technical architecture docs
├── 📄 CHANGELOG.md                 # Version history
├── 📄 LICENSE                      # MIT license
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .env.example                 # Environment template
├── 📄 pyproject.toml               # Modern Python packaging
├── 📄 requirements.txt             # Production dependencies
├── 📄 requirements-dev.txt         # Development dependencies
├── 📄 docker-compose.yml           # Container orchestration
├── 📄 Dockerfile                   # Multi-stage container
├── 📄 Makefile                     # Development shortcuts
│
├── 📁 .github/                     # GitHub automation
│   ├── workflows/
│   │   ├── ci.yml                  # Continuous integration
│   │   ├── cd.yml                  # Continuous deployment
│   │   └── security.yml            # Security scanning
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
│
├── 📁 src/                         # Source code (importable package)
│   ├── fantasy_ai/
│   │   ├── __init__.py
│   │   ├── core/                   # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── models/             # ML models & predictors
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # Abstract base classes
│   │   │   │   ├── predictor.py    # Your neural network wrapper
│   │   │   │   ├── clustering.py   # Your GMM clustering
│   │   │   │   └── ensemble.py     # Model ensemble logic
│   │   │   ├── data/               # Data management
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sources/        # API integrations
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── sleeper.py  # Sleeper API
│   │   │   │   │   ├── espn.py     # ESPN API
│   │   │   │   │   └── nfl.py      # NFL.com API
│   │   │   │   ├── processors/     # Data processing
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── features.py # Feature engineering
│   │   │   │   │   ├── validation.py # Data validation
│   │   │   │   │   └── cache.py    # Caching layer
│   │   │   │   └── storage/        # Database operations
│   │   │   │       ├── __init__.py
│   │   │   │       ├── models.py   # SQLAlchemy models
│   │   │   │       ├── migrations/ # Alembic migrations
│   │   │   │       └── repositories.py # Data access layer
│   │   │   ├── services/           # Business services
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prediction.py   # Prediction service
│   │   │   │   ├── analysis.py     # Player analysis service
│   │   │   │   └── recommendations.py # Waiver wire service
│   │   │   └── utils/              # Utilities
│   │   │       ├── __init__.py
│   │   │       ├── config.py       # Configuration management
│   │   │       ├── logging.py      # Logging setup
│   │   │       ├── monitoring.py   # Performance monitoring
│   │   │       └── exceptions.py   # Custom exceptions
│   │   ├── api/                    # API layer (FastAPI)
│   │   │   ├── __init__.py
│   │   │   ├── main.py             # FastAPI app
│   │   │   ├── routers/            # API routes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── players.py      # Player endpoints
│   │   │   │   ├── predictions.py  # Prediction endpoints
│   │   │   │   └── health.py       # Health checks
│   │   │   ├── middleware/         # Custom middleware
│   │   │   └── schemas/            # Pydantic schemas
│   │   └── web/                    # Web interface (Streamlit)
│   │       ├── __init__.py
│   │       ├── app.py              # Main Streamlit app
│   │       ├── components/         # Reusable components
│   │       │   ├── __init__.py
│   │       │   ├── dashboard.py    # Dashboard components
│   │       │   ├── player_analysis.py # Player analysis
│   │       │   ├── charts.py       # Chart components
│   │       │   └── sidebar.py      # Sidebar components
│   │       ├── pages/              # Multi-page app
│   │       │   ├── __init__.py
│   │       │   ├── 1_🏠_Dashboard.py
│   │       │   ├── 2_🔍_Player_Analysis.py
│   │       │   ├── 3_💹_Waiver_Wire.py
│   │       │   └── 4_📊_Analytics.py
│   │       ├── styles/             # CSS & styling
│   │       │   ├── main.css
│   │       │   └── components.css
│   │       └── utils/              # Web utilities
│   │           ├── __init__.py
│   │           ├── session.py      # Session management
│   │           └── cache.py        # Streamlit caching
│
├── 📁 tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_models/
│   │   ├── test_data/
│   │   └── test_services/
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   └── test_database/
│   └── e2e/                        # End-to-end tests
│       ├── __init__.py
│       └── test_workflows/
│
├── 📁 models/                      # ML model artifacts
│   ├── trained/                    # Production models
│   │   ├── neural_network_v1.0.h5
│   │   ├── gmm_model_v1.0.pkl
│   │   └── scaler_v1.0.pkl
│   ├── experiments/                # Experimental models
│   └── model_registry.json         # Model metadata
│
├── 📁 data/                        # Data directory
│   ├── raw/                        # Raw data files
│   ├── processed/                  # Processed data
│   ├── fixtures/                   # Test data
│   └── schemas/                    # Data schemas
│
├── 📁 docs/                        # Documentation
│   ├── api/                        # API documentation
│   ├── deployment/                 # Deployment guides
│   ├── development/                # Development setup
│   └── user-guide/                 # User documentation
│
├── 📁 scripts/                     # Utility scripts
│   ├── setup.sh                    # Initial setup
│   ├── train_models.py             # Model training
│   ├── data_ingestion.py           # Data collection
│   ├── backup.sh                   # Database backup
│   └── deploy.sh                   # Deployment script
│
├── 📁 infrastructure/              # Infrastructure as Code
│   ├── docker/                     # Docker configurations
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.web
│   │   └── Dockerfile.worker
│   ├── kubernetes/                 # K8s manifests
│   ├── terraform/                  # AWS infrastructure
│   └── monitoring/                 # Monitoring configs
│       ├── grafana/
│       └── prometheus/
│
└── 📁 notebooks/                   # Jupyter notebooks
    ├── exploration/                # Data exploration
    ├── experiments/                # Model experiments
    └── analysis/                   # Post-hoc analysis
```

---

## 🚀 Step-by-Step Repository Setup

### **Step 1: Create New Repository**

```bash
# Create new directory
mkdir fantasy-football-ai
cd fantasy-football-ai

# Initialize git
git init
git branch -M main

# Create GitHub repository (using GitHub CLI)
gh repo create fantasy-football-ai --public --description "🏈 AI-powered Fantasy Football Assistant with ML predictions, real-time data, and modern web interface"

# Set remote
git remote add origin https://github.com/[your-username]/fantasy-football-ai.git
```

### **Step 2: Core Configuration Files**

Create these essential files first:

#### **pyproject.toml** (Modern Python packaging)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fantasy-football-ai"
version = "1.0.0"
description = "AI-powered Fantasy Football Assistant"
authors = [{name = "Christopher Bratkovics", email = "cbratkovics@gmail.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
keywords = ["fantasy-football", "ai", "machine-learning", "nfl", "predictions"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "streamlit>=1.28.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "tensorflow>=2.14.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "redis>=5.0.0",
    "aiohttp>=3.8.0",
    "pydantic>=2.5.0",
    "plotly>=5.17.0",
    "psycopg2-binary>=2.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.10.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "jupyter>=1.0.0",
]

[project.urls]
"Homepage" = "https://github.com/[your-username]/fantasy-football-ai"
"Bug Reports" = "https://github.com/[your-username]/fantasy-football-ai/issues"
"Source" = "https://github.com/[your-username]/fantasy-football-ai"

[project.scripts]
fantasy-ai = "fantasy_ai.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
```

#### **Makefile** (Development shortcuts)
```makefile
.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down

help:  ## Show this help message
	@echo "Fantasy Football AI - Development Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run test suite
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 src tests
	mypy src

format:  ## Format code
	black src tests
	isort src tests

clean:  ## Clean up temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker containers
	docker-compose build

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

migrate:  ## Run database migrations
	alembic upgrade head

migrate-create:  ## Create new migration
	alembic revision --autogenerate -m "$(name)"

setup-dev:  ## Complete development setup
	make install-dev
	make docker-up
	sleep 10
	make migrate
	@echo "✅ Development environment ready!"

deploy-staging:  ## Deploy to staging
	@echo "🚀 Deploying to staging..."
	# Add staging deployment commands

deploy-prod:  ## Deploy to production
	@echo "🚀 Deploying to production..."
	# Add production deployment commands
```

### **Step 3: Git Configuration**

#### **.gitignore**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.nox/

# ML Models & Data
models/experiments/
models/trained/*.h5
models/trained/*.pkl
data/raw/
data/processed/
*.csv
*.parquet

# Logs
logs/
*.log

# Database
*.db
*.sqlite

# Docker
.docker/

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local
.env.production

# Streamlit
.streamlit/secrets.toml

# Monitoring
grafana/data/
prometheus/data/
```