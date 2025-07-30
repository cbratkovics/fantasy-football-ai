# Fantasy Football AI - Clean Project Structure

## Architecture Overview

This project follows a microservices architecture with clear separation of concerns:

```
fantasy-football-ai/
├── backend/                    # FastAPI Backend Service
│   ├── api/                   # API endpoints
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── players.py        # Player data endpoints
│   │   ├── predictions.py    # ML predictions endpoints
│   │   └── subscriptions.py  # Subscription management
│   ├── data/                  # Data pipeline layer
│   │   ├── data_pipeline.py  # Main data processing
│   │   ├── fetch_players.py  # Player data fetching
│   │   ├── scoring.py        # Fantasy scoring logic
│   │   └── sleeper_client.py # Sleeper API client
│   ├── ml/                    # Machine Learning models
│   │   ├── features.py       # Feature engineering
│   │   ├── gmm_clustering.py # GMM draft tiers
│   │   ├── neural_network.py # Neural network predictions
│   │   └── train.py          # Model training
│   ├── models/                # Database models
│   │   ├── database.py       # SQLAlchemy models
│   │   └── schemas.py        # Pydantic schemas
│   ├── tasks/                 # Celery async tasks
│   │   ├── train_models.py   # Async model training
│   │   └── update_data.py    # Async data updates
│   ├── main.py               # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Backend container
│
├── frontend/                   # Streamlit Frontend Service
│   ├── app.py                # Main Streamlit app
│   ├── components/           # Reusable UI components
│   │   ├── auth.py          # Authentication UI
│   │   └── charts.py        # Data visualization
│   ├── pages/                # Application pages
│   │   ├── account.py       # User account page
│   │   ├── draft_assistant.py # Draft helper
│   │   ├── rankings.py      # Player rankings
│   │   └── weekly_picks.py  # Weekly predictions
│   ├── requirements.txt     # Frontend dependencies
│   └── Dockerfile           # Frontend container
│
├── infrastructure/            # Infrastructure as Code
│   ├── terraform/           # Terraform AWS setup
│   │   └── main.tf         # AWS infrastructure
│   ├── nginx.conf          # Reverse proxy config
│   └── docker-compose.yml  # Local orchestration
│
├── scripts/                  # Utility scripts
│   ├── init_database.py    # Database initialization
│   ├── fetch_sleeper_data.py # Data fetching
│   ├── test_mvp.sh         # System tests
│   └── deploy.sh           # Deployment script
│
├── ssl/                     # SSL certificates (production)
├── docker-compose.yml       # Docker orchestration
└── README.md               # Project documentation
```

## Key Design Principles

### 1. Microservices Architecture
- **Backend**: FastAPI service handling all business logic and data
- **Frontend**: Streamlit service consuming the API
- **Database**: PostgreSQL with Redis caching
- **Clear separation**: Frontend never directly accesses the database

### 2. Asynchronous Processing
- **Celery**: Background tasks for model training and data updates
- **Redis**: Message broker for Celery and caching layer
- **Async API**: FastAPI with async/await for high performance

### 3. Infrastructure as Code
- **Terraform**: Reproducible AWS deployments
- **Docker Compose**: Local development environment
- **Environment configs**: Separate dev/prod configurations

### 4. Containerization
- **Docker**: Consistent environments across dev/staging/prod
- **Multi-stage builds**: Optimized container sizes
- **Health checks**: Automatic container monitoring

### 5. API-First Design
- **RESTful API**: Clean, predictable endpoints
- **OpenAPI/Swagger**: Auto-generated API documentation
- **Type safety**: Pydantic models for validation
- **Frontend consumes API**: No direct database access

## Service Communication

```
User → Nginx → Frontend (Streamlit:8501)
                    ↓
                Backend API (FastAPI:8000)
                    ↓
            PostgreSQL / Redis
                    ↓
            Celery Workers (Async Tasks)
```

## Data Flow

1. **Data Ingestion**: Sleeper API → Backend → PostgreSQL
2. **ML Pipeline**: PostgreSQL → Feature Engineering → Model Training → Predictions
3. **User Requests**: Frontend → API → Database → Response
4. **Async Tasks**: API → Celery → Background Processing → Database Update

This clean architecture ensures scalability, maintainability, and clear separation of concerns.