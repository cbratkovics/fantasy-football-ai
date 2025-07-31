# Fantasy Football AI Backend

## Overview
This is the backend service for the Fantasy Football AI application, providing ML-powered predictions, player analysis, and real-time updates for fantasy football enthusiasts.

## Key Features
- **ML Predictions**: Production-ready models trained on 31,000+ NFL records (2019-2024)
- **Real-time Updates**: WebSocket support for live game updates
- **Data Sources**: Integration with nfl_data_py, Open-Meteo weather, and ESPN APIs
- **Rate Limiting**: Redis-based rate limiting and caching
- **Authentication**: JWT-based auth with role-based access control
- **Payment Processing**: Stripe integration for subscriptions

## Quick Start

### Prerequisites
- Python 3.10
- PostgreSQL
- Redis
- Docker (for deployment)

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the server
uvicorn main:app --reload --port 8000
```

### Docker Development
```bash
docker build -t fantasy-football-ai .
docker run -p 8000:8000 --env-file .env fantasy-football-ai
```

## Project Structure
```
backend/
├── api/              # API endpoints
├── core/             # Core utilities (cache, rate limiting, websocket)
├── data/             # Data collection and processing
│   └── sources/      # External data source clients
├── ml/               # Machine learning models and features
├── models/           # Database models and schemas
│   └── production/   # Production ML models
├── services/         # Business logic services
├── tasks/            # Celery background tasks
└── alembic/          # Database migrations
```

## ML Models

### Performance Metrics
Our production models achieve the following Mean Absolute Error (MAE) scores:
- **QB**: 6.17 points (18.1% improvement over baseline)
- **RB**: 4.92 points (24.1% improvement over baseline)
- **WR**: 4.99 points (20.3% improvement over baseline)
- **TE**: 3.88 points (17.5% improvement over baseline)

### Features
- Lagged statistics (L1W, L3W, L5W)
- Season-to-date averages
- Position-specific efficiency metrics
- Temporal features (week of season, games played)
- No data leakage - only pre-game available features

## API Documentation
Once running, visit:
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints
- `/health` - Health check
- `/api/v1/auth/*` - Authentication endpoints
- `/api/v1/players/*` - Player data and stats
- `/api/v1/predictions/*` - ML predictions
- `/api/v1/llm/*` - AI-powered analysis
- `/ws` - WebSocket for real-time updates

## Data Sources

### Free Commercial-Use Sources
1. **nfl_data_py** (MIT License)
   - Play-by-play data
   - Player stats
   - Team schedules
   
2. **Open-Meteo Weather API**
   - Stadium weather conditions
   - Free tier, commercial use allowed
   
3. **ESPN API** (with authentication)
   - Fantasy projections
   - Player news

## Deployment

### Railway Deployment
The project is configured for Railway deployment with:
- Dockerfile optimized for production
- Health checks configured
- Auto-restart on failure

### Environment Variables
Required for production:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - JWT secret key
- `OPENAI_API_KEY` - For AI features (optional)
- `STRIPE_API_KEY` - For payments (optional)

See `.env.example` for complete list.

## Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=backend tests/
```

## Contributing
1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running and DATABASE_URL is correct
2. **Redis Connection**: Verify Redis is running on the configured port
3. **Missing Dependencies**: Run `pip install -r requirements.txt` again
4. **Port Already in Use**: Change port with `--port` flag

### Railway Deployment Issues
- Check logs: `railway logs`
- Verify environment variables are set
- Ensure Dockerfile builds locally first

## License
[Add your license here]

## Support
[Add support contact information]