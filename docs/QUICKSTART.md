# Fantasy Football AI - Quick Start Guide

## Overview

This guide will help you get the Fantasy Football AI system running with real NFL player data from the Sleeper API.

## Prerequisites

- Docker and Docker Compose installed
- Make command available
- Git installed
- 8GB RAM minimum (for ML models)
- Ports 8000, 8501, 5432, and 6379 available

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:cbratkovics/fantasy-football-ai.git
cd fantasy-football-ai
```

### 2. Quick Setup Using Existing Makefile

Your project includes a Makefile for easy setup:

```bash
make setup-dev
make docker-up
```

### 3. Initialize Database and Fetch Real Data

If the Makefile doesn't include data initialization, run these additional steps:

```bash
# Create required files
mkdir -p frontend/.streamlit
mkdir -p scripts

# Copy the initialization scripts from the artifacts above to:
# - scripts/init_database.py
# - scripts/fetch_sleeper_data.py
# - api/players.py (update existing file)
# - frontend/app.py (update existing file)

# Run the complete setup
chmod +x scripts/setup_production.sh
./scripts/setup_production.sh
```

### 4. Manual Setup (if needed)

If you prefer manual setup or need to troubleshoot:

```bash
# Start services
docker-compose up -d

# Wait for services to start
sleep 10

# Initialize database
docker-compose exec backend python scripts/init_database.py

# Fetch NFL player data (takes 3-5 minutes)
docker-compose exec backend python scripts/fetch_sleeper_data.py

# Restart services to ensure all connections are fresh
docker-compose restart
```

## Accessing the System

Once setup is complete:

- **Frontend Application**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Verifying the Installation

Check if everything is working:

```bash
# Check service status
docker-compose ps

# Verify API health
curl http://localhost:8000/health

# Check player data
curl "http://localhost:8000/players/rankings?limit=5"

# Run verification script
./scripts/verify_setup.sh
```

## Features Available

After setup, you'll have:

1. **Real NFL Player Data**: All active players from the 2024 season
2. **Calculated Fantasy Points**: Standard, PPR, and Half-PPR scoring
3. **Player Rankings**: Tier-based rankings with consistency metrics
4. **Weekly Projections**: Simple projections based on recent performance
5. **API Access**: Full REST API for all player data

## Common Issues and Solutions

### Frontend Connection Error
```bash
# Ensure backend is running
docker-compose logs backend

# Check if API is accessible
curl http://localhost:8000/health
```

### No Player Data Showing
```bash
# Re-run data fetch
docker-compose exec backend python scripts/fetch_sleeper_data.py

# Check database
docker-compose exec postgres psql -U postgres -d fantasy_football -c "SELECT COUNT(*) FROM players;"
```

### Port Already in Use
```bash
# Stop conflicting services or change ports in docker-compose.yml
docker-compose down
# Edit docker-compose.yml to use different ports
docker-compose up -d
```

## Updating Data

To refresh player data (e.g., for a new week):

```bash
docker-compose exec backend python scripts/fetch_sleeper_data.py
```

## Development Commands

Your Makefile includes these useful commands:

```bash
make test          # Run tests
make format        # Format code with Black
make lint          # Run linting
make install-dev   # Install development dependencies
```

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  FastAPI    │────▶│ PostgreSQL  │
│  Frontend   │     │  Backend    │     │  Database   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                     ▲
                           ▼                     │
                    ┌─────────────┐     ┌────────┴────┐
                    │    Redis    │     │ Sleeper API │
                    │    Cache    │     │   Client    │
                    └─────────────┘     └─────────────┘
```

## Next Steps

1. **Explore the UI**: Navigate through Player Rankings, Draft Assistant, and Projections
2. **Check API Docs**: Visit http://localhost:8000/docs for interactive API documentation
3. **Integrate ML Models**: The ML pipeline in `ml/` is ready for integration
4. **Customize Scoring**: Modify `data/scoring_engine.py` for custom league settings
5. **Add Authentication**: Implement user authentication for team management

## Support

For issues or questions:
1. Check the logs: `docker-compose logs [service_name]`
2. Review the documentation in the repository
3. Ensure all prerequisites are met
4. Verify database connectivity and data loading