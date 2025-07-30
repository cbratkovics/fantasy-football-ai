# Scripts Directory

This directory contains essential scripts for managing the Fantasy Football AI system.

## Core Scripts

### Setup & Initialization
- `init_database.py` - Initialize database schema and tables
- `fetch_sleeper_data.py` - Fetch player data from Sleeper API
- `setup_production.sh` - Complete production setup script

### Testing & Verification
- `test_mvp.sh` - Comprehensive system test suite
- `test_data_quality.py` - Validate data integrity
- `verify_api_methods.py` - Check API client methods

### Deployment
- `deploy.sh` - Deploy to production environment

### Data Management
- `run_data_setup.sh` - Initialize database and fetch player data

## Usage

Most scripts should be run inside Docker containers:

```bash
# Initialize database
docker-compose exec backend python scripts/init_database.py

# Fetch player data
docker-compose exec backend python scripts/fetch_sleeper_data.py

# Run tests
./scripts/test_mvp.sh
```