#!/bin/bash

echo "Fixing Fantasy Football AI Setup Issues"
echo "======================================"

# 1. Stop everything
echo "Stopping all containers..."
docker-compose down

# 2. Clean up volumes if needed (optional - uncomment if you want fresh start)
# docker volume rm fantasy-football-ai_postgres_data fantasy-football-ai_redis_data

# 3. Update frontend app.py to use environment variables
echo "Updating frontend to use environment variables..."
sed -i.bak '136s/API_BASE_URL = st.secrets.get("API_BASE_URL", "http:\/\/localhost:8000")/API_BASE_URL = os.getenv("API_BASE_URL", "http:\/\/localhost:8000")/' frontend/app.py

# 4. Update secrets.toml
echo "Updating secrets.toml with correct database credentials..."
cat > frontend/.streamlit/secrets.toml << 'EOF'
# Streamlit secrets configuration
API_BASE_URL = "http://backend:8000"
DATABASE_URL = "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football"
ENVIRONMENT = "development"
EOF

# 5. Start services
echo "Starting services..."
docker-compose up -d

# 6. Wait for services
echo "Waiting for services to be healthy..."
sleep 15

# 7. Check service status
docker-compose ps

# 8. Update database connection in scripts
echo "Updating database connection in init script..."
sed -i.bak 's/postgres:postgres/fantasy_user:fantasy_pass/g' scripts/init_database.py
sed -i.bak 's/postgres:postgres/fantasy_user:fantasy_pass/g' scripts/fetch_sleeper_data.py

# 9. Initialize database with correct user
echo "Initializing database..."
docker-compose exec backend python scripts/init_database.py

if [ $? -eq 0 ]; then
    echo "Database initialized successfully!"
else
    echo "Database initialization failed. Checking connection..."
    docker-compose exec postgres psql -U fantasy_user -d fantasy_football -c "SELECT 1;"
fi

# 10. Import data
echo "Fetching player data (this may take a few minutes)..."
docker-compose exec backend python scripts/fetch_sleeper_data.py

# 11. Restart backend to ensure all changes are loaded
echo "Restarting backend..."
docker-compose restart backend

# 12. Final check
echo ""
echo "Checking if everything is working..."
sleep 5

# Check API
echo "Testing API..."
curl -s "http://localhost:8000/health" | python -m json.tool

# Check for real data
echo ""
echo "Checking for real player data..."
curl -s "http://localhost:8000/players/rankings?position=QB&limit=3" | python -m json.tool

echo ""
echo "Setup fix complete!"
echo "Access the application at http://localhost:8501"