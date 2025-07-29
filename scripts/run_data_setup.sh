#!/bin/bash

# Fantasy Football AI - Data Setup Script
# This script initializes the database and fetches player data

echo "Fantasy Football AI - Data Setup"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Test database connection
echo -e "\n${YELLOW}Step 1: Testing database connection...${NC}"
docker-compose exec backend python scripts/test_db_connection.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Database connection failed. Please check your setup.${NC}"
    exit 1
fi

# Step 2: Verify API methods exist
echo -e "\n${YELLOW}Step 2: Verifying Sleeper API client...${NC}"
docker-compose exec backend python scripts/verify_api_methods.py

# Step 3: Initialize database schema
echo -e "\n${YELLOW}Step 3: Initializing database schema...${NC}"
docker-compose exec backend python scripts/init_database.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Database initialization failed.${NC}"
    exit 1
fi

# Step 4: Fetch player data from Sleeper API
echo -e "\n${YELLOW}Step 4: Fetching player data from Sleeper API...${NC}"
echo "This may take 5-10 minutes for the first run..."
docker-compose exec backend python scripts/fetch_sleeper_data.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Data fetch failed.${NC}"
    echo "Check if get_week_stats method exists in SleeperAPIClient"
    exit 1
fi

# Step 5: Verify data was imported
echo -e "\n${YELLOW}Step 5: Verifying data import...${NC}"

# Check player count
PLAYER_COUNT=$(docker-compose exec -T postgres psql -U fantasy_user -d fantasy_football -t -c "SELECT COUNT(*) FROM players;" 2>/dev/null | tr -d ' ')
echo "Players in database: $PLAYER_COUNT"

# Check stats count
STATS_COUNT=$(docker-compose exec -T postgres psql -U fantasy_user -d fantasy_football -t -c "SELECT COUNT(*) FROM player_stats;" 2>/dev/null | tr -d ' ')
echo "Player stats entries: $STATS_COUNT"

# Show sample players
echo -e "\nSample players:"
docker-compose exec -T postgres psql -U fantasy_user -d fantasy_football -c "
SELECT full_name, position, team 
FROM players 
WHERE position IN ('QB', 'RB', 'WR') 
ORDER BY full_name 
LIMIT 5;"

# Step 6: Test API with real data
echo -e "\n${YELLOW}Step 6: Testing API endpoints...${NC}"
echo "Fetching top QBs..."
curl -s "http://localhost:8000/players/rankings?position=QB&limit=3" | python -m json.tool

echo -e "\n${GREEN}Data setup complete!${NC}"
echo "You can now access:"
echo "  - Frontend: http://localhost:8501"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"