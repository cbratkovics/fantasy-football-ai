#!/bin/bash

# Fantasy Football AI - Complete Setup with Correct Schema
echo "Fantasy Football AI - Complete Setup"
echo "===================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Initialize database with correct schema
echo -e "\n${YELLOW}Step 1: Initializing database...${NC}"
docker-compose exec backend python scripts/init_database.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Database initialization failed${NC}"
    exit 1
fi

# Step 2: Import players
echo -e "\n${YELLOW}Step 2: Importing players from Sleeper API...${NC}"
docker-compose exec backend python scripts/fetch_sleeper_data.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Player import failed${NC}"
    exit 1
fi

# Step 3: Verify import
echo -e "\n${YELLOW}Step 3: Verifying data import...${NC}"

# Check total players
echo "Player counts by position:"
docker-compose exec postgres psql -U fantasy_user -d fantasy_football -c "
SELECT position, COUNT(*) as count 
FROM players 
WHERE status IN ('Active', 'Injured Reserve')
GROUP BY position 
ORDER BY count DESC;"

# Show some real QBs
echo -e "\n${GREEN}Sample QBs in database:${NC}"
docker-compose exec postgres psql -U fantasy_user -d fantasy_football -c "
SELECT first_name || ' ' || last_name as name, team, age, status
FROM players 
WHERE position = 'QB' 
AND status = 'Active'
ORDER BY last_name
LIMIT 10;"

# Show total counts
echo -e "\n${GREEN}Summary:${NC}"
docker-compose exec postgres psql -U fantasy_user -d fantasy_football -t -c "
SELECT 
  'Total Active Players: ' || COUNT(*) 
FROM players 
WHERE status = 'Active';"

# Step 4: Test API
echo -e "\n${YELLOW}Step 4: Testing API...${NC}"
echo "Note: API still returns mock data until we update the endpoints"
curl -s "http://localhost:8000/players/rankings?position=QB&limit=3" | python -m json.tool | head -20

echo -e "\n${GREEN}Setup complete!${NC}"
echo "Players are now in the database. Next steps:"
echo "1. Update api/players.py to query the database instead of returning mock data"
echo "2. Implement stats fetching (get_week_stats method or similar)"
echo "3. Update the PlayerStats table with actual game data"