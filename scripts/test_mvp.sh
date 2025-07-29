#!/bin/bash

# Fantasy Football AI - MVP Testing Script
# Run this to verify all components are working before committing

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Testing: $test_name ... "
    
    if eval $test_command > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

print_header "Fantasy Football AI - MVP Test Suite"
echo "Starting comprehensive system tests..."
echo "Time: $(date)"

# 1. Docker Services Tests
print_header "1. Docker Services"

run_test "Docker Compose services running" "docker-compose ps | grep -E 'Up|running' > /dev/null"
run_test "PostgreSQL container healthy" "docker-compose ps postgres | grep -E 'healthy|running'"
run_test "Redis container healthy" "docker-compose ps redis | grep -E 'healthy|running'"
run_test "Backend container running" "docker-compose ps backend | grep -E 'Up|running'"
run_test "Frontend container running" "docker-compose ps frontend | grep -E 'Up|running'"

# 2. Database Tests
print_header "2. Database Verification"

echo "Checking database connection and data..."

# Test database connection
run_test "Database connection" "docker-compose exec -T postgres psql -U postgres -d fantasy_football -c 'SELECT 1' > /dev/null 2>&1"

# Check tables exist
TABLES=$(docker-compose exec -T postgres psql -U postgres -d fantasy_football -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>/dev/null | tr -d ' ')
run_test "Database tables created (found $TABLES tables)" "[ $TABLES -gt 0 ]"

# Check player data
PLAYER_COUNT=$(docker-compose exec -T postgres psql -U postgres -d fantasy_football -t -c "SELECT COUNT(*) FROM players" 2>/dev/null | tr -d ' ' || echo "0")
run_test "Players table has data (found $PLAYER_COUNT players)" "[ $PLAYER_COUNT -gt 100 ]"

# Check weekly stats
STATS_COUNT=$(docker-compose exec -T postgres psql -U postgres -d fantasy_football -t -c "SELECT COUNT(*) FROM weekly_stats" 2>/dev/null | tr -d ' ' || echo "0")
run_test "Weekly stats table has data (found $STATS_COUNT entries)" "[ $STATS_COUNT -gt 1000 ]"

# Sample data verification
echo -e "\nSample Players in Database:"
docker-compose exec -T postgres psql -U postgres -d fantasy_football -c "
SELECT full_name, position, team, age 
FROM players 
WHERE position IN ('QB', 'RB', 'WR') 
ORDER BY full_name 
LIMIT 5
" 2>/dev/null || echo "Could not query players"

# 3. API Tests
print_header "3. Backend API Tests"

# Health check
run_test "API health endpoint" "curl -s -f http://localhost:8000/health"

# Test player rankings endpoint
run_test "Player rankings endpoint" "curl -s -f http://localhost:8000/players/rankings?limit=5"

# Test with parameters
run_test "Rankings with position filter" "curl -s -f 'http://localhost:8000/players/rankings?position=QB&limit=5'"
run_test "Rankings with PPR scoring" "curl -s -f 'http://localhost:8000/players/rankings?scoring_type=ppr&limit=5'"

# Check if data is real (not mock)
RESPONSE=$(curl -s http://localhost:8000/players/rankings?limit=10 2>/dev/null || echo "{}")
if echo "$RESPONSE" | grep -q "sleeper_id"; then
    echo -e "${GREEN}API returns real player data with sleeper_id${NC}"
else
    echo -e "${YELLOW}API may be returning mock data${NC}"
fi

# Sample API response
echo -e "\nSample API Response (top 3 QBs):"
curl -s "http://localhost:8000/players/rankings?position=QB&limit=3" 2>/dev/null | python -m json.tool 2>/dev/null | head -30 || echo "Could not fetch API data"

# 4. Data Quality Tests
print_header "4. Data Quality Verification"

# Check for variety in positions
echo "Checking position distribution..."
docker-compose exec -T postgres psql -U postgres -d fantasy_football -t -c "
SELECT position, COUNT(*) as count 
FROM players 
WHERE status = 'Active' 
GROUP BY position 
ORDER BY count DESC
" 2>/dev/null

# Check scoring calculations
echo -e "\nTop 5 players by average PPR points:"
docker-compose exec -T postgres psql -U postgres -d fantasy_football -c "
SELECT p.full_name, p.position, p.team, 
       ROUND(AVG(ws.fantasy_points_ppr), 1) as avg_ppr_points,
       COUNT(ws.id) as games_played
FROM players p
JOIN weekly_stats ws ON p.id = ws.player_id
WHERE ws.season = 2024
GROUP BY p.id, p.full_name, p.position, p.team
HAVING COUNT(ws.id) >= 10
ORDER BY avg_ppr_points DESC
LIMIT 5
" 2>/dev/null || echo "Could not calculate scoring averages"

# 5. Frontend Tests
print_header "5. Frontend Accessibility"

run_test "Frontend HTTP response" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501 | grep -E '200|304'"
run_test "Frontend has content" "curl -s http://localhost:8501 | grep -q 'Fantasy Football'"

# 6. Integration Tests
print_header "6. Integration Tests"

# Test full data flow
echo "Testing complete data flow..."

# Get a player from API
PLAYER_DATA=$(curl -s http://localhost:8000/players/rankings?limit=1 2>/dev/null)
if [ ! -z "$PLAYER_DATA" ] && echo "$PLAYER_DATA" | grep -q "player_id"; then
    PLAYER_ID=$(echo "$PLAYER_DATA" | grep -o '"player_id":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ ! -z "$PLAYER_ID" ]; then
        run_test "Fetch individual player details" "curl -s -f http://localhost:8000/players/$PLAYER_ID"
    fi
fi

# Test projections endpoint
WEEK=1
run_test "Weekly projections endpoint" "curl -s -f http://localhost:8000/players/projections/week/$WEEK?limit=10"

# 7. Performance Tests
print_header "7. Performance Checks"

echo "Testing API response times..."
START=$(date +%s.%N)
curl -s http://localhost:8000/players/rankings?limit=100 > /dev/null 2>&1
END=$(date +%s.%N)
DURATION=$(echo "$END - $START" | bc)
echo "Rankings endpoint (100 players): ${DURATION}s"

if (( $(echo "$DURATION < 2.0" | bc -l) )); then
    echo -e "${GREEN}Performance is good (<2s)${NC}"
else
    echo -e "${YELLOW}Performance could be improved (>2s)${NC}"
fi

# 8. Error Handling Tests
print_header "8. Error Handling"

run_test "API handles invalid position" "curl -s http://localhost:8000/players/rankings?position=INVALID | grep -E 'error|\\[\\]'"
run_test "API handles invalid player ID" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/players/invalid-id-12345 | grep '404'"

# Summary
print_header "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! Your MVP is ready for git push.${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please fix issues before pushing to git.${NC}"
    exit 1
fi