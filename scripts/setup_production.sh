#!/bin/bash

# Fantasy Football AI - Production Setup Script
# Sets up the complete system with real NFL data from Sleeper API

set -e  # Exit on error

# Configuration
PROJECT_NAME="Fantasy Football AI"
PYTHON_VERSION="3.11"

# Color codes for output (no emojis)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "[INFO] $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Main setup process
print_header "$PROJECT_NAME - Production Setup"

# Step 1: Check prerequisites
print_header "Step 1: Checking Prerequisites"

MISSING_DEPS=0

if ! check_command docker; then
    MISSING_DEPS=1
fi

if ! check_command docker-compose; then
    MISSING_DEPS=1
fi

if ! check_command make; then
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    print_error "Missing required dependencies. Please install Docker, Docker Compose, and Make."
    exit 1
fi

print_success "All prerequisites installed"

# Step 2: Create necessary directories and files
print_header "Step 2: Setting Up Directory Structure"

# Create directories if they don't exist
mkdir -p frontend/.streamlit
mkdir -p scripts
mkdir -p logs

print_success "Directory structure created"

# Step 3: Create Streamlit secrets
print_header "Step 3: Configuring Streamlit"

if [ ! -f frontend/.streamlit/secrets.toml ]; then
    cat > frontend/.streamlit/secrets.toml << 'EOF'
# Streamlit secrets configuration
API_BASE_URL = "http://backend:8000"
DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/fantasy_football"
ENVIRONMENT = "development"
EOF
    print_success "Created frontend/.streamlit/secrets.toml"
else
    print_info "Streamlit secrets already exist"
fi

# Step 4: Start Docker containers
print_header "Step 4: Starting Docker Services"

if [ -f Makefile ]; then
    print_info "Using Makefile to start services..."
    make docker-up
else
    print_info "Using docker-compose directly..."
    docker-compose up -d
fi

# Wait for services to be ready
print_info "Waiting for services to start..."
sleep 10

# Check if services are running
docker-compose ps

# Step 5: Initialize database
print_header "Step 5: Initializing Database"

print_info "Creating database schema..."
docker-compose exec -T backend python scripts/init_database.py

if [ $? -eq 0 ]; then
    print_success "Database initialized successfully"
else
    print_error "Database initialization failed"
    exit 1
fi

# Step 6: Fetch real NFL data
print_header "Step 6: Fetching NFL Player Data"

print_info "Fetching player data from Sleeper API..."
print_info "This may take several minutes for the first run..."

docker-compose exec -T backend python scripts/fetch_sleeper_data.py

if [ $? -eq 0 ]; then
    print_success "Player data fetched successfully"
else
    print_error "Failed to fetch player data"
    exit 1
fi

# Step 7: Run health checks
print_header "Step 7: Running Health Checks"

# Check backend API
print_info "Checking backend API..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend API is healthy"
    
    # Show health status
    echo -e "\nHealth Status:"
    curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "Could not parse health response"
else
    print_error "Backend API is not responding"
fi

# Check frontend
print_info "Checking frontend..."
if curl -s -f http://localhost:8501 > /dev/null 2>&1; then
    print_success "Frontend is accessible"
else
    print_error "Frontend is not responding"
fi

# Check sample data
print_info "Checking player data..."
PLAYER_COUNT=$(curl -s http://localhost:8000/players/rankings?limit=1 2>/dev/null | grep -c "player_id" || echo "0")

if [ "$PLAYER_COUNT" -gt "0" ]; then
    print_success "Player data is available in the API"
    
    # Show sample player
    echo -e "\nSample Player Data:"
    curl -s "http://localhost:8000/players/rankings?limit=3" | python -m json.tool 2>/dev/null | head -50
else
    print_warning "No player data found in API"
fi

# Step 8: Final summary
print_header "Setup Complete!"

echo -e "${GREEN}The Fantasy Football AI system is now running!${NC}\n"

echo "Access Points:"
echo "  - Frontend Application: http://localhost:8501"
echo "  - Backend API: http://localhost:8000"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - PostgreSQL Database: localhost:5432"
echo "  - Redis Cache: localhost:6379"

echo -e "\nUseful Commands:"
echo "  - View logs: docker-compose logs -f [service_name]"
echo "  - Stop services: docker-compose stop"
echo "  - Restart services: docker-compose restart"
echo "  - Update data: docker-compose exec backend python scripts/fetch_sleeper_data.py"

echo -e "\nNext Steps:"
echo "  1. Visit http://localhost:8501 to access the application"
echo "  2. Check http://localhost:8000/docs for API documentation"
echo "  3. The system now contains real NFL player data from the 2024 season"
echo "  4. Player rankings are calculated based on actual performance metrics"

# Create a verification script for later use
cat > scripts/verify_setup.sh << 'EOF'
#!/bin/bash

echo "Verifying Fantasy Football AI Setup..."
echo "======================================"

# Check containers
echo -e "\n1. Docker Containers:"
docker-compose ps

# Check database
echo -e "\n2. Database Status:"
docker-compose exec -T postgres psql -U postgres -d fantasy_football -c "SELECT COUNT(*) as player_count FROM players;" 2>/dev/null || echo "Database not accessible"

# Check API
echo -e "\n3. API Status:"
curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "API not responding"

# Check player data
echo -e "\n4. Sample Players:"
curl -s "http://localhost:8000/players/rankings?limit=5&position=QB" | python -m json.tool 2>/dev/null || echo "No player data"

echo -e "\nVerification complete!"
EOF

chmod +x scripts/verify_setup.sh

print_info "Created verification script at scripts/verify_setup.sh"
print_info "Run './scripts/verify_setup.sh' anytime to check system status"

# End of setup
echo -e "\n${GREEN}Setup completed at $(date)${NC}"