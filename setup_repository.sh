#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ️${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠️${NC} $1"; }

safe_mkdir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_status "Created directory: $1"
    else
        print_info "Directory already exists: $1"
    fi
}

echo "🏈 Creating Fantasy Football AI Repository Structure..."

# Create directories
safe_mkdir "src/fantasy_ai/core/models"
safe_mkdir "src/fantasy_ai/core/data/sources"
safe_mkdir "src/fantasy_ai/core/data/storage"
safe_mkdir "src/fantasy_ai/core/services"
safe_mkdir "src/fantasy_ai/api"
safe_mkdir "src/fantasy_ai/web/components"
safe_mkdir "tests/unit"
safe_mkdir "tests/integration"
safe_mkdir "models/trained"
safe_mkdir "data/raw"
safe_mkdir "docs"
safe_mkdir "infrastructure/docker"
safe_mkdir "notebooks"
safe_mkdir ".github/workflows"

# Create __init__.py files
touch src/fantasy_ai/__init__.py
touch src/fantasy_ai/core/__init__.py
touch src/fantasy_ai/core/models/__init__.py
touch src/fantasy_ai/core/data/__init__.py
touch src/fantasy_ai/api/__init__.py
touch src/fantasy_ai/web/__init__.py
touch tests/__init__.py

echo "✅ Repository structure created successfully!"
