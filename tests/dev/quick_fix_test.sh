#!/bin/bash

echo "🔧 Quick Fix Test for Fantasy Football AI"
echo "========================================="

# Test 1: Check if CLI loads without errors
echo "1. Testing CLI loading..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from fantasy_ai.cli.main import cli
    print('✅ CLI imports successfully')
except Exception as e:
    print(f'❌ CLI import failed: {e}')
    exit(1)
"

# Test 2: Try version command (should be simple)
echo ""
echo "2. Testing version command..."
python -m fantasy_ai.cli.main version

# Test 3: Try help command
echo ""
echo "3. Testing help command..."
python -m fantasy_ai.cli.main --help

echo ""
echo "✅ Quick fix test completed!"