#!/usr/bin/env python3
"""Verify that the Sleeper API client has the methods we need"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Checking Sleeper API client methods...")
print("=" * 50)

try:
    from backend.data.sleeper_client import SleeperAPIClient
    
    client = SleeperAPIClient()
    
    print("✓ SleeperAPIClient imported successfully")
    
    # Check for required methods
    methods_to_check = [
        'get_all_players',
        'get_week_stats',
        'get_trending_players'
    ]
    
    for method_name in methods_to_check:
        if hasattr(client, method_name):
            print(f"✓ Method '{method_name}' exists")
        else:
            print(f"✗ Method '{method_name}' NOT FOUND")
            
            # Look for similar methods
            similar = [attr for attr in dir(client) if 'get' in attr and not attr.startswith('_')]
            if similar:
                print(f"  Available 'get' methods: {', '.join(similar)}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 50)
print("If get_week_stats is missing, we'll need to implement it or find the correct method name.")