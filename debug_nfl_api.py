"""
Debug script to check NFL API response structure.
Save this as: debug_nfl_api.py and run it to see the actual API response.
"""

import asyncio
import json
from src.fantasy_ai.core.data.sources.nfl_comprehensive import create_nfl_client

async def debug_nfl_api():
    """Debug the NFL API response structure."""
    
    print("=== NFL API Debug ===")
    
    # Create NFL client
    client = await create_nfl_client()
    
    # Get teams data (this is what's failing)
    print("\n1. Testing get_teams()...")
    try:
        teams_data = await client.get_teams()
        print(f"Teams response type: {type(teams_data)}")
        print(f"Teams response length: {len(teams_data) if isinstance(teams_data, (list, dict)) else 'N/A'}")
        
        # Show first few items
        if isinstance(teams_data, list) and len(teams_data) > 0:
            print(f"\nFirst team structure:")
            print(json.dumps(teams_data[0], indent=2, default=str))
            
            print(f"\nAll keys in first team:")
            if isinstance(teams_data[0], dict):
                for key, value in teams_data[0].items():
                    print(f"  {key}: {type(value)} = {value}")
                    
        elif isinstance(teams_data, dict):
            print(f"\nTeams data is a dict with keys: {list(teams_data.keys())}")
            # Look for nested structure
            for key, value in teams_data.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"\nFound list under key '{key}' with {len(value)} items")
                    print(f"First item in {key}:")
                    print(json.dumps(value[0], indent=2, default=str))
                    break
        else:
            print(f"Unexpected teams data format: {teams_data}")
            
    except Exception as e:
        print(f"Error getting teams: {e}")
        print(f"Error type: {type(e)}")
    
    # Test a specific team's players
    print(f"\n2. Testing get_players_by_team()...")
    try:
        # Try to get players for a known team ID (let's try 1 for New England Patriots)
        players_data = await client.get_players_by_team(1, 2023)
        print(f"Players response type: {type(players_data)}")
        print(f"Players response length: {len(players_data) if isinstance(players_data, (list, dict)) else 'N/A'}")
        
        if isinstance(players_data, list) and len(players_data) > 0:
            print(f"\nFirst player structure:")
            print(json.dumps(players_data[0], indent=2, default=str))
        elif isinstance(players_data, dict):
            print(f"\nPlayers data keys: {list(players_data.keys())}")
            
    except Exception as e:
        print(f"Error getting players: {e}")
        print(f"Error type: {type(e)}")
    
    # Get API client stats
    print(f"\n3. API Client Stats:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    await client.close()
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    asyncio.run(debug_nfl_api())