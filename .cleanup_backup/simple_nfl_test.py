#!/usr/bin/env python3
"""
Simple test for the corrected NFL API client.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from fantasy_ai.core.data.sources.nfl_comprehensive import NFLAPIClient

def main():
    print("Testing corrected NFL API client...")
    print("=" * 40)
    
    try:
        # Create client
        client = NFLAPIClient()
        
        # Test leagues to see the league IDs
        print("\n1. Testing leagues...")
        leagues = client.get_leagues()
        print(f"Found {len(leagues)} leagues:")
        for league_data in leagues:
            league = league_data.get('league', {})
            print(f"  - ID: {league.get('id')}, Name: {league.get('name')}")
        
        print(f"\nUsing NFL League ID: {client.nfl_league_id}")
        
        # Test teams with correct league ID
        print("\n2. Testing teams...")
        teams = client.get_teams(season=2024)
        print(f"Found {len(teams)} teams")
        
        if teams:
            print("Sample teams:")
            for team in teams[:5]:
                print(f"  - {team.name} ({team.code})")
            
            # Test players for first team
            print("\n3. Testing players...")
            test_team = teams[0]
            players = client.get_players(test_team.team_id, season=2024)
            print(f"Found {len(players)} players for {test_team.name}")
            
            if players:
                print("Sample players:")
                for player_data in players[:3]:
                    player = player_data.get('player', {})
                    print(f"  - {player.get('name')} ({player.get('position')})")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()