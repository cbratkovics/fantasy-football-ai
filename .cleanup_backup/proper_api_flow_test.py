#!/usr/bin/env python3
"""
Test script following the proper API architecture from documentation.
Architecture: Seasons → Leagues → Teams/Players/Games
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from fantasy_ai.core.data.sources.nfl_comprehensive import NFLAPIClient

def test_proper_api_flow():
    """Test API following the proper hierarchical flow."""
    print("Testing NFL API following proper architecture flow...")
    print("=" * 60)
    
    client = NFLAPIClient()
    
    # Step 1: Get available seasons
    print("\n1. Getting available seasons...")
    seasons_response = client._make_request("seasons")
    
    print("Raw seasons response:")
    print(json.dumps(seasons_response, indent=2)[:300] + "...")
    
    if seasons_response and 'response' in seasons_response:
        seasons = seasons_response['response']
        print(f"Found {len(seasons)} seasons:")
        
        available_seasons = []
        for season_data in seasons:
            # Handle both integer seasons and object seasons
            if isinstance(season_data, int):
                season_year = season_data
            elif isinstance(season_data, dict):
                season_year = season_data.get('season')
            else:
                continue
                
            if season_year:
                available_seasons.append(season_year)
                print(f"  - Season: {season_year}")
        
        if not available_seasons:
            print("No seasons found, trying with manual season values...")
            available_seasons = [2024, 2023, 2022]
    else:
        print("Could not get seasons, trying with manual values...")
        available_seasons = [2024, 2023, 2022]
    
    # Step 2: For each available season, get leagues
    working_season = None
    working_league_id = None
    
    for season in available_seasons[:3]:  # Test first 3 seasons
        print(f"\n2. Getting leagues for season {season}...")
        
        # Try getting leagues with season parameter
        leagues_response = client._make_request("leagues", {"season": season})
        
        print(f"Raw leagues response for season {season}:")
        print(json.dumps(leagues_response, indent=2)[:300] + "...")
        
        if leagues_response and 'response' in leagues_response:
            leagues = leagues_response['response']
            print(f"Found {len(leagues)} leagues for season {season}:")
            
            for league_data in leagues:
                league = league_data.get('league', {})
                league_id = league.get('id')
                league_name = league.get('name', '')
                
                print(f"  - ID: {league_id}, Name: {league_name}")
                
                if league_name.upper() == 'NFL':
                    working_season = season
                    working_league_id = league_id
                    print(f"  *** Found NFL league: ID {league_id} for season {season}")
                    break
            
            if working_season:
                break
        else:
            print(f"No leagues found for season {season}")
    
    # Also try getting leagues without season parameter
    if not working_season:
        print(f"\n2b. Getting leagues without season parameter...")
        leagues_response = client._make_request("leagues")
        
        print("Raw leagues response (no season):")
        print(json.dumps(leagues_response, indent=2)[:300] + "...")
        
        if leagues_response and 'response' in leagues_response:
            leagues = leagues_response['response']
            print(f"Found {len(leagues)} leagues:")
            
            for league_data in leagues:
                league = league_data.get('league', {})
                league_id = league.get('id')
                league_name = league.get('name', '')
                
                print(f"  - ID: {league_id}, Name: {league_name}")
                
                if league_name.upper() == 'NFL':
                    working_season = available_seasons[0] if available_seasons else 2024
                    working_league_id = league_id
                    print(f"  *** Found NFL league: ID {league_id}")
                    break
    
    if not working_season or not working_league_id:
        print("\nCould not find NFL league via API, trying known values...")
        working_season = available_seasons[0] if available_seasons else 2024
        working_league_id = 1  # We know from previous tests NFL is league 1
    
    # Step 3: Get teams using the correct season and league combination
    print(f"\n3. Getting teams for NFL (League {working_league_id}, Season {working_season})...")
    
    teams_params = {
        "league": working_league_id,
        "season": working_season
    }
    
    print(f"Using parameters: {teams_params}")
    teams_response = client._make_request("teams", teams_params)
    
    if teams_response:
        print("Raw teams response:")
        print(json.dumps(teams_response, indent=2)[:1000] + "...")
        
        teams = teams_response.get('response', [])
        print(f"\nFound {len(teams)} teams")
        
        if teams:
            print("Success! Sample teams:")
            for team_data in teams[:5]:
                # Handle different team data structures
                if isinstance(team_data, dict):
                    team = team_data.get('team', team_data)  # team might be at root level
                    team_name = team.get('name', 'Unknown')
                    team_id = team.get('id', 'Unknown')
                    print(f"  - {team_name} (ID: {team_id})")
                else:
                    print(f"  - Unexpected team data type: {type(team_data)}")
            
            return working_season, working_league_id, teams
        else:
            # Check for errors
            if 'errors' in teams_response:
                print(f"API Errors: {teams_response['errors']}")
            elif 'error' in teams_response:
                print(f"API Error: {teams_response['error']}")
    else:
        print("No response received")
    # Step 4: If teams still not working, try alternative approaches
    print("\n4. Trying alternative approaches...")
    
    alternative_params = [
        {"league": 1},  # No season
        {"season": working_season},  # No league
        {},  # No parameters
        {"league": 1, "season": str(working_season)},  # Season as string
        {"league": "1", "season": working_season},  # League as string
    ]
    
    for i, params in enumerate(alternative_params):
        print(f"\n--- Alternative {i+1}: {params} ---")
        response = client._make_request("teams", params)
        
        if response:
            print(f"Response keys: {list(response.keys())}")
            if 'response' in response:
                teams = response.get('response', [])
                print(f"Result: {len(teams)} teams")
                
                if teams:
                    # Handle different team data structures
                    team_data = teams[0]
                    if isinstance(team_data, dict):
                        team = team_data.get('team', team_data)
                        team_name = team.get('name', 'Unknown')
                        print(f"Sample: {team_name}")
                        print("SUCCESS with alternative parameters!")
                        return working_season, working_league_id, teams
            
            # Check for errors
            if 'errors' in response:
                print(f"Errors: {response['errors']}")
            elif 'error' in response:
                print(f"Error: {response['error']}")
        else:
            print("No response received")
    
    print("\nAll attempts failed. Need to check API documentation or contact support.")
    return None, None, []

def test_other_endpoints():
    """Test other endpoints to see what's working."""
    print("\n\n" + "="*60)
    print("TESTING OTHER ENDPOINTS")
    print("="*60)
    
    client = NFLAPIClient()
    
    endpoints_to_test = [
        ("games", {"league": 1, "season": 2024}),
        ("games", {"league": 1}),
        ("games", {}),
        ("standings", {"league": 1, "season": 2024}),
        ("standings", {"league": 1}),
        ("players", {"league": 1, "season": 2024}),
        ("players", {"league": 1}),
    ]
    
    for endpoint, params in endpoints_to_test:
        print(f"\n--- Testing {endpoint} with {params} ---")
        response = client._make_request(endpoint, params)
        
        if response:
            print(f"Response keys: {list(response.keys())}")
            
            if 'response' in response:
                data = response.get('response', [])
                print(f"Result: {len(data)} items")
                
                if data:
                    if endpoint == "games":
                        game = data[0]
                        teams = game.get('teams', {})
                        home_team = teams.get('home', {}).get('name', 'Unknown')
                        away_team = teams.get('away', {}).get('name', 'Unknown')
                        print(f"Sample game: {away_team} @ {home_team}")
                    elif endpoint == "standings":
                        standing = data[0]
                        print(f"Sample standing: {standing}")
                    elif endpoint == "players":
                        player = data[0]
                        player_info = player.get('player', player)
                        print(f"Sample player: {player_info.get('name', 'Unknown')}")
            
            # Check for errors
            if 'errors' in response:
                print(f"Errors: {response['errors']}")
            elif 'error' in response:
                print(f"Error: {response['error']}")
        else:
            print("No response received")

def main():
    """Run the complete test."""
    try:
        season, league_id, teams = test_proper_api_flow()
        
        if teams:
            print(f"\n\nSUCCESS SUMMARY:")
            print(f"Working Season: {season}")
            print(f"Working League ID: {league_id}")
            print(f"Teams Found: {len(teams)}")
            print("\nThe API is working! You can proceed with data collection.")
        else:
            print(f"\n\nTEST SUMMARY:")
            print("Teams endpoint not working with standard parameters.")
            print("Testing other endpoints to see what works...")
            test_other_endpoints()
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()