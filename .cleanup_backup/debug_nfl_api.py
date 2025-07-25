#!/usr/bin/env python3
"""
Debug script for NFL API integration.

This script helps debug API responses to understand the correct
parameters and data structure.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from fantasy_ai.core.data.sources.nfl_comprehensive import NFLAPIClient

def debug_api_response(endpoint, params=None):
    """Debug API response to see raw data."""
    print(f"\nDebugging endpoint: {endpoint}")
    print(f"Parameters: {params}")
    print("-" * 50)
    
    try:
        client = NFLAPIClient()
        response = client._make_request(endpoint, params)
        
        print("Raw API Response:")
        print(json.dumps(response, indent=2))
        
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def debug_leagues():
    """Debug leagues endpoint to understand structure."""
    print("DEBUGGING LEAGUES ENDPOINT")
    print("=" * 50)
    
    response = debug_api_response("leagues")
    
    if response and 'response' in response:
        leagues = response['response']
        print(f"\nFound {len(leagues)} leagues:")
        
        for i, league_data in enumerate(leagues):
            league = league_data.get('league', {})
            print(f"\nLeague {i+1}:")
            print(f"  ID: {league.get('id')}")
            print(f"  Name: {league.get('name')}")
            print(f"  Type: {league.get('type')}")
            print(f"  Logo: {league.get('logo')}")
            
            # Show seasons if available
            seasons = league_data.get('seasons', [])
            if seasons:
                print(f"  Available seasons: {[s.get('season') for s in seasons[:5]]}")
    
    return response

def debug_teams_with_different_params():
    """Try different parameters for teams endpoint."""
    print("\n\nDEBUGGING TEAMS ENDPOINT WITH DIFFERENT PARAMETERS")
    print("=" * 60)
    
    # Try different combinations
    test_params = [
        {"league": 1, "season": 2024},
        {"league": 2, "season": 2024},  # Maybe NFL is league 2
        {"league": 1, "season": 2023},
        {"league": 1},  # Without season
        {"season": 2024},  # Without league
        {}  # No parameters
    ]
    
    for i, params in enumerate(test_params):
        print(f"\n--- Test {i+1}: {params} ---")
        response = debug_api_response("teams", params)
        
        if response and 'response' in response:
            teams = response['response']
            print(f"Result: Found {len(teams)} teams")
            
            if teams:
                # Show first team as example
                first_team = teams[0]
                team_info = first_team.get('team', {})
                print(f"Sample team: {team_info.get('name')} (ID: {team_info.get('id')})")
                break  # Stop at first successful result
        else:
            print("Result: No teams found or error")

def debug_games_endpoint():
    """Debug games endpoint to see if we can get schedule data."""
    print("\n\nDEBUGGING GAMES ENDPOINT")
    print("=" * 40)
    
    test_params = [
        {"league": 1, "season": 2024},
        {"league": 2, "season": 2024},
        {"season": 2024}
    ]
    
    for params in test_params:
        print(f"\n--- Testing games with: {params} ---")
        response = debug_api_response("games", params)
        
        if response and 'response' in response:
            games = response['response']
            print(f"Found {len(games)} games")
            
            if games:
                # Show first game
                first_game = games[0]
                print(f"Sample game: {first_game.get('date')} - Week {first_game.get('week')}")
                print(f"Teams: {first_game.get('teams', {}).get('away', {}).get('name')} vs {first_game.get('teams', {}).get('home', {}).get('name')}")
                break

def debug_api_info():
    """Get general API information."""
    print("\n\nDEBUGGING API INFO")
    print("=" * 30)
    
    # Try to get status or info endpoint
    info_endpoints = ["status", "timezone", "countries"]
    
    for endpoint in info_endpoints:
        print(f"\n--- Testing {endpoint} endpoint ---")
        response = debug_api_response(endpoint)
        
        if response:
            print(f"Success: Got response from {endpoint}")

def check_api_key():
    """Verify API key is working."""
    print("CHECKING API KEY")
    print("=" * 30)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('NFL_API_KEY')
    if api_key:
        print(f"API Key found: {api_key[:10]}...{api_key[-5:]}")
        print(f"Key length: {len(api_key)}")
    else:
        print("No API key found in environment")
        return False
    
    return True

def main():
    """Run debug analysis."""
    print("NFL API Debug Analysis")
    print("=" * 50)
    
    # Check API key first
    if not check_api_key():
        print("Please set NFL_API_KEY in your .env file")
        return
    
    # Debug leagues to understand structure
    leagues_response = debug_leagues()
    
    # Debug teams with different parameters
    debug_teams_with_different_params()
    
    # Debug games endpoint
    debug_games_endpoint()
    
    # Check other endpoints
    debug_api_info()
    
    print("\n\nDEBUG SUMMARY")
    print("=" * 30)
    print("Based on the debug output above:")
    print("1. Check which league ID corresponds to NFL")
    print("2. Verify the correct season format")
    print("3. Look for any error messages in the API responses")
    print("4. Check if your API key has access to teams endpoint")
    
    print("\nIf you see successful responses with data, we can update")
    print("the main client with the correct parameters.")

if __name__ == "__main__":
    main()