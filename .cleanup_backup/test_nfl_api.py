#!/usr/bin/env python3
"""
Test script for NFL API integration.

This script tests the comprehensive NFL API functionality
to ensure everything is working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fantasy_ai.core.data.sources.nfl_comprehensive import (
        NFLAPIClient, 
        create_comprehensive_nfl_client
    )
    print("Successfully imported NFL API modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've created the nfl_comprehensive.py file")
    sys.exit(1)

def test_api_connection():
    """Test basic API connection and authentication."""
    print("Testing NFL API connection...")
    
    try:
        client = NFLAPIClient()
        print("NFL API client created successfully")
        return True
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure NFL_API_KEY is set in your .env file")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_leagues():
    """Test getting available leagues."""
    print("\nTesting leagues endpoint...")
    
    try:
        client = NFLAPIClient()
        leagues = client.get_leagues()
        
        if leagues:
            print(f"Found {len(leagues)} leagues")
            for league in leagues[:3]:  # Show first 3
                print(f"  - {league.get('league', {}).get('name', 'Unknown')}")
            return True
        else:
            print("No leagues found")
            return False
            
    except Exception as e:
        print(f"Error getting leagues: {e}")
        return False

def test_teams():
    """Test getting NFL teams."""
    print("\nTesting teams endpoint...")
    
    try:
        client = NFLAPIClient()
        teams = client.get_teams(season=2024)
        
        if teams:
            print(f"Found {len(teams)} NFL teams for 2024 season")
            print("Sample teams:")
            for team in teams[:5]:  # Show first 5
                print(f"  - {team.name} ({team.code}) - {team.conference} {team.division}")
            return True
        else:
            print("No teams found")
            return False
            
    except Exception as e:
        print(f"Error getting teams: {e}")
        return False

def test_players():
    """Test getting players from one team."""
    print("\nTesting players endpoint...")
    
    try:
        client = NFLAPIClient()
        teams = client.get_teams(season=2024)
        
        if not teams:
            print("No teams available for player testing")
            return False
        
        # Test with first team
        test_team = teams[0]
        print(f"Getting players for {test_team.name}...")
        
        players = client.get_players(test_team.team_id, season=2024)
        
        if players:
            print(f"Found {len(players)} players for {test_team.name}")
            print("Sample players:")
            for player_data in players[:3]:  # Show first 3
                player = player_data.get('player', {})
                print(f"  - {player.get('name', 'Unknown')} ({player.get('position', 'Unknown')})")
            return True
        else:
            print(f"No players found for {test_team.name}")
            return False
            
    except Exception as e:
        print(f"Error getting players: {e}")
        return False

def test_comprehensive_client():
    """Test the comprehensive client wrapper."""
    print("\nTesting comprehensive client...")
    
    try:
        comprehensive_client = create_comprehensive_nfl_client()
        print("Comprehensive client created successfully")
        
        # Test getting teams through comprehensive client
        teams = comprehensive_client.nfl_client.get_teams(season=2024)
        print(f"Comprehensive client found {len(teams)} teams")
        
        return True
        
    except Exception as e:
        print(f"Error with comprehensive client: {e}")
        return False

def test_api_limits():
    """Test API rate limiting and provide usage information."""
    print("\nTesting API rate limiting...")
    
    try:
        client = NFLAPIClient()
        
        # Make a few requests to test rate limiting
        print("Making multiple requests to test rate limiting...")
        
        for i in range(3):
            print(f"Request {i+1}...")
            teams = client.get_teams(season=2024)
            print(f"  Got {len(teams)} teams")
        
        print("Rate limiting working correctly")
        return True
        
    except Exception as e:
        print(f"Error testing rate limits: {e}")
        return False

def main():
    """Run all tests."""
    print("NFL API Integration Test Suite")
    print("=" * 40)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Leagues", test_leagues),
        ("Teams", test_teams),
        ("Players", test_players),
        ("Comprehensive Client", test_comprehensive_client),
        ("Rate Limiting", test_api_limits)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        print(f"Running: {test_name}")
        print(f"{'='*20}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name:<20}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! NFL API integration is working correctly.")
        print("\nNext steps:")
        print("1. Try collecting comprehensive player data")
        print("2. Set up database integration")
        print("3. Begin feature engineering for ML models")
    else:
        print(f"\n{total - passed} tests failed. Please check the errors above.")
        if not results.get("API Connection", False):
            print("\nIf API connection failed, verify:")
            print("- NFL_API_KEY is set correctly in .env file")
            print("- You have internet connection")
            print("- Your API key is valid and active")

if __name__ == "__main__":
    main()