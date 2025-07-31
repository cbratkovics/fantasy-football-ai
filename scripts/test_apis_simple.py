#!/usr/bin/env python3
"""
Simple test for API connections
"""

import os
import sys
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_apis():
    """Test API connections"""
    print("Testing API connections...\n")
    
    # Check API keys
    api_keys = {
        'SPORTSDATA_API_KEY': os.getenv('SPORTSDATA_API_KEY'),
        'OPENWEATHER_API_KEY': os.getenv('OPENWEATHER_API_KEY'),
        'CFBD_API_KEY': os.getenv('CFBD_API_KEY')
    }
    
    print("1. API Keys Status:")
    for key, value in api_keys.items():
        status = "✓ Present" if value else "✗ Missing"
        print(f"   {key}: {status}")
    
    # Test SportsData API
    print("\n2. Testing SportsData API...")
    if api_keys['SPORTSDATA_API_KEY']:
        try:
            async with aiohttp.ClientSession() as session:
                # Try a simpler endpoint first
                url = "https://api.sportsdata.io/v3/nfl/scores/json/Teams"
                headers = {'Ocp-Apim-Subscription-Key': api_keys['SPORTSDATA_API_KEY']}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ Success! Retrieved {len(data)} player records")
                        if data:
                            sample = data[0]
                            print(f"   Sample player: {sample.get('Name', 'Unknown')}")
                            print(f"   Team: {sample.get('Team', 'Unknown')}")
                            print(f"   Position: {sample.get('Position', 'Unknown')}")
                    else:
                        print(f"   ✗ Error: HTTP {response.status}")
                        text = await response.text()
                        print(f"   Response: {text[:200]}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Test OpenWeather API
    print("\n3. Testing OpenWeather API...")
    if api_keys['OPENWEATHER_API_KEY']:
        try:
            async with aiohttp.ClientSession() as session:
                # Test with current weather (simpler than historical)
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': 44.5013,  # Green Bay
                    'lon': -88.0622,
                    'appid': api_keys['OPENWEATHER_API_KEY'],
                    'units': 'imperial'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ Success! Current weather in Green Bay:")
                        print(f"   Temperature: {data.get('main', {}).get('temp')}°F")
                        print(f"   Conditions: {data.get('weather', [{}])[0].get('description')}")
                    else:
                        print(f"   ✗ Error: HTTP {response.status}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Test CFBD API
    print("\n4. Testing College Football Data API...")
    if api_keys['CFBD_API_KEY']:
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.collegefootballdata.com/games"
                params = {'year': 2023, 'seasonType': 'regular', 'week': 1}
                headers = {'Authorization': f'Bearer {api_keys["CFBD_API_KEY"]}'}
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ Success! Retrieved {len(data)} games")
                        if data:
                            print(f"   Sample game: {data[0].get('home_team')} vs {data[0].get('away_team')}")
                    else:
                        print(f"   ✗ Error: HTTP {response.status}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "="*50)
    print("API Testing Complete!")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(test_apis())