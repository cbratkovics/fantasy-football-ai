#!/usr/bin/env python3
"""
Simple test script to debug database and API issues.
Location: debug_test.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_database():
    """Test basic database functionality."""
    
    print("🗃️  Testing database...")
    
    try:
        from fantasy_ai.core.data.storage.database import DatabaseManager
        
        # Create simple database manager with minimal settings
        db_manager = DatabaseManager()
        
        print(f"Database URL: {db_manager.database_url}")
        
        # Test connection
        if await db_manager.test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            # Try creating tables
            print("Attempting to create tables...")
            await db_manager.create_tables(drop_existing=True)
            
            if await db_manager.test_connection():
                print("✅ Database connection successful after table creation")
            else:
                print("❌ Database connection still failing")
        
        await db_manager.close_connections()
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_api():
    """Test basic API functionality."""
    
    print("\n🔗 Testing API...")
    
    if not os.getenv('NFL_API_KEY'):
        print("❌ NFL_API_KEY not set")
        return
    
    try:
        import aiohttp
        import json
        
        headers = {
            'X-RapidAPI-Key': os.getenv('NFL_API_KEY'),
            'X-RapidAPI-Host': 'v1.american-football.api-sports.io'
        }
        
        async with aiohttp.ClientSession() as session:
            # Test basic connection
            url = "https://v1.american-football.api-sports.io/teams"
            params = {'league': 1}
            
            async with session.get(url, headers=headers, params=params) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    teams = data.get('response', [])
                    print(f"✅ API working - found {len(teams)} teams")
                    
                    # Show first team
                    if teams:
                        first_team = teams[0].get('team', {})
                        print(f"Sample team: {first_team.get('name', 'Unknown')}")
                else:
                    text = await response.text()
                    print(f"❌ API failed: {text[:200]}")
                    
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    
    print("🏈 Fantasy Football AI Debug Test")
    print("=" * 40)
    
    await test_database()
    await test_api()
    
    print("\n✅ Debug test completed")

if __name__ == "__main__":
    asyncio.run(main())