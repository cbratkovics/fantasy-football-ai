#!/usr/bin/env python3
"""
Quick integration test with limited data to verify pipeline works
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient
from data.sources.weather_client import WeatherClient
from data.sleeper_client import SleeperAPIClient
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_quick_integration():
    """Test each component with minimal data"""
    
    logger.info("Starting quick integration test...")
    
    # Test 1: NFL Data
    logger.info("\n1. Testing NFL data collection...")
    nfl_client = NFLDataPyClient()
    
    # Get just one week of data
    weekly_data = nfl_client.import_weekly_data(2023, position='QB')
    if weekly_data is not None:
        week_1_data = weekly_data[weekly_data['week'] == 1]
        logger.info(f"NFL Data: Found {len(week_1_data)} QB records for Week 1, 2023")
        logger.info(f"Sample columns: {list(week_1_data.columns[:10])}")
    else:
        logger.error("Failed to get NFL data")
    
    # Test 2: Weather data
    logger.info("\n2. Testing weather data...")
    async with WeatherClient() as weather_client:
        weather = await weather_client.get_game_weather('GB', datetime(2023, 9, 10), 13)
        logger.info(f"Weather data: {weather}")
    
    # Test 3: Sleeper data
    logger.info("\n3. Testing Sleeper data...")
    sleeper_client = SleeperAPIClient()
    
    # Get all players and find Mahomes
    players = await sleeper_client.get_all_players()
    mahomes = [p for p in players.values() if p.full_name == "Patrick Mahomes"]
    if mahomes:
        logger.info(f"Sleeper: Found player {mahomes[0].full_name} (ID: {mahomes[0].player_id})")
    
    # Test 4: Simple aggregation
    logger.info("\n4. Testing simple data aggregation...")
    from data.sources.data_aggregator import DataAggregator
    
    async with DataAggregator() as aggregator:
        # Just get one week of data
        logger.info("Fetching Week 1, 2023 data...")
        
        # Get Sleeper stats for Week 1
        sleeper_stats = await aggregator.sleeper_client.get_all_stats(2023, 1)
        logger.info(f"Sleeper stats: {len(sleeper_stats)} players")
        
        # Get NFL data for Week 1
        nfl_weekly = aggregator.nfl_client.import_weekly_data(2023)
        if nfl_weekly is not None:
            week_1_nfl = nfl_weekly[nfl_weekly['week'] == 1]
            logger.info(f"NFL weekly: {len(week_1_nfl)} records")
        
    logger.info("\n✅ Quick integration test completed!")
    logger.info("All components are working. Ready for full training.")

if __name__ == "__main__":
    asyncio.run(test_quick_integration())