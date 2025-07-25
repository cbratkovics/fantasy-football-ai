"""
Setup Script for Fantasy Football AI Database
This script initializes the database and collects initial data.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add your src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database_and_collect_data():
    """Initialize database and collect initial data."""
    
    print("Fantasy Football AI - Database Setup")
    print("=" * 50)
    
    try:
        # Step 1: Initialize database
        print("1. Initializing database...")
        
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        
        db_manager = get_simple_db_manager()
        
        # Create tables
        await db_manager.create_tables(drop_existing=True)
        print(f"   Database created at: {db_manager.db_path}")
        
        # Test connection
        if await db_manager.test_connection():
            print("   Database connection: SUCCESS")
        else:
            print("   Database connection: FAILED")
            return False
        
        # Step 2: Check if we need API setup
        print("\n2. Checking API configuration...")
        
        import os
        if not os.getenv('NFL_API_KEY'):
            print("   WARNING: NFL_API_KEY not set")
            print("   Set with: export NFL_API_KEY='your_key_here'")
            print("   Skipping data collection for now")
            return True
        else:
            print("   NFL_API_KEY found")
        
        # Step 3: Test API connection
        print("\n3. Testing API connection...")
        
        try:
            from fantasy_ai.core.data.sources.nfl_comprehensive import create_nfl_client
            
            client = await create_nfl_client()
            
            try:
                health_status = await client.health_check()
                
                if health_status['api_accessible']:
                    print("   API connection: SUCCESS")
                    print(f"   Response time: {health_status['response_time']:.2f}s")
                    api_available = True
                else:
                    print("   API connection: FAILED")
                    if health_status.get('last_error'):
                        print(f"   Error: {health_status['last_error']}")
                    api_available = False
            finally:
                await client.close()
                
        except Exception as e:
            print(f"   API test failed: {e}")
            api_available = False
        
        # Step 4: Collect initial data if API is available
        if api_available:
            print("\n4. Collecting initial data...")
            
            try:
                from fantasy_ai.core.data.etl import FantasyFootballETL, CollectionConfig
                
                # Limited collection for setup
                config = CollectionConfig(
                    api_calls_per_day=20,  # Conservative for setup
                    priority_positions=['QB', 'RB', 'WR', 'TE'],
                    target_seasons=[2023],  # Just one season for setup
                    max_concurrent_tasks=1
                )
                
                etl = FantasyFootballETL(config)
                
                print("   Initializing ETL pipeline...")
                if await etl.initialize_pipeline():
                    print("   Running limited data collection...")
                    metrics = await etl.run_full_pipeline()
                    
                    print(f"   Players processed: {metrics.total_players_processed}")
                    print(f"   Stats collected: {metrics.total_stats_collected}")
                    print(f"   API calls made: {metrics.total_api_calls}")
                    print(f"   Quality score: {metrics.validation_score:.3f}")
                else:
                    print("   Failed to initialize ETL pipeline")
                    
            except Exception as e:
                print(f"   Data collection failed: {e}")
                print("   Database is initialized but empty")
        else:
            print("\n4. Skipping data collection (API not available)")
        
        # Step 5: Verify database contents
        print("\n5. Verifying database contents...")
        
        async with db_manager.get_session() as session:
            from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
            from sqlalchemy import select, func
            
            # Get table counts
            result = await session.execute(select(func.count()).select_from(Team))
            team_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(Player))
            player_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(WeeklyStats))
            stats_count = result.scalar()
            
            print(f"   Teams: {team_count}")
            print(f"   Players: {player_count}")
            print(f"   Weekly Stats: {stats_count}")
        
        print("\n" + "=" * 50)
        print("DATABASE SETUP COMPLETE")
        
        if team_count > 0 or player_count > 0:
            print("Database has data - ready for ML integration")
            return True
        else:
            print("Database is empty but ready for data collection")
            return True
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMissing required modules. Check your project structure:")
        print("Expected:")
        print("  src/fantasy_ai/core/data/storage/simple_database.py")
        print("  src/fantasy_ai/core/data/storage/models.py")
        return False
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

async def show_next_steps():
    """Show next steps based on setup results."""
    
    print("\nNEXT STEPS:")
    print("-" * 20)
    
    print("1. Run the database inspector:")
    print("   python database_inspector.py")
    
    print("\n2. If you need to collect more data:")
    print("   python src/fantasy_ai/cli/main.py collect quick --max-calls 50")
    
    print("\n3. Once data is available, test ML integration:")
    print("   python src/fantasy_ai/cli/main.py ml status")
    
    print("\n4. Train ML models:")
    print("   python src/fantasy_ai/cli/main.py ml train --seasons 2023 --epochs 30")

if __name__ == "__main__":
    try:
        success = asyncio.run(setup_database_and_collect_data())
        
        if success:
            asyncio.run(show_next_steps())
        else:
            print("\nSetup failed. Please check error messages above.")
            
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
    except Exception as e:
        print(f"Setup script failed: {e}")