"""
Database Schema Inspector
Run this script to understand your database structure for ML integration.

Usage: python database_inspector.py
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add your src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def inspect_database():
    """Inspect database schema and show sample data."""
    
    print("Fantasy Football AI - Database Schema Inspector")
    print("=" * 60)
    
    try:
        # Import your database components
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
        from sqlalchemy import select, func, inspect
        
        db_manager = get_simple_db_manager()
        
        print(f"Database Path: {db_manager.db_path}")
        print(f"Database Type: SQLite")
        
        # Test connection
        if not await db_manager.test_connection():
            print("ERROR: Cannot connect to database")
            return
        
        print("Database connection: SUCCESS")
        
        async with db_manager.get_session() as session:
            
            # 1. Show table structure
            print("\n1. DATABASE TABLES")
            print("-" * 30)
            
            inspector = inspect(session.bind)
            tables = inspector.get_table_names()
            
            for table_name in tables:
                print(f"\nTable: {table_name}")
                columns = inspector.get_columns(table_name)
                
                for column in columns:
                    print(f"  {column['name']:<20} {str(column['type']):<15} "
                          f"{'NULL' if column['nullable'] else 'NOT NULL'}")
            
            # 2. Show record counts
            print("\n\n2. TABLE RECORD COUNTS")
            print("-" * 30)
            
            # Teams
            result = await session.execute(select(func.count()).select_from(Team))
            team_count = result.scalar()
            print(f"Teams: {team_count:,}")
            
            # Players  
            result = await session.execute(select(func.count()).select_from(Player))
            player_count = result.scalar()
            print(f"Players: {player_count:,}")
            
            # Weekly Stats
            result = await session.execute(select(func.count()).select_from(WeeklyStats))
            stats_count = result.scalar()
            print(f"Weekly Stats: {stats_count:,}")
            
            # 3. Show sample data structure
            print("\n\n3. SAMPLE DATA STRUCTURE")
            print("-" * 30)
            
            # Sample Team
            result = await session.execute(select(Team).limit(1))
            sample_team = result.scalar_one_or_none()
            
            if sample_team:
                print(f"\nSample Team Record:")
                print(f"  ID: {sample_team.id}")
                print(f"  Name: {sample_team.name}")
                print(f"  City: {getattr(sample_team, 'city', 'N/A')}")
                print(f"  Conference: {getattr(sample_team, 'conference', 'N/A')}")
                print(f"  Division: {getattr(sample_team, 'division', 'N/A')}")
                
                # Show all attributes
                print(f"  All Team attributes: {[attr for attr in dir(sample_team) if not attr.startswith('_')]}")
            
            # Sample Player
            result = await session.execute(select(Player).limit(1))
            sample_player = result.scalar_one_or_none()
            
            if sample_player:
                print(f"\nSample Player Record:")
                print(f"  ID: {sample_player.id}")
                print(f"  Name: {sample_player.name}")
                print(f"  Position: {getattr(sample_player, 'position', 'N/A')}")
                print(f"  Team ID: {getattr(sample_player, 'team_id', 'N/A')}")
                
                # Show all attributes
                print(f"  All Player attributes: {[attr for attr in dir(sample_player) if not attr.startswith('_')]}")
            
            # Sample WeeklyStats
            result = await session.execute(select(WeeklyStats).limit(1))
            sample_stats = result.scalar_one_or_none()
            
            if sample_stats:
                print(f"\nSample WeeklyStats Record:")
                print(f"  ID: {sample_stats.id}")
                print(f"  Player ID: {getattr(sample_stats, 'player_id', 'N/A')}")
                print(f"  Week: {getattr(sample_stats, 'week', 'N/A')}")
                print(f"  Season: {getattr(sample_stats, 'season', 'N/A')}")
                
                # Show all numeric attributes (potential fantasy stats)
                stats_attrs = [attr for attr in dir(sample_stats) if not attr.startswith('_')]
                print(f"  All WeeklyStats attributes: {stats_attrs}")
                
                # Show sample values for key attributes
                print(f"\n  Sample stat values:")
                for attr in stats_attrs[:10]:  # Show first 10 attributes
                    try:
                        value = getattr(sample_stats, attr)
                        if not callable(value) and not attr.startswith('_'):
                            print(f"    {attr}: {value}")
                    except:
                        pass
            
            # 4. Check for fantasy-relevant columns
            print("\n\n4. FANTASY FOOTBALL COLUMN ANALYSIS")
            print("-" * 30)
            
            # Look for fantasy point related columns
            fantasy_keywords = ['fantasy', 'points', 'passing', 'rushing', 'receiving', 
                              'touchdown', 'yard', 'reception', 'target', 'fumble', 'interception']
            
            if sample_stats:
                stats_columns = [attr for attr in dir(sample_stats) if not attr.startswith('_') and not callable(getattr(sample_stats, attr))]
                
                print("Potential fantasy football columns in WeeklyStats:")
                for col in stats_columns:
                    for keyword in fantasy_keywords:
                        if keyword.lower() in col.lower():
                            try:
                                value = getattr(sample_stats, col)
                                print(f"  {col}: {value} (matches '{keyword}')")
                                break
                            except:
                                pass
            
            # 5. Check data distribution by position and season
            print("\n\n5. DATA DISTRIBUTION")
            print("-" * 30)
            
            if player_count > 0 and stats_count > 0:
                # Position distribution
                result = await session.execute(
                    select(Player.position, func.count(Player.id))
                    .group_by(Player.position)
                    .order_by(func.count(Player.id).desc())
                )
                
                position_counts = result.fetchall()
                print("\nPlayers by Position:")
                for position, count in position_counts:
                    print(f"  {position}: {count}")
                
                # Season distribution (if season column exists)
                try:
                    result = await session.execute(
                        select(WeeklyStats.season, func.count(WeeklyStats.id))
                        .group_by(WeeklyStats.season)
                        .order_by(WeeklyStats.season.desc())
                    )
                    
                    season_counts = result.fetchall()
                    print("\nStats by Season:")
                    for season, count in season_counts:
                        print(f"  {season}: {count:,} records")
                        
                except Exception as e:
                    print(f"  Could not get season distribution: {e}")
                
                # Week distribution
                try:
                    result = await session.execute(
                        select(WeeklyStats.week, func.count(WeeklyStats.id))
                        .group_by(WeeklyStats.week)
                        .order_by(WeeklyStats.week)
                    )
                    
                    week_counts = result.fetchall()
                    print("\nStats by Week:")
                    for week, count in week_counts[:5]:  # Show first 5 weeks
                        print(f"  Week {week}: {count:,} records")
                    if len(week_counts) > 5:
                        print(f"  ... and {len(week_counts) - 5} more weeks")
                        
                except Exception as e:
                    print(f"  Could not get week distribution: {e}")
            
            # 6. Show session management info
            print("\n\n6. SESSION MANAGEMENT INFO")
            print("-" * 30)
            print(f"Session type: {type(session)}")
            print(f"Database manager type: {type(db_manager)}")
            print(f"Connection method: get_simple_db_manager().get_session()")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nCannot import database modules. Check your src path and module structure.")
        print("Expected modules:")
        print("  - fantasy_ai.core.data.storage.simple_database")
        print("  - fantasy_ai.core.data.storage.models")
        
    except Exception as e:
        print(f"Inspection failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")


async def test_ml_data_format():
    """Test if we can extract data in ML-compatible format."""
    
    print("\n\n7. ML DATA FORMAT TEST")
    print("-" * 30)
    
    try:
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        from fantasy_ai.core.data.storage.models import Player, WeeklyStats
        from sqlalchemy import select
        
        db_manager = get_simple_db_manager()
        
        async with db_manager.get_session() as session:
            # Try to extract sample data in ML format
            query = (
                select(Player.id, Player.name, Player.position, 
                       WeeklyStats.week, WeeklyStats.season)
                .select_from(Player)
                .join(WeeklyStats, Player.id == WeeklyStats.player_id)
                .limit(5)
            )
            
            result = await session.execute(query)
            rows = result.fetchall()
            
            if rows:
                print("Successfully joined Player and WeeklyStats tables:")
                print("Sample ML-ready data format:")
                
                for row in rows:
                    print(f"  Player: {row.name} ({row.position}) - Week {row.week}, Season {row.season}")
                
                print("\nML Integration: READY")
                print("Your database structure supports ML integration.")
                
            else:
                print("No joined data found. Check if WeeklyStats has player_id foreign key.")
                
    except Exception as e:
        print(f"ML format test failed: {e}")
        print("This indicates potential issues with ML integration.")


if __name__ == "__main__":
    try:
        asyncio.run(inspect_database())
        asyncio.run(test_ml_data_format())
        
        print("\n\nINSPECTION COMPLETE")
        print("=" * 60)
        print("Please share this output so I can:")
        print("1. Identify your exact table/column names")
        print("2. Understand your session management")
        print("3. Map your schema to ML requirements")
        print("4. Fix the database integration in cli_commands.py")
        
    except KeyboardInterrupt:
        print("\nInspection cancelled by user.")
    except Exception as e:
        print(f"Inspector failed to run: {e}")
        print("\nPlease check:")
        print("1. Run from project root directory")
        print("2. Database is accessible") 
        print("3. Required modules are installed")