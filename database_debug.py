"""
Database Connection Debug Script
Debug why the database connection is failing.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add your src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_database_connection():
    """Debug database connection issues."""
    
    print("Fantasy Football AI - Database Connection Debug")
    print("=" * 60)
    
    try:
        # Step 1: Import and create database manager
        print("1. Importing database components...")
        
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
        
        print("   Imports successful")
        
        # Step 2: Create database manager
        print("\n2. Creating database manager...")
        
        db_manager = get_simple_db_manager()
        print(f"   Database path: {db_manager.db_path}")
        print(f"   Database file exists: {Path(db_manager.db_path).exists()}")
        print(f"   Database file size: {Path(db_manager.db_path).stat().st_size} bytes")
        
        # Step 3: Test connection with detailed error info
        print("\n3. Testing database connection...")
        
        try:
            connection_result = await db_manager.test_connection()
            print(f"   Connection result: {connection_result}")
            
            if not connection_result:
                print("   Connection returned False - investigating...")
                
        except Exception as e:
            print(f"   Connection test failed with exception: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
        
        # Step 4: Try to create a session manually
        print("\n4. Testing session creation...")
        
        try:
            async with db_manager.get_session() as session:
                print("   Session created successfully")
                
                # Test a simple query
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                test_value = result.scalar()
                print(f"   Simple query result: {test_value}")
                
        except Exception as e:
            print(f"   Session creation failed: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
        
        # Step 5: Check database schema
        print("\n5. Checking database schema...")
        
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                
                # Get table list
                result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
                print(f"   Tables found: {tables}")
                
                # Check each table structure
                for table in tables:
                    result = await session.execute(text(f"PRAGMA table_info({table})"))
                    columns = result.fetchall()
                    print(f"   {table} columns: {[col[1] for col in columns]}")
                    
                    # Get row count
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"   {table} row count: {count}")
                
        except Exception as e:
            print(f"   Schema check failed: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
        
        # Step 6: Test ORM models
        print("\n6. Testing ORM models...")
        
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                
                # Test Team model
                try:
                    result = await session.execute(select(Team).limit(1))
                    sample_team = result.scalar_one_or_none()
                    if sample_team:
                        print(f"   Sample Team: ID={sample_team.id}, Name={getattr(sample_team, 'name', 'N/A')}")
                    else:
                        print("   No teams found")
                except Exception as e:
                    print(f"   Team query failed: {e}")
                
                # Test Player model
                try:
                    result = await session.execute(select(Player).limit(1))
                    sample_player = result.scalar_one_or_none()
                    if sample_player:
                        print(f"   Sample Player: ID={sample_player.id}, Name={getattr(sample_player, 'name', 'N/A')}")
                        
                        # Show all player attributes
                        player_attrs = [attr for attr in dir(sample_player) if not attr.startswith('_') and not callable(getattr(sample_player, attr))]
                        print(f"   Player attributes: {player_attrs}")
                        
                    else:
                        print("   No players found")
                except Exception as e:
                    print(f"   Player query failed: {e}")
                
                # Test WeeklyStats model
                try:
                    result = await session.execute(select(WeeklyStats).limit(1))
                    sample_stats = result.scalar_one_or_none()
                    if sample_stats:
                        print(f"   Sample WeeklyStats: ID={sample_stats.id}")
                        
                        # Show all stats attributes
                        stats_attrs = [attr for attr in dir(sample_stats) if not attr.startswith('_') and not callable(getattr(sample_stats, attr))]
                        print(f"   WeeklyStats attributes: {stats_attrs}")
                        
                        # Show sample values
                        print("   Sample stat values:")
                        for attr in stats_attrs[:10]:  # First 10 attributes
                            try:
                                value = getattr(sample_stats, attr)
                                print(f"     {attr}: {value}")
                            except:
                                pass
                                
                    else:
                        print("   No weekly stats found")
                except Exception as e:
                    print(f"   WeeklyStats query failed: {e}")
        
        except Exception as e:
            print(f"   ORM model testing failed: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
        
        # Step 7: Check for fantasy-relevant columns
        print("\n7. Looking for fantasy football columns...")
        
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import text
                
                # Get all column names from all tables
                result = await session.execute(text("""
                    SELECT m.name as table_name, p.name as column_name, p.type as column_type
                    FROM sqlite_master m
                    LEFT OUTER JOIN pragma_table_info((m.name)) p ON m.name <> p.name
                    WHERE m.type = 'table'
                    ORDER BY m.name, p.cid
                """))
                
                columns = result.fetchall()
                
                # Look for fantasy-relevant columns
                fantasy_keywords = ['fantasy', 'points', 'passing', 'rushing', 'receiving', 
                                  'touchdown', 'yard', 'reception', 'target', 'fumble', 'interception']
                
                fantasy_columns = []
                for table_name, column_name, column_type in columns:
                    if column_name:  # Skip None values
                        for keyword in fantasy_keywords:
                            if keyword.lower() in column_name.lower():
                                fantasy_columns.append((table_name, column_name, column_type))
                                break
                
                if fantasy_columns:
                    print("   Fantasy-relevant columns found:")
                    for table, column, col_type in fantasy_columns:
                        print(f"     {table}.{column} ({col_type})")
                else:
                    print("   No obvious fantasy columns found")
                    
                    # Show all columns for debugging
                    print("   All columns in database:")
                    current_table = None
                    for table_name, column_name, column_type in columns:
                        if column_name:  # Skip None values
                            if table_name != current_table:
                                print(f"     {table_name}:")
                                current_table = table_name
                            print(f"       {column_name} ({column_type})")
        
        except Exception as e:
            print(f"   Fantasy column search failed: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
    
    except Exception as e:
        print(f"Debug failed completely: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(debug_database_connection())