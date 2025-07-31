#!/usr/bin/env python3
"""Test database connection and basic operations"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats

# Set DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres")

print(f"Testing connection to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

try:
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    # Test basic connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        
        # Check if players table has data
        result = conn.execute(text("SELECT COUNT(*) FROM players"))
        player_count = result.scalar()
        print(f"✅ Players table has {player_count} records")
        
        # Check if player_stats table has data
        result = conn.execute(text("SELECT COUNT(*) FROM player_stats"))
        stats_count = result.scalar()
        print(f"✅ Player stats table has {stats_count} records")
        
        # Check recent stats
        result = conn.execute(text("""
            SELECT season, week, COUNT(*) as count 
            FROM player_stats 
            GROUP BY season, week 
            ORDER BY season DESC, week DESC 
            LIMIT 5
        """))
        
        print("\nRecent stats data:")
        for row in result:
            print(f"  Season {row.season}, Week {row.week}: {row.count} records")
            
except Exception as e:
    print(f"❌ Error: {str(e)}")
    sys.exit(1)

print("\n✅ All database tests passed!")