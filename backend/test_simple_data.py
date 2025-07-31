#!/usr/bin/env python3
"""Simple test for pulling real data from Sleeper API"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres")
os.environ["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6379")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats
from backend.data.sleeper_client import SleeperAPIClient

async def test_simple_fetch():
    """Simple test to fetch and display data"""
    client = SleeperAPIClient()
    
    print("1. Testing Player Fetch")
    print("-" * 40)
    
    # Get all players
    players = await client.get_all_players()
    print(f"✅ Fetched {len(players)} players from Sleeper API")
    
    # Show some QB data
    print("\nSample QBs:")
    qb_count = 0
    for player_id, player in players.items():
        if hasattr(player, 'position') and player.position == 'QB':
            print(f"  - {player.full_name} ({player.team})")
            qb_count += 1
            if qb_count >= 5:
                break
                
    print("\n2. Testing Stats Fetch")
    print("-" * 40)
    
    # Get stats for a recent week
    stats = await client.get_week_stats("regular", 2023, 1)
    print(f"✅ Fetched stats for {len(stats)} players")
    
    # Show top scorers
    print("\nTop scorers (Week 1, 2023):")
    scores = []
    for player_id, player_stats in stats.items():
        if isinstance(player_stats, dict) and 'pts_ppr' in player_stats:
            scores.append((player_id, player_stats['pts_ppr']))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for player_id, points in scores[:5]:
        if player_id in players:
            player = players[player_id]
            print(f"  - {player.full_name} ({player.position}): {points:.1f} pts")
            
    print("\n3. Database Storage Test")
    print("-" * 40)
    
    # Test database connection
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Store a few players
        stored = 0
        for player_id, player in list(players.items())[:10]:
            if hasattr(player, 'position') and player.position in ['QB', 'RB', 'WR', 'TE']:
                # Check if exists
                existing = db.query(Player).filter(Player.player_id == player_id).first()
                if not existing:
                    db_player = Player(
                        player_id=player_id,
                        first_name=player.first_name,
                        last_name=player.last_name,
                        position=player.position,
                        team=player.team,
                        fantasy_positions=player.fantasy_positions,
                        age=player.age,
                        years_exp=player.years_exp,
                        status=player.status,
                        meta_data=player.metadata
                    )
                    db.add(db_player)
                    stored += 1
                    
        db.commit()
        print(f"✅ Stored {stored} players in database")
        
        # Query back
        player_count = db.query(Player).count()
        print(f"✅ Total players in database: {player_count}")
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        db.rollback()
    finally:
        db.close()
        
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_simple_fetch())