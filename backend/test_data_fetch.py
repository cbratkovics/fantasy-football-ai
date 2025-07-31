#!/usr/bin/env python3
"""Test fetching real fantasy football data"""

import os
import sys
import requests
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats

def fetch_players_from_sleeper():
    """Fetch all NFL players from Sleeper API"""
    print("Fetching players from Sleeper API...")
    response = requests.get("https://api.sleeper.app/v1/players/nfl")
    
    if response.status_code == 200:
        players = response.json()
        print(f"✅ Fetched {len(players)} players")
        return players
    else:
        print(f"❌ Failed to fetch players: {response.status_code}")
        return None

def fetch_stats_from_sleeper(season=2023, week=1):
    """Fetch player stats for a specific week"""
    print(f"\nFetching stats for {season} Week {week}...")
    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
    response = requests.get(url)
    
    if response.status_code == 200:
        stats = response.json()
        print(f"✅ Fetched stats for {len(stats)} players")
        return stats
    else:
        print(f"❌ Failed to fetch stats: {response.status_code}")
        return None

def store_players_in_db(players_data):
    """Store players in database"""
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    stored = 0
    skipped = 0
    
    print("\nStoring players in database...")
    
    try:
        for player_id, player_info in players_data.items():
            # Skip if no position or inactive
            if not player_info.get('position') or player_info.get('position') not in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                skipped += 1
                continue
                
            # Check if player exists
            existing = db.query(Player).filter(Player.player_id == player_id).first()
            if existing:
                continue
                
            # Create new player
            player = Player(
                player_id=player_id,
                first_name=player_info.get('first_name', ''),
                last_name=player_info.get('last_name', ''),
                position=player_info.get('position'),
                team=player_info.get('team'),
                fantasy_positions=player_info.get('fantasy_positions', []),
                age=player_info.get('age'),
                years_exp=player_info.get('years_exp'),
                status=player_info.get('status', 'Unknown'),
                meta_data=player_info
            )
            
            db.add(player)
            stored += 1
            
            # Commit in batches
            if stored % 100 == 0:
                db.commit()
                print(f"  Stored {stored} players...")
                
        db.commit()
        print(f"✅ Stored {stored} new players (skipped {skipped})")
        
    except Exception as e:
        print(f"❌ Error storing players: {str(e)}")
        db.rollback()
    finally:
        db.close()

def store_stats_in_db(stats_data, season, week):
    """Store player stats in database"""
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    stored = 0
    
    print(f"\nStoring stats for {season} Week {week}...")
    
    try:
        for player_id, stats in stats_data.items():
            # Skip if no fantasy points
            if not any([stats.get('pts_std', 0), stats.get('pts_ppr', 0)]):
                continue
                
            # Check if player exists
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                continue
                
            # Check if stats already exist
            existing = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week == week
            ).first()
            
            if existing:
                continue
                
            # Create stats record
            player_stats = PlayerStats(
                player_id=player_id,
                season=season,
                week=week,
                stats=stats,
                fantasy_points_std=stats.get('pts_std', 0),
                fantasy_points_ppr=stats.get('pts_ppr', 0),
                fantasy_points_half=stats.get('pts_half_ppr', 0),
                opponent=stats.get('opponent'),
                is_home=stats.get('home') == 1 if 'home' in stats else None
            )
            
            db.add(player_stats)
            stored += 1
            
        db.commit()
        print(f"✅ Stored {stored} stat records")
        
    except Exception as e:
        print(f"❌ Error storing stats: {str(e)}")
        db.rollback()
    finally:
        db.close()

def display_sample_data():
    """Display sample data from database"""
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    
    try:
        # Player counts by position
        print("\nPlayers by position:")
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            count = db.query(Player).filter(Player.position == pos).count()
            print(f"  {pos}: {count}")
            
        # Total stats records
        stats_count = db.query(PlayerStats).count()
        print(f"\nTotal stat records: {stats_count}")
        
        # Top performers from most recent data
        print("\nTop 10 Fantasy Performers (PPR):")
        result = db.execute(text("""
            SELECT p.first_name, p.last_name, p.position, p.team, 
                   ps.season, ps.week, ps.fantasy_points_ppr
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE ps.fantasy_points_ppr > 0
            ORDER BY ps.fantasy_points_ppr DESC
            LIMIT 10
        """))
        
        for i, row in enumerate(result, 1):
            print(f"  {i}. {row.first_name} {row.last_name} ({row.position}, {row.team}) - "
                  f"{row.season} W{row.week}: {row.fantasy_points_ppr:.1f} pts")
                  
    except Exception as e:
        print(f"❌ Error displaying data: {str(e)}")
    finally:
        db.close()

def main():
    """Main test function"""
    print("FANTASY FOOTBALL DATA FETCH TEST")
    print("="*60)
    
    # 1. Fetch players
    players = fetch_players_from_sleeper()
    if players:
        # Store in database
        store_players_in_db(players)
    
    # 2. Fetch stats for multiple weeks
    for week in range(1, 4):  # Weeks 1-3
        stats = fetch_stats_from_sleeper(2023, week)
        if stats:
            store_stats_in_db(stats, 2023, week)
    
    # 3. Display summary
    display_sample_data()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()