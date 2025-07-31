#!/usr/bin/env python3
"""Continue collecting missing NFL seasons"""

import os
import sys
import requests
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["DATABASE_URL"] = "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"

from backend.models.database import Player, PlayerStats

def collect_missing_seasons():
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    # Check what we need
    print("📊 CONTINUING DATA COLLECTION")
    print("="*60)
    
    # Missing: 2021 (partial), 2022 (none), 2023 (partial), 2024 (none)
    missing_data = [
        (2021, 8, 18),   # Complete 2021
        (2022, 1, 18),   # All of 2022
        (2023, 2, 18),   # Complete 2023
        (2024, 1, 10),   # 2024 up to week 10
    ]
    
    for season, start_week, end_week in missing_data:
        print(f"\n📊 Collecting {season} weeks {start_week}-{end_week}")
        
        for week in range(start_week, end_week + 1):
            try:
                url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
                response = requests.get(url, timeout=30)
                
                if response.status_code != 200:
                    print(f"  Week {week}: API error {response.status_code}")
                    continue
                    
                stats_data = response.json()
                stored = 0
                
                for player_id, stats in stats_data.items():
                    if not any([stats.get('pts_std', 0), stats.get('pts_ppr', 0)]):
                        continue
                        
                    # Check player exists
                    player = db.query(Player).filter(Player.player_id == player_id).first()
                    if not player:
                        continue
                        
                    # Check if stats exist
                    existing = db.query(PlayerStats).filter(
                        PlayerStats.player_id == player_id,
                        PlayerStats.season == season,
                        PlayerStats.week == week
                    ).first()
                    
                    if not existing:
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
                        
                if stored > 0:
                    db.commit()
                    print(f"  Week {week}: ✓ Stored {stored} records")
                else:
                    print(f"  Week {week}: No new records")
                    
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                print(f"  Week {week}: Error - {str(e)}")
                db.rollback()
                
    db.close()
    
    # Final verification
    print("\n🔍 FINAL DATA VERIFICATION")
    print("="*60)
    
    with engine.connect() as conn:
        # Total stats
        total_stats = conn.execute(text("SELECT COUNT(*) FROM player_stats")).scalar()
        print(f"Total stats records: {total_stats}")
        
        # By season
        result = conn.execute(text("""
            SELECT season, COUNT(*) as count 
            FROM player_stats 
            GROUP BY season 
            ORDER BY season
        """))
        
        print("\nRecords by season:")
        for row in result:
            status = "✓" if row.count >= 1000 else "✗"
            print(f"  {row.season}: {row.count} records {status}")
            
        # Check if we have all required seasons
        required = [2019, 2020, 2021, 2022, 2023, 2024]
        seasons_with_data = conn.execute(text("""
            SELECT DISTINCT season FROM player_stats ORDER BY season
        """)).fetchall()
        
        seasons_list = [row[0] for row in seasons_with_data]
        
        print("\nRequired seasons check:")
        all_present = True
        for season in required:
            if season in seasons_list:
                print(f"  {season}: ✓ PRESENT")
            else:
                print(f"  {season}: ✗ MISSING")
                all_present = False
                
        if total_stats >= 10000 and all_present:
            print("\n✅ DATA COLLECTION COMPLETE - Ready for ML training")
            return True
        else:
            print("\n❌ INCOMPLETE DATA - Cannot proceed with training")
            return False

if __name__ == "__main__":
    collect_missing_seasons()