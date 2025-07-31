#!/usr/bin/env python3
"""Quick test to verify data fetching works"""

import os
import sys
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set DATABASE_URL
os.environ["DATABASE_URL"] = "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats

def quick_test():
    print("QUICK DATA FETCH TEST")
    print("="*50)
    
    # 1. Test Sleeper API
    print("\n1. Testing Sleeper API...")
    try:
        # Get a few top QBs
        response = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=30)
        if response.status_code == 200:
            all_players = response.json()
            print(f"✅ API returned {len(all_players)} players")
            
            # Find some popular QBs
            qbs = []
            for pid, pinfo in all_players.items():
                if pinfo.get('position') == 'QB' and pinfo.get('team') and pinfo.get('search_rank', 10000) < 1000:
                    qbs.append((pid, pinfo))
                    if len(qbs) >= 5:
                        break
                        
            print("\nTop QBs found:")
            for pid, pinfo in qbs:
                print(f"  - {pinfo['first_name']} {pinfo['last_name']} ({pinfo['team']})")
        else:
            print(f"❌ API error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error calling API: {str(e)}")
        return
        
    # 2. Test Database Storage
    print("\n2. Testing Database Storage...")
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Store the QBs
        stored = 0
        for pid, pinfo in qbs:
            existing = db.query(Player).filter(Player.player_id == pid).first()
            if not existing:
                player = Player(
                    player_id=pid,
                    first_name=pinfo['first_name'],
                    last_name=pinfo['last_name'],
                    position='QB',
                    team=pinfo['team'],
                    fantasy_positions=pinfo.get('fantasy_positions', ['QB']),
                    age=pinfo.get('age'),
                    years_exp=pinfo.get('years_exp'),
                    status=pinfo.get('status', 'Active'),
                    meta_data=pinfo
                )
                db.add(player)
                stored += 1
                
        db.commit()
        print(f"✅ Stored {stored} new QBs")
        
        # Query back
        total_qbs = db.query(Player).filter(Player.position == 'QB').count()
        print(f"✅ Total QBs in database: {total_qbs}")
        
        # Show stored QBs
        print("\nQBs in database:")
        qbs_in_db = db.query(Player).filter(Player.position == 'QB').limit(10).all()
        for qb in qbs_in_db:
            print(f"  - {qb.full_name} ({qb.team})")
            
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        db.rollback()
    finally:
        db.close()
        
    # 3. Test Stats Fetch
    print("\n3. Testing Stats Fetch...")
    try:
        # Get Week 1 2023 stats
        response = requests.get("https://api.sleeper.app/v1/stats/nfl/regular/2023/1", timeout=30)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Fetched stats for {len(stats)} players")
            
            # Find top scorers
            top_scorers = []
            for pid, pstats in stats.items():
                if pstats.get('pts_ppr', 0) > 20:
                    top_scorers.append((pid, pstats.get('pts_ppr', 0)))
                    
            top_scorers.sort(key=lambda x: x[1], reverse=True)
            print(f"\nTop {min(5, len(top_scorers))} PPR scorers Week 1, 2023:")
            
            for pid, points in top_scorers[:5]:
                # Try to get player name from our data
                if pid in all_players:
                    p = all_players[pid]
                    print(f"  - {p['first_name']} {p['last_name']} ({p['position']}): {points:.1f} pts")
                else:
                    print(f"  - Player {pid}: {points:.1f} pts")
                    
        else:
            print(f"❌ Stats API error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching stats: {str(e)}")
        
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    quick_test()