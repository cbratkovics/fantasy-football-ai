#!/usr/bin/env python3
"""
STRICT ML TRAINING - COMPLETE NFL DATA COLLECTION
No shortcuts, no samples, only real comprehensive data
"""

import os
import sys
import requests
import json
import time
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["DATABASE_URL"] = "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"

from backend.models.database import Player, PlayerStats

class StrictDataCollector:
    def __init__(self):
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.Session = sessionmaker(bind=self.engine)
        self.required_seasons = [2019, 2020, 2021, 2022, 2023, 2024]
        self.errors = []
        
    def verify_before_start(self):
        """Initial verification before data collection"""
        print("🔍 INITIAL DATABASE STATE")
        print("="*60)
        
        with self.engine.connect() as conn:
            # Current player count
            player_count = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
            print(f"Current players in database: {player_count}")
            
            # Current stats count
            stats_count = conn.execute(text("SELECT COUNT(*) FROM player_stats")).scalar()
            print(f"Current stats records: {stats_count}")
            
            # Check existing seasons
            result = conn.execute(text("""
                SELECT season, COUNT(*) as count 
                FROM player_stats 
                GROUP BY season 
                ORDER BY season
            """))
            
            existing_seasons = {}
            for row in result:
                existing_seasons[row.season] = row.count
                
            print("\nExisting data by season:")
            for season in self.required_seasons:
                count = existing_seasons.get(season, 0)
                status = "✓" if count > 1000 else "✗"
                print(f"  {season}: {count} records {status}")
                
        return existing_seasons
        
    def collect_all_players(self):
        """Collect ALL NFL players"""
        print("\n📊 PLAYER COLLECTION")
        print("├── Starting comprehensive player fetch...")
        
        try:
            # Fetch all players
            response = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=120)
            
            if response.status_code != 200:
                self.errors.append(f"Player API failed with status {response.status_code}")
                print(f"└── ✗ FAILED: API returned {response.status_code}")
                return False
                
            players_data = response.json()
            print(f"├── Fetched {len(players_data)} total players from API")
            
            # Store ALL players
            db = self.Session()
            stored = 0
            updated = 0
            
            try:
                for player_id, player_info in players_data.items():
                    # Skip if no first name (invalid player)
                    if not player_info.get('first_name'):
                        continue
                        
                    existing = db.query(Player).filter(Player.player_id == player_id).first()
                    
                    if existing:
                        # Update existing player
                        existing.team = player_info.get('team')
                        existing.status = player_info.get('status', 'Unknown')
                        existing.age = player_info.get('age')
                        existing.years_exp = player_info.get('years_exp')
                        existing.meta_data = player_info
                        updated += 1
                    else:
                        # Create new player
                        player = Player(
                            player_id=player_id,
                            first_name=player_info.get('first_name', ''),
                            last_name=player_info.get('last_name', ''),
                            position=player_info.get('position', 'Unknown'),
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
                    if (stored + updated) % 500 == 0:
                        db.commit()
                        print(f"├── Progress: {stored} new, {updated} updated")
                        
                db.commit()
                print(f"├── Stored {stored} new players")
                print(f"├── Updated {updated} existing players")
                
                # Verify
                total_players = db.query(Player).count()
                print(f"├── Total players in database: {total_players}")
                print("└── ✓ PASSED: Player collection complete")
                
                return True
                
            except Exception as e:
                db.rollback()
                self.errors.append(f"Player storage error: {str(e)}")
                print(f"└── ✗ FAILED: {str(e)}")
                return False
            finally:
                db.close()
                
        except Exception as e:
            self.errors.append(f"Player fetch error: {str(e)}")
            print(f"└── ✗ FAILED: {str(e)}")
            return False
            
    def collect_season_stats(self, season, start_week=1, end_week=18):
        """Collect complete season stats"""
        print(f"\n📊 SEASON {season} STATS COLLECTION")
        print(f"├── Collecting weeks {start_week}-{end_week}")
        
        db = self.Session()
        season_total = 0
        
        try:
            for week in range(start_week, end_week + 1):
                print(f"├── Week {week}:")
                
                # Check existing data
                existing_count = db.query(PlayerStats).filter(
                    PlayerStats.season == season,
                    PlayerStats.week == week
                ).count()
                
                print(f"│   ├── Existing records: {existing_count}")
                
                # Fetch from API
                url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
                
                try:
                    response = requests.get(url, timeout=60)
                    
                    if response.status_code != 200:
                        print(f"│   └── ✗ API error: {response.status_code}")
                        continue
                        
                    stats_data = response.json()
                    print(f"│   ├── Fetched {len(stats_data)} player stats")
                    
                    # Store stats
                    stored = 0
                    for player_id, stats in stats_data.items():
                        # Skip if no points
                        if not any([stats.get('pts_std', 0), stats.get('pts_ppr', 0)]):
                            continue
                            
                        # Verify player exists
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
                            
                    db.commit()
                    season_total += stored
                    print(f"│   └── ✓ Stored {stored} new records")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"│   └── ✗ Error: {str(e)}")
                    db.rollback()
                    
            print(f"├── Season {season} total new records: {season_total}")
            
            # Verify season completeness
            season_count = db.query(PlayerStats).filter(
                PlayerStats.season == season
            ).count()
            
            print(f"├── Total records for season {season}: {season_count}")
            
            if season_count < 1000:
                print(f"└── ✗ FAILED: Insufficient data for season {season}")
                return False
            else:
                print(f"└── ✓ PASSED: Season {season} collection complete")
                return True
                
        except Exception as e:
            self.errors.append(f"Season {season} error: {str(e)}")
            print(f"└── ✗ FAILED: {str(e)}")
            return False
        finally:
            db.close()
            
    def verify_data_completeness(self):
        """Mandatory verification of collected data"""
        print("\n🔍 MANDATORY DATA VERIFICATION")
        print("="*60)
        
        with self.engine.connect() as conn:
            # 1. Exact row count
            print("\n1. EXACT ROW COUNTS:")
            player_count = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
            stats_count = conn.execute(text("SELECT COUNT(*) FROM player_stats")).scalar()
            print(f"   Players table: {player_count} rows")
            print(f"   Player stats table: {stats_count} rows")
            
            # 2. Date range
            print("\n2. DATE RANGE:")
            result = conn.execute(text("""
                SELECT MIN(season), MAX(season), COUNT(DISTINCT season) 
                FROM player_stats
            """)).first()
            
            if result:
                min_season, max_season, season_count = result
                print(f"   Min season: {min_season}")
                print(f"   Max season: {max_season}")
                print(f"   Distinct seasons: {season_count}")
            else:
                print("   ✗ NO DATA FOUND")
                return False
                
            # 3. Player count
            print("\n3. DISTINCT PLAYERS:")
            distinct_players = conn.execute(text("""
                SELECT COUNT(DISTINCT player_id) FROM player_stats
            """)).scalar()
            print(f"   Players with stats: {distinct_players}")
            
            # 4. Random sample
            print("\n4. RANDOM SAMPLE (10 rows):")
            result = conn.execute(text("""
                SELECT p.first_name, p.last_name, p.position, 
                       ps.season, ps.week, ps.fantasy_points_ppr
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                ORDER BY RANDOM() 
                LIMIT 10
            """))
            
            for row in result:
                print(f"   {row.first_name} {row.last_name} ({row.position}) - "
                      f"{row.season} W{row.week}: {row.fantasy_points_ppr:.1f} pts")
                      
            # 5. Season breakdown
            print("\n5. DATA BY SEASON:")
            result = conn.execute(text("""
                SELECT season, COUNT(*) as count 
                FROM player_stats 
                GROUP BY season 
                ORDER BY season
            """))
            
            season_data = {}
            for row in result:
                season_data[row.season] = row.count
                status = "✓" if row.count >= 1000 else "✗"
                print(f"   {row.season}: {row.count} records {status}")
                
            # Check all required seasons
            print("\n6. REQUIRED SEASONS CHECK:")
            all_seasons_present = True
            for season in self.required_seasons:
                if season not in season_data or season_data[season] < 1000:
                    print(f"   {season}: ✗ MISSING OR INSUFFICIENT")
                    all_seasons_present = False
                else:
                    print(f"   {season}: ✓ PRESENT")
                    
            # Final validation
            print("\n7. FINAL VALIDATION:")
            if stats_count < 10000:
                print(f"   ✗ INSUFFICIENT DATA: Only {stats_count} rows (need >= 10,000)")
                return False
            elif not all_seasons_present:
                print(f"   ✗ INCOMPLETE DATA: Missing required seasons")
                return False
            else:
                print(f"   ✓ PASSED: All validation checks passed")
                return True
                
    def run_complete_collection(self):
        """Run the complete data collection process"""
        print("🚀 STRICT NFL DATA COLLECTION - NO SHORTCUTS")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initial state
        existing_seasons = self.verify_before_start()
        
        # Step 1: Collect all players
        if not self.collect_all_players():
            print("\n❌ ERROR: Player collection failed - Cannot proceed with real data training")
            return False
            
        # Step 2: Collect stats for each required season
        for season in self.required_seasons:
            # Skip if already have sufficient data
            if existing_seasons.get(season, 0) >= 1000:
                print(f"\n✓ Season {season} already has sufficient data")
                continue
                
            # Determine week range based on season
            if season == 2024:
                # Current season, limited weeks
                end_week = 10  # Adjust based on current week
            else:
                # Full season
                end_week = 18 if season >= 2021 else 17
                
            if not self.collect_season_stats(season, 1, end_week):
                print(f"\n❌ ERROR: Season {season} collection failed")
                
        # Step 3: Verify completeness
        if not self.verify_data_completeness():
            print("\n❌ TRAINING INCOMPLETE - Data validation failed")
            return False
            
        print("\n✅ DATA COLLECTION COMPLETE")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True


if __name__ == "__main__":
    collector = StrictDataCollector()
    
    # Ask for permission if running with limited time/resources
    print("\n⚠️  This will collect 6 seasons of NFL data (2019-2024)")
    print("⚠️  Estimated time: 30-60 minutes")
    print("⚠️  Required: Stable internet connection")
    
    response = input("\nProceed with COMPLETE data collection? (yes/no): ")
    
    if response.lower() == 'yes':
        success = collector.run_complete_collection()
        
        if not success:
            print("\n❌ Data collection failed. See errors above.")
            sys.exit(1)
    else:
        print("\n❌ Data collection cancelled by user")
        print("Cannot proceed with ML training without complete data")
        sys.exit(1)