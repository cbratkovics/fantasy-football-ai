#!/usr/bin/env python3
"""Test script for pulling real data from Sleeper API and storing in database"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats, Base
from backend.data.sleeper_client import SleeperAPIClient

class DataFetchTest:
    def __init__(self):
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.sleeper_client = SleeperAPIClient()
        
    async def test_fetch_players(self):
        """Test fetching NFL players from Sleeper API"""
        print("\n1. Testing Player Data Fetch")
        print("=" * 50)
        
        # Fetch all NFL players
        print("Fetching NFL players from Sleeper API...")
        players_data = await self.sleeper_client.get_all_players()
        
        if players_data:
            print(f"✅ Fetched {len(players_data)} players")
            
            # Show sample of players
            sample_players = list(players_data.items())[:5]
            print("\nSample players:")
            for player_id, player_info in sample_players:
                name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}"
                position = player_info.get('position', 'N/A')
                team = player_info.get('team', 'N/A')
                print(f"  - {name} ({position}) - {team}")
                
            # Store players in database
            await self.store_players(players_data)
        else:
            print("❌ Failed to fetch players")
            
    async def store_players(self, players_data):
        """Store players in database"""
        print("\n2. Storing Players in Database")
        print("=" * 50)
        
        db = self.SessionLocal()
        stored_count = 0
        
        try:
            # Only store active players with valid data
            for player_id, player_info in players_data.items():
                # Skip if no name or position
                if not player_info.get('first_name') or not player_info.get('position'):
                    continue
                    
                # Only store skill position players and active players
                if player_info.get('position') in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                    # Check if player already exists
                    existing = db.query(Player).filter(Player.player_id == player_id).first()
                    
                    if not existing:
                        player = Player(
                            player_id=player_id,
                            first_name=player_info.get('first_name', ''),
                            last_name=player_info.get('last_name', ''),
                            position=player_info.get('position'),
                            team=player_info.get('team'),
                            fantasy_positions=player_info.get('fantasy_positions', []),
                            age=player_info.get('age'),
                            years_exp=player_info.get('years_exp'),
                            status=player_info.get('status', 'Active'),
                            meta_data=player_info
                        )
                        db.add(player)
                        stored_count += 1
                        
                        # Commit in batches
                        if stored_count % 100 == 0:
                            db.commit()
                            print(f"  Stored {stored_count} players...")
                            
            db.commit()
            print(f"✅ Stored {stored_count} new players in database")
            
        except Exception as e:
            print(f"❌ Error storing players: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
    async def test_fetch_stats(self):
        """Test fetching player stats for recent weeks"""
        print("\n3. Testing Stats Data Fetch")
        print("=" * 50)
        
        # Fetch stats for 2023 season, week 1
        season = 2023
        week = 1
        
        print(f"Fetching stats for {season} Week {week}...")
        stats_data = await self.sleeper_client.get_week_stats("regular", season, week)
        
        if stats_data:
            print(f"✅ Fetched stats for {len(stats_data)} players")
            
            # Show top performers
            top_performers = []
            for player_id, stats in stats_data.items():
                if stats.get('pts_ppr', 0) > 0:
                    top_performers.append((player_id, stats.get('pts_ppr', 0)))
                    
            top_performers.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 performers (PPR points):")
            db = self.SessionLocal()
            for player_id, points in top_performers[:10]:
                player = db.query(Player).filter(Player.player_id == player_id).first()
                if player:
                    print(f"  - {player.full_name} ({player.position}): {points:.1f} pts")
            db.close()
            
            # Store stats in database
            await self.store_stats(stats_data, season, week)
        else:
            print("❌ Failed to fetch stats")
            
    async def store_stats(self, stats_data, season, week):
        """Store player stats in database"""
        print("\n4. Storing Stats in Database")
        print("=" * 50)
        
        db = self.SessionLocal()
        stored_count = 0
        
        try:
            for player_id, stats in stats_data.items():
                # Skip if no points scored
                if not any([stats.get('pts_std', 0), stats.get('pts_ppr', 0), stats.get('pts_half_ppr', 0)]):
                    continue
                    
                # Check if stats already exist
                existing = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season,
                    PlayerStats.week == week
                ).first()
                
                if not existing:
                    # Verify player exists
                    player = db.query(Player).filter(Player.player_id == player_id).first()
                    if player:
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
                        stored_count += 1
                        
            db.commit()
            print(f"✅ Stored stats for {stored_count} players")
            
        except Exception as e:
            print(f"❌ Error storing stats: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
    async def test_data_retrieval(self):
        """Test retrieving data from database"""
        print("\n5. Testing Data Retrieval from Database")
        print("=" * 50)
        
        db = self.SessionLocal()
        
        try:
            # Count total players
            player_count = db.query(Player).count()
            print(f"Total players in database: {player_count}")
            
            # Count by position
            print("\nPlayers by position:")
            positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
            for pos in positions:
                count = db.query(Player).filter(Player.position == pos).count()
                print(f"  {pos}: {count}")
                
            # Count stats records
            stats_count = db.query(PlayerStats).count()
            print(f"\nTotal stats records: {stats_count}")
            
            # Get recent top performers
            print("\nRecent top RB performances:")
            top_rbs = db.execute(text("""
                SELECT p.first_name, p.last_name, ps.season, ps.week, ps.fantasy_points_ppr
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE p.position = 'RB'
                ORDER BY ps.fantasy_points_ppr DESC
                LIMIT 5
            """))
            
            for row in top_rbs:
                print(f"  - {row.first_name} {row.last_name} ({row.season} W{row.week}): {row.fantasy_points_ppr:.1f} pts")
                
        except Exception as e:
            print(f"❌ Error retrieving data: {str(e)}")
        finally:
            db.close()
            
    async def run_all_tests(self):
        """Run all data fetching tests"""
        print("Starting Real Data Fetch Tests")
        print("=" * 70)
        
        # Test 1: Fetch and store players
        await self.test_fetch_players()
        
        # Test 2: Fetch and store stats
        await self.test_fetch_stats()
        
        # Test 3: Retrieve and display data
        await self.test_data_retrieval()
        
        print("\n" + "=" * 70)
        print("✅ All tests completed!")


async def main():
    """Main entry point"""
    tester = DataFetchTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())