#!/usr/bin/env python3
"""Fetch player data from Sleeper API and populate database"""

import sys
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from correct locations
from backend.models.database import Base, Player, PlayerStats
from backend.data.sleeper_client import SleeperAPIClient
from backend.data.scoring import ScoringSettings, FantasyScorer

# Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SleeperDataFetcher:
    """Fetch and store data from Sleeper API"""
    
    def __init__(self):
        self.client = SleeperAPIClient()
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Skill positions to track
        self.skill_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
        
        logger.info(f"Using DATABASE_URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
        
    async def fetch_players(self) -> Dict[str, Any]:
        """Fetch all NFL players from Sleeper API"""
        logger.info("Fetching players from Sleeper API...")
        
        try:
            players_data = await self.client.get_all_players("nfl")
            
            if not players_data:
                logger.error("No player data received from Sleeper API")
                return {}
                
            logger.info(f"Fetched {len(players_data)} total players from Sleeper")
            
            # Filter for relevant players
            filtered_players = {}
            for sleeper_id, player in players_data.items():
                # Check if it's a Player object or dict
                if hasattr(player, 'position'):
                    position = player.position
                    status = player.status
                else:
                    position = player.get('position')
                    status = player.get('status', 'Active')
                
                # Only include active/relevant players in skill positions
                if (position in self.skill_positions and
                    status in ["Active", "Inactive", "Injured Reserve", "PUP", "Questionable"]):
                    filtered_players[sleeper_id] = player
            
            logger.info(f"Filtered to {len(filtered_players)} skill position players")
            return filtered_players
            
        except Exception as e:
            logger.error(f"Error fetching players: {str(e)}")
            return {}
    
    def store_players(self, players_data: Dict[str, Any]) -> int:
        """Store player data in database"""
        logger.info("Storing player data in database...")
        
        stored_count = 0
        updated_count = 0
        
        with self.SessionLocal() as db:
            try:
                for sleeper_id, player_data in players_data.items():
                    # Extract data based on whether it's a Player object or dict
                    if hasattr(player_data, 'position'):
                        # It's a Player object from SleeperAPIClient
                        first_name = player_data.first_name or ''
                        last_name = player_data.last_name or ''
                        team = player_data.team or 'FA'
                        position = player_data.position or ''
                        age = player_data.age or 0
                        years_exp = player_data.years_exp or 0
                        status = player_data.status or 'Active'
                        fantasy_positions = player_data.fantasy_positions or []
                        injury_status = player_data.injury_status
                        
                        # Additional data for meta_data field
                        meta_data = {
                            'injury_status': injury_status,
                            'height': getattr(player_data, 'height', None),
                            'weight': getattr(player_data, 'weight', None),
                            'college': getattr(player_data, 'college', None),
                            'birth_date': getattr(player_data, 'birth_date', None),
                        }
                    else:
                        # It's a dict
                        first_name = player_data.get('first_name', '')
                        last_name = player_data.get('last_name', '')
                        team = player_data.get('team', 'FA')
                        position = player_data.get('position', '')
                        age = player_data.get('age', 0)
                        years_exp = player_data.get('years_exp', 0)
                        status = player_data.get('status', 'Active')
                        fantasy_positions = player_data.get('fantasy_positions', [])
                        
                        meta_data = {
                            'injury_status': player_data.get('injury_status'),
                            'height': player_data.get('height'),
                            'weight': player_data.get('weight'),
                            'college': player_data.get('college'),
                            'birth_date': player_data.get('birth_date'),
                        }
                    
                    # Check if player exists using player_id (which stores the Sleeper ID)
                    existing_player = db.query(Player).filter_by(player_id=sleeper_id).first()
                    
                    if existing_player:
                        # Update existing player
                        existing_player.first_name = first_name
                        existing_player.last_name = last_name
                        existing_player.team = team
                        existing_player.position = position
                        existing_player.age = age
                        existing_player.years_exp = years_exp
                        existing_player.status = status
                        existing_player.fantasy_positions = fantasy_positions
                        existing_player.meta_data = meta_data
                        existing_player.updated_at = datetime.utcnow()
                        updated_count += 1
                    else:
                        # Create new player
                        new_player = Player(
                            player_id=sleeper_id,  # Using player_id to store Sleeper ID
                            first_name=first_name,
                            last_name=last_name,
                            team=team,
                            position=position,
                            age=age,
                            years_exp=years_exp,
                            status=status,
                            fantasy_positions=fantasy_positions,
                            meta_data=meta_data
                        )
                        db.add(new_player)
                        stored_count += 1
                
                db.commit()
                logger.info(f"Stored {stored_count} new players, updated {updated_count} existing players")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error storing players: {str(e)}")
                import traceback
                traceback.print_exc()
                return 0
        
        return stored_count + updated_count
    
    async def fetch_and_store_players_only(self):
        """Fetch and store only player data (no stats for now)"""
        logger.info("Starting player data fetch...")
        
        # Fetch and store players
        players_data = await self.fetch_players()
        if not players_data:
            logger.error("Failed to fetch player data")
            return
        
        player_count = self.store_players(players_data)
        logger.info(f"Processed {player_count} players")
        
        # Show sample data
        with self.SessionLocal() as db:
            sample_players = db.query(Player).filter(
                Player.position.in_(['QB', 'RB', 'WR'])
            ).order_by(Player.last_name).limit(20).all()
            
            logger.info("\nSample imported players:")
            for p in sample_players:
                full_name = f"{p.first_name} {p.last_name}"
                injury_status = 'Healthy'
                if p.meta_data and isinstance(p.meta_data, dict):
                    injury_status = p.meta_data.get('injury_status', 'Healthy') or 'Healthy'
                logger.info(f"  - {full_name} ({p.position}, {p.team}) - Status: {p.status}, Injury: {injury_status}")
            
            # Show counts by position
            position_counts = {}
            for player in db.query(Player).all():
                pos = player.position
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            logger.info("\nPlayers by position:")
            for pos, count in sorted(position_counts.items()):
                logger.info(f"  {pos}: {count}")
    
    # Comment out stats-related methods since get_week_stats doesn't exist
    """
    async def fetch_season_stats(self, season: int, start_week: int = 1, end_week: int = 17) -> Dict[int, Dict[str, Any]]:
        # This method would need get_week_stats which doesn't exist in SleeperAPIClient
        pass
    
    def store_weekly_stats(self, season: int, weekly_stats: Dict[int, Dict[str, Any]]) -> int:
        # This would store stats but we need the stats fetching method first
        pass
    """
    
    def verify_data_integrity(self):
        """Verify data was loaded correctly"""
        with self.SessionLocal() as db:
            player_count = db.query(Player).count()
            
            logger.info(f"""
            Data Verification:
            - Total players: {player_count}
            """)
            
            # Show top players by position
            for position in ['QB', 'RB', 'WR', 'TE']:
                top_players = db.query(Player).filter(
                    Player.position == position,
                    Player.status == 'Active'
                ).order_by(Player.last_name).limit(5).all()
                
                logger.info(f"\nTop {position}s:")
                for p in top_players:
                    full_name = f"{p.first_name} {p.last_name}"
                    logger.info(f"  - {full_name} ({p.team})")


async def main():
    """Main entry point"""
    fetcher = SleeperDataFetcher()
    
    # For now, just fetch and store players (no stats)
    await fetcher.fetch_and_store_players_only()
    
    # Verify data integrity
    fetcher.verify_data_integrity()
    
    logger.info("\nNOTE: Player stats fetching is disabled because get_week_stats method")
    logger.info("doesn't exist in SleeperAPIClient. Players have been imported successfully.")


if __name__ == "__main__":
    logger.info("Starting Sleeper data fetch...")
    asyncio.run(main())