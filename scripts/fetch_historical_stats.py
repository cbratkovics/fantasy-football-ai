#!/usr/bin/env python3
"""
Fetch historical player stats from Sleeper API
This script fetches season-by-season stats for all players
"""

import sys
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


class HistoricalStatsFetcher:
    """Fetch and store historical stats from Sleeper API"""
    
    def __init__(self):
        self.client = SleeperAPIClient()
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.scorer = FantasyScorer()
        
        # Seasons to fetch (last 3 years)
        self.seasons = ['2021', '2022', '2023']  # Last 3 seasons
        self.weeks_per_season = 18  # Full regular season
        
    async def fetch_stats_for_week(self, season: str, week: int) -> Dict[str, Any]:
        """Fetch stats for a specific week"""
        logger.info(f"Fetching stats for {season} Week {week}...")
        
        try:
            stats = await self.client.get_stats(
                sport='nfl',
                season_type='regular',
                season=season,
                week=week
            )
            return stats
        except Exception as e:
            logger.error(f"Error fetching stats for {season} Week {week}: {str(e)}")
            return {}
    
    def store_player_stats(self, stats_data: Dict[str, Any], season: int, week: int, db: Session) -> int:
        """Store player stats in database"""
        stored_count = 0
        
        for player_id, stats in stats_data.items():
            try:
                # Skip if no stats
                if not stats or not isinstance(stats, dict):
                    continue
                
                # Check if player exists
                player = db.query(Player).filter(Player.player_id == player_id).first()
                if not player:
                    continue
                
                # Check if stats already exist
                existing_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season,
                    PlayerStats.week == week
                ).first()
                
                if existing_stats:
                    continue
                
                # Get fantasy points from Sleeper's pre-calculated values
                # Sleeper provides pts_std, pts_ppr, pts_half_ppr in the stats
                fantasy_points = {
                    'std': float(stats.get('pts_std', 0.0)),
                    'ppr': float(stats.get('pts_ppr', 0.0)),
                    'half': float(stats.get('pts_half_ppr', 0.0))
                }
                
                # Create new stats entry
                new_stats = PlayerStats(
                    player_id=player_id,
                    season=season,
                    week=week,
                    stats=stats,  # Store raw stats as JSONB
                    fantasy_points_std=fantasy_points['std'],
                    fantasy_points_ppr=fantasy_points['ppr'],
                    fantasy_points_half=fantasy_points['half'],
                    opponent=stats.get('opponent'),
                    is_home=stats.get('home') == 1 if 'home' in stats else None,
                    game_date=None  # Would need to fetch from schedule
                )
                
                db.add(new_stats)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing stats for player {player_id}: {str(e)}")
                continue
        
        return stored_count
    
    async def fetch_season_stats(self, season: str):
        """Fetch all stats for a season"""
        logger.info(f"Starting to fetch stats for {season} season...")
        
        total_stored = 0
        
        with self.SessionLocal() as db:
            for week in range(1, self.weeks_per_season + 1):
                # Fetch stats for the week
                week_stats = await self.fetch_stats_for_week(season, week)
                
                if not week_stats:
                    logger.warning(f"No stats found for {season} Week {week}")
                    continue
                
                # Store stats
                stored = self.store_player_stats(week_stats, int(season), week, db)
                total_stored += stored
                
                if stored > 0:
                    db.commit()
                    logger.info(f"Stored {stored} player stats for {season} Week {week}")
                
                # Small delay to be respectful to API
                await asyncio.sleep(0.5)
        
        logger.info(f"Completed {season} season. Total stats stored: {total_stored}")
        return total_stored
    
    async def fetch_all_historical_stats(self):
        """Fetch stats for all configured seasons"""
        logger.info("Starting historical stats fetch...")
        
        async with self.client:
            for season in self.seasons:
                await self.fetch_season_stats(season)
        
        # Show summary
        with self.SessionLocal() as db:
            total_stats = db.query(PlayerStats).count()
            
            logger.info(f"\nHistorical Stats Summary:")
            logger.info(f"Total player-week stats: {total_stats}")
            
            # Stats by season
            for season in self.seasons:
                season_count = db.query(PlayerStats).filter(
                    PlayerStats.season == int(season)
                ).count()
                logger.info(f"  {season}: {season_count} entries")
            
            # Top performers
            top_qbs = db.query(
                Player.first_name,
                Player.last_name,
                PlayerStats.fantasy_points_ppr
            ).join(
                Player, Player.player_id == PlayerStats.player_id
            ).filter(
                Player.position == 'QB'
            ).order_by(
                PlayerStats.fantasy_points_ppr.desc()
            ).limit(5).all()
            
            logger.info("\nTop 5 QB performances (PPR):")
            for qb in top_qbs:
                logger.info(f"  {qb[0]} {qb[1]}: {qb[2]:.1f} points")


async def main():
    """Main entry point"""
    fetcher = HistoricalStatsFetcher()
    await fetcher.fetch_all_historical_stats()


if __name__ == "__main__":
    logger.info("Starting historical stats fetch...")
    asyncio.run(main())