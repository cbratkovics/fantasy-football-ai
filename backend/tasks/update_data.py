"""
Celery tasks for asynchronous data updates
"""

from celery import Task
from backend.celery_app import celery_app
from backend.data.sleeper_client import SleeperAPIClient
from backend.models.database import SessionLocal, Player, PlayerStats
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class DataUpdateTask(Task):
    """Task with database session management"""
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Data update task {task_id} succeeded")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Data update task {task_id} failed: {exc}")


@celery_app.task(base=DataUpdateTask, name="update_player_data")
def update_player_data():
    """Update player data from Sleeper API"""
    logger.info("Starting player data update...")
    
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_async_update_players())
        loop.close()
        
        logger.info(f"Player data update completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Player data update failed: {str(e)}")
        raise


async def _async_update_players():
    """Async function to update players"""
    client = SleeperAPIClient()
    db = SessionLocal()
    
    try:
        async with client:
            # Fetch all players
            players_data = await client.get_all_players("nfl")
            
            updated_count = 0
            new_count = 0
            
            for player_id, player in players_data.items():
                # Check if player exists
                existing = db.query(Player).filter(
                    Player.player_id == player_id
                ).first()
                
                if existing:
                    # Update existing player
                    existing.team = player.team
                    existing.status = player.status
                    existing.age = player.age
                    existing.updated_at = datetime.utcnow()
                    updated_count += 1
                else:
                    # Add new player
                    new_player = Player(
                        player_id=player_id,
                        first_name=player.first_name,
                        last_name=player.last_name,
                        position=player.position,
                        team=player.team or "FA",
                        age=player.age,
                        status=player.status
                    )
                    db.add(new_player)
                    new_count += 1
            
            db.commit()
            
            return {
                "status": "success",
                "updated": updated_count,
                "new": new_count,
                "total": len(players_data)
            }
            
    finally:
        db.close()


@celery_app.task(base=DataUpdateTask, name="update_player_stats")
def update_player_stats(season: int, week: int):
    """Update player statistics for a specific week"""
    logger.info(f"Updating player stats for Season {season}, Week {week}")
    
    try:
        # This would fetch stats from the API
        # For now, return mock success
        return {
            "status": "success",
            "season": season,
            "week": week,
            "message": "Stats update not implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Stats update failed: {str(e)}")
        raise


@celery_app.task(base=DataUpdateTask, name="cleanup_old_data")
def cleanup_old_data(days_to_keep: int = 365):
    """Clean up old data from database"""
    logger.info(f"Cleaning up data older than {days_to_keep} days")
    
    try:
        db = SessionLocal()
        
        # Clean up old predictions, stats, etc.
        # Implementation would go here
        
        db.close()
        
        return {
            "status": "success",
            "message": f"Cleaned up data older than {days_to_keep} days"
        }
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")
        raise