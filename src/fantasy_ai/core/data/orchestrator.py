"""
Intelligent Data Collection Orchestrator with Priority Queuing and Adaptive Rate Limiting.
Location: src/fantasy_ai/core/data/orchestrator.py
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from pydantic import BaseModel, Field

from .storage.models import (
    CollectionTask, CollectionStatus, Player, Team, WeeklyStats,
    ApiRateLimit, CollectionProgress, PlayerPosition, calculate_fantasy_priority_score
)
from .storage.database import get_db_session
from .rate_limiter import AdaptiveRateLimiter, RateLimitConfig
from .sources.nfl_comprehensive import NFLAPIClient
from .quality.anomaly_detector import DataQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class CollectionConfig:
    """Configuration for data collection orchestrator."""
    max_concurrent_tasks: int = 3
    batch_size: int = 10
    api_calls_per_day: int = 100
    priority_positions: List[str] = None
    target_seasons: List[int] = None
    enable_quality_validation: bool = True
    retry_exponential_base: float = 2.0
    max_retry_delay: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.priority_positions is None:
            self.priority_positions = ['QB', 'RB', 'WR', 'TE']
        if self.target_seasons is None:
            self.target_seasons = [2021, 2022, 2023]

class TaskRequest(BaseModel):
    """Request for creating a collection task."""
    task_type: str
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    season: Optional[int] = None
    week: Optional[int] = None
    priority: int = Field(default=3, ge=1, le=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CollectionOrchestrator:
    """
    Intelligent orchestrator for NFL data collection with advanced prioritization,
    rate limiting, and quality monitoring.
    """
    
    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        self.rate_limiter = AdaptiveRateLimiter(
            RateLimitConfig(
                requests_per_day=self.config.api_calls_per_day,
                requests_per_hour=self.config.api_calls_per_day // 24,
                requests_per_minute=5
            )
        )
        self.nfl_client = NFLAPIClient()
        self.quality_validator = DataQualityValidator()
        
        # State tracking
        self.is_running = False
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'api_calls_made': 0,
            'start_time': None
        }

    async def start_collection(self) -> None:
        """Start the intelligent data collection process."""
        logger.info("Starting intelligent data collection orchestrator")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Initialize collection progress tracking
            await self._initialize_collection_progress()
            
            # Generate initial task queue
            await self._generate_priority_tasks()
            
            # Start main collection loop
            await self._run_collection_loop()
            
        except Exception as e:
            logger.error(f"Collection orchestrator error: {e}")
            raise
        finally:
            self.is_running = False

    async def _initialize_collection_progress(self) -> None:
        """Initialize or update collection progress tracking."""
        async with get_db_session() as session:
            # Get or create progress record
            progress = session.query(CollectionProgress).first()
            if not progress:
                progress = CollectionProgress(
                    collection_start_date=datetime.now(timezone.utc),
                    current_season=self.config.target_seasons[-1]
                )
                session.add(progress)
            
            # Update totals based on current database state
            total_teams = session.query(Team).count()
            priority_players = session.query(Player).filter(
                Player.position.in_(self.config.priority_positions),
                Player.is_active == True
            ).count()
            
            progress.total_players = priority_players
            progress.total_seasons = len(self.config.target_seasons)
            progress.total_weeks = 18 * len(self.config.target_seasons)  # 17 regular + 1 playoff
            
            # Calculate API calls remaining today
            today = datetime.now(timezone.utc).date()
            rate_limit = session.query(ApiRateLimit).filter(
                ApiRateLimit.api_name == 'nfl_api',
                ApiRateLimit.reset_time >= today
            ).first()
            
            if rate_limit:
                progress.api_calls_remaining = max(0, rate_limit.requests_limit - rate_limit.requests_made)
            else:
                progress.api_calls_remaining = self.config.api_calls_per_day
            
            session.commit()
            logger.info(f"Collection progress initialized: {priority_players} priority players, "
                       f"{progress.api_calls_remaining} API calls remaining")

    async def _generate_priority_tasks(self) -> None:
        """Generate prioritized collection tasks using intelligent scheduling."""
        async with get_db_session() as session:
            logger.info("Generating intelligent priority task queue")
            
            # Clear old pending tasks
            session.query(CollectionTask).filter(
                CollectionTask.status == CollectionStatus.PENDING.value
            ).delete()
            
            tasks_created = 0
            
            # Priority 1: High-value players missing recent data
            priority_players = await self._get_priority_players(session)
            
            for player in priority_players:
                # Check what data is missing for this player
                missing_data = await self._analyze_missing_data(session, player)
                
                for season, weeks in missing_data.items():
                    if season not in self.config.target_seasons:
                        continue
                    
                    # Create tasks for missing weeks with intelligent priority
                    priority = self._calculate_task_priority(player, season, weeks)
                    
                    for week in weeks[:5]:  # Limit to prevent queue overflow
                        task = CollectionTask(
                            task_type='player_stats',
                            player_id=player.id,
                            team_id=player.team_id,
                            season=season,
                            week=week,
                            priority=priority.value,
                            scheduled_at=datetime.now(timezone.utc)
                        )
                        session.add(task)
                        tasks_created += 1
            
            # Priority 2: Team data updates
            teams_needing_update = session.query(Team).filter(
                or_(
                    Team.updated_at < datetime.now(timezone.utc) - timedelta(days=7),
                    Team.updated_at.is_(None)
                )
            ).limit(5).all()
            
            for team in teams_needing_update:
                task = CollectionTask(
                    task_type='team_info',
                    team_id=team.id,
                    priority=TaskPriority.MEDIUM.value,
                    scheduled_at=datetime.now(timezone.utc)
                )
                session.add(task)
                tasks_created += 1
            
            session.commit()
            logger.info(f"Generated {tasks_created} prioritized collection tasks")

    async def _get_priority_players(self, session: Session) -> List[Player]:
        """Get players ordered by collection priority."""
        return session.query(Player).filter(
            Player.position.in_(self.config.priority_positions),
            Player.is_active == True
        ).order_by(
            asc(Player.collection_priority),
            desc(Player.fantasy_priority_score)
        ).limit(100).all()

    async def _analyze_missing_data(self, session: Session, player: Player) -> Dict[int, List[int]]:
        """Analyze what data is missing for a player."""
        missing_data = {}
        
        for season in self.config.target_seasons:
            # Get existing weeks for this player/season
            existing_weeks = set(
                row[0] for row in session.query(WeeklyStats.week).filter(
                    WeeklyStats.player_id == player.id,
                    WeeklyStats.season == season
                ).all()
            )
            
            # Determine expected weeks (1-17 for regular season)
            expected_weeks = set(range(1, 18))
            missing_weeks = list(expected_weeks - existing_weeks)
            
            if missing_weeks:
                missing_data[season] = sorted(missing_weeks)
        
        return missing_data

    def _calculate_task_priority(self, player: Player, season: int, weeks: List[int]) -> TaskPriority:
        """Calculate intelligent task priority based on multiple factors."""
        
        # Base priority by position
        position_priorities = {
            'QB': TaskPriority.HIGH,
            'RB': TaskPriority.HIGH,
            'WR': TaskPriority.MEDIUM,
            'TE': TaskPriority.MEDIUM
        }
        
        base_priority = position_priorities.get(player.position, TaskPriority.LOW)
        
        # Adjust based on fantasy priority score
        if player.fantasy_priority_score > 0.8:
            if base_priority.value > 1:
                return TaskPriority(base_priority.value - 1)
        
        # Recent seasons get higher priority
        if season == max(self.config.target_seasons):
            if base_priority.value > 1:
                return TaskPriority(base_priority.value - 1)
        
        return base_priority

    async def _run_collection_loop(self) -> None:
        """Main collection loop with intelligent task processing."""
        logger.info("Starting intelligent collection loop")
        
        while self.is_running:
            try:
                # Check rate limits
                if not await self.rate_limiter.can_make_request():
                    wait_time = await self.rate_limiter.get_wait_time()
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(min(wait_time, 300))  # Max 5 min wait
                    continue
                
                # Get next batch of tasks
                tasks = await self._get_next_task_batch()
                
                if not tasks:
                    logger.info("No tasks available, checking for new work")
                    await self._generate_priority_tasks()
                    await asyncio.sleep(30)  # Wait before checking again
                    continue
                
                # Process tasks concurrently
                await self._process_task_batch(tasks)
                
                # Update progress
                await self._update_collection_progress()
                
                # Brief pause between batches
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _get_next_task_batch(self) -> List[CollectionTask]:
        """Get next batch of tasks ordered by intelligent priority."""
        async with get_db_session() as session:
            # Get available task slots
            active_tasks = len(self.current_tasks)
            available_slots = self.config.max_concurrent_tasks - active_tasks
            
            if available_slots <= 0:
                return []
            
            # Get highest priority pending tasks
            tasks = session.query(CollectionTask).filter(
                CollectionTask.status == CollectionStatus.PENDING.value,
                or_(
                    CollectionTask.scheduled_at <= datetime.now(timezone.utc),
                    CollectionTask.scheduled_at.is_(None)
                )
            ).order_by(
                asc(CollectionTask.priority),
                asc(CollectionTask.scheduled_at)
            ).limit(min(available_slots, self.config.batch_size)).all()
            
            # Mark tasks as in progress
            for task in tasks:
                task.status = CollectionStatus.IN_PROGRESS.value
                task.started_at = datetime.now(timezone.utc)
            
            session.commit()
            return tasks

    async def _process_task_batch(self, tasks: List[CollectionTask]) -> None:
        """Process a batch of collection tasks concurrently."""
        if not tasks:
            return
        
        logger.info(f"Processing batch of {len(tasks)} tasks")
        
        # Create async tasks for each collection task
        async_tasks = []
        for task in tasks:
            async_task = asyncio.create_task(self._process_single_task(task))
            self.current_tasks[str(task.id)] = async_task
            async_tasks.append(async_task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Clean up completed tasks
        for task, result in zip(tasks, results):
            task_id = str(task.id)
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
            
            if isinstance(result, Exception):
                logger.error(f"Task {task.id} failed: {result}")

    async def _process_single_task(self, task: CollectionTask) -> bool:
        """Process a single collection task with error handling and retries."""
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Processing task {task.id}: {task.task_type}")
            
            # Wait for rate limit
            await self.rate_limiter.wait_for_request()
            
            # Process based on task type
            if task.task_type == 'player_stats':
                success = await self._collect_player_stats(task)
            elif task.task_type == 'team_info':
                success = await self._collect_team_info(task)
            elif task.task_type == 'player_info':
                success = await self._collect_player_info(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                success = False
            
            # Update task status
            async with get_db_session() as session:
                db_task = session.query(CollectionTask).get(task.id)
                if db_task:
                    db_task.completed_at = datetime.now(timezone.utc)
                    db_task.api_response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    
                    if success:
                        db_task.status = CollectionStatus.COMPLETED.value
                        self.stats['tasks_completed'] += 1
                    else:
                        db_task.status = CollectionStatus.FAILED.value
                        self.stats['tasks_failed'] += 1
                        
                        # Schedule retry if attempts remain
                        if db_task.retry_count < db_task.max_retries:
                            await self._schedule_retry(db_task)
                    
                    session.commit()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            
            # Update task with error
            async with get_db_session() as session:
                db_task = session.query(CollectionTask).get(task.id)
                if db_task:
                    db_task.status = CollectionStatus.FAILED.value
                    db_task.error_message = str(e)
                    db_task.completed_at = datetime.now(timezone.utc)
                    
                    if db_task.retry_count < db_task.max_retries:
                        await self._schedule_retry(db_task)
                    
                    session.commit()
            
            return False

    async def _collect_player_stats(self, task: CollectionTask) -> bool:
        """Collect player statistics for a specific week."""
        try:
            # Get player stats from API
            stats_data = await self.nfl_client.get_player_stats(
                player_id=task.player_id,
                season=task.season,
                week=task.week
            )
            
            if not stats_data:
                logger.warning(f"No stats data returned for player {task.player_id}, "
                              f"season {task.season}, week {task.week}")
                return False
            
            # Store in database
            async with get_db_session() as session:
                # Check if record already exists
                existing = session.query(WeeklyStats).filter(
                    WeeklyStats.player_id == task.player_id,
                    WeeklyStats.season == task.season,
                    WeeklyStats.week == task.week
                ).first()
                
                if existing:
                    # Update existing record
                    self._update_weekly_stats(existing, stats_data)
                else:
                    # Create new record
                    weekly_stats = WeeklyStats(
                        player_id=task.player_id,
                        season=task.season,
                        week=task.week,
                        raw_api_data=stats_data
                    )
                    self._update_weekly_stats(weekly_stats, stats_data)
                    session.add(weekly_stats)
                
                # Update collection task
                task.records_collected = 1
                task.api_calls_made = 1
                
                session.commit()
                
            # Validate data quality if enabled
            if self.config.enable_quality_validation:
                await self.quality_validator.validate_player_stats(
                    task.player_id, task.season, task.week
                )
            
            self.stats['api_calls_made'] += 1
            logger.info(f"Successfully collected stats for player {task.player_id}, "
                       f"season {task.season}, week {task.week}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting player stats: {e}")
            return False

    def _update_weekly_stats(self, weekly_stats: WeeklyStats, stats_data: Dict[str, Any]) -> None:
        """Update WeeklyStats object with API data."""
        
        # Extract offensive stats (safely with defaults)
        stats = stats_data.get('statistics', {})
        
        weekly_stats.passing_attempts = stats.get('passing_attempts', 0) or 0
        weekly_stats.passing_completions = stats.get('passing_completions', 0) or 0
        weekly_stats.passing_yards = stats.get('passing_yards', 0) or 0
        weekly_stats.passing_touchdowns = stats.get('passing_touchdowns', 0) or 0
        weekly_stats.interceptions = stats.get('interceptions', 0) or 0
        
        weekly_stats.rushing_attempts = stats.get('rushing_attempts', 0) or 0
        weekly_stats.rushing_yards = stats.get('rushing_yards', 0) or 0
        weekly_stats.rushing_touchdowns = stats.get('rushing_touchdowns', 0) or 0
        
        weekly_stats.receiving_targets = stats.get('receiving_targets', 0) or 0
        weekly_stats.receptions = stats.get('receptions', 0) or 0
        weekly_stats.receiving_yards = stats.get('receiving_yards', 0) or 0
        weekly_stats.receiving_touchdowns = stats.get('receiving_touchdowns', 0) or 0
        
        weekly_stats.fumbles = stats.get('fumbles', 0) or 0
        weekly_stats.fumbles_lost = stats.get('fumbles_lost', 0) or 0
        
        # Calculate fantasy points
        weekly_stats.fantasy_points_standard = self._calculate_fantasy_points(weekly_stats, 'standard')
        weekly_stats.fantasy_points_ppr = self._calculate_fantasy_points(weekly_stats, 'ppr')
        weekly_stats.fantasy_points_half_ppr = self._calculate_fantasy_points(weekly_stats, 'half_ppr')
        
        # Game context
        game_info = stats_data.get('game', {})
        weekly_stats.opponent_team = game_info.get('opponent')
        weekly_stats.is_home = game_info.get('is_home', False)
        weekly_stats.game_date = game_info.get('date')
        weekly_stats.game_id = game_info.get('id')

    def _calculate_fantasy_points(self, stats: WeeklyStats, scoring_type: str) -> float:
        """Calculate fantasy points based on scoring system."""
        
        points = 0.0
        
        # Passing (1 point per 25 yards, 4 points per TD, -2 per INT)
        points += (stats.passing_yards or 0) * 0.04
        points += (stats.passing_touchdowns or 0) * 4
        points -= (stats.interceptions or 0) * 2
        
        # Rushing (1 point per 10 yards, 6 points per TD)
        points += (stats.rushing_yards or 0) * 0.1
        points += (stats.rushing_touchdowns or 0) * 6
        
        # Receiving (1 point per 10 yards, 6 points per TD)
        points += (stats.receiving_yards or 0) * 0.1
        points += (stats.receiving_touchdowns or 0) * 6
        
        # PPR bonuses
        if scoring_type == 'ppr':
            points += (stats.receptions or 0) * 1.0
        elif scoring_type == 'half_ppr':
            points += (stats.receptions or 0) * 0.5
        
        # Fumbles (-2 points per lost fumble)
        points -= (stats.fumbles_lost or 0) * 2
        
        return round(points, 2)

    async def _collect_team_info(self, task: CollectionTask) -> bool:
        """Collect team information."""
        try:
            team_data = await self.nfl_client.get_team_info(task.team_id)
            
            if not team_data:
                return False
            
            async with get_db_session() as session:
                team = session.query(Team).get(task.team_id)
                if team:
                    # Update team info
                    team.coach = team_data.get('coach')
                    team.stadium = team_data.get('stadium')
                    team.updated_at = datetime.now(timezone.utc)
                    
                    task.records_collected = 1
                    task.api_calls_made = 1
                    session.commit()
            
            self.stats['api_calls_made'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error collecting team info: {e}")
            return False

    async def _collect_player_info(self, task: CollectionTask) -> bool:
        """Collect detailed player information."""
        try:
            player_data = await self.nfl_client.get_player_info(task.player_id)
            
            if not player_data:
                return False
            
            async with get_db_session() as session:
                player = session.query(Player).get(task.player_id)
                if player:
                    # Update player info
                    player.age = player_data.get('age')
                    player.height = player_data.get('height')
                    player.weight = player_data.get('weight')
                    player.college = player_data.get('college')
                    player.experience = player_data.get('experience')
                    player.updated_at = datetime.now(timezone.utc)
                    
                    # Recalculate fantasy priority
                    stats_summary = await self._get_player_stats_summary(session, player.id)
                    player.fantasy_priority_score = calculate_fantasy_priority_score(
                        player.position, stats_summary
                    )
                    
                    task.records_collected = 1
                    task.api_calls_made = 1
                    session.commit()
            
            self.stats['api_calls_made'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error collecting player info: {e}")
            return False

    async def _get_player_stats_summary(self, session: Session, player_id: int) -> Dict[str, Any]:
        """Get summary stats for fantasy priority calculation."""
        
        recent_stats = session.query(WeeklyStats).filter(
            WeeklyStats.player_id == player_id,
            WeeklyStats.season == max(self.config.target_seasons)
        ).all()
        
        if not recent_stats:
            return {}
        
        total_fantasy_points = sum(s.fantasy_points_ppr or 0 for s in recent_stats)
        games_played = len([s for s in recent_stats if (s.fantasy_points_ppr or 0) > 0])
        
        return {
            'fantasy_points_ppr': total_fantasy_points,
            'games_played': games_played
        }

    async def _schedule_retry(self, task: CollectionTask) -> None:
        """Schedule task retry with exponential backoff."""
        
        task.retry_count += 1
        
        # Calculate exponential backoff delay
        delay_seconds = min(
            self.config.retry_exponential_base ** task.retry_count * 60,
            self.config.max_retry_delay
        )
        
        task.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        task.status = CollectionStatus.PENDING.value
        task.started_at = None
        
        logger.info(f"Scheduled retry {task.retry_count} for task {task.id} "
                   f"in {delay_seconds:.0f} seconds")

    async def _update_collection_progress(self) -> None:
        """Update overall collection progress metrics."""
        
        async with get_db_session() as session:
            progress = session.query(CollectionProgress).first()
            if not progress:
                return
            
            # Count completed items
            completed_tasks = session.query(CollectionTask).filter(
                CollectionTask.status == CollectionStatus.COMPLETED.value
            ).count()
            
            completed_players = session.query(WeeklyStats.player_id).distinct().count()
            
            completed_weeks = session.query(WeeklyStats).count()
            
            # Update progress
            progress.players_completed = completed_players
            progress.weeks_completed = completed_weeks
            progress.total_api_calls = self.stats['api_calls_made']
            
            # Calculate estimated completion
            if completed_tasks > 0 and self.stats['start_time']:
                elapsed = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
                avg_time_per_task = elapsed / completed_tasks
                remaining_tasks = session.query(CollectionTask).filter(
                    CollectionTask.status.in_([
                        CollectionStatus.PENDING.value,
                        CollectionStatus.IN_PROGRESS.value
                    ])
                ).count()
                
                if remaining_tasks > 0:
                    eta_seconds = remaining_tasks * avg_time_per_task
                    progress.estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
                    progress.avg_collection_time = avg_time_per_task
            
            session.commit()
            
            # Log progress
            if completed_tasks % 50 == 0:  # Log every 50 tasks
                logger.info(f"Collection Progress: {completed_players}/{progress.total_players} players, "
                           f"{completed_weeks}/{progress.total_weeks} weeks, "
                           f"{self.stats['api_calls_made']} API calls made")

    async def stop_collection(self) -> None:
        """Gracefully stop the collection process."""
        logger.info("Stopping data collection orchestrator")
        self.is_running = False
        
        # Wait for current tasks to complete
        if self.current_tasks:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete")
            await asyncio.gather(*self.current_tasks.values(), return_exceptions=True)
        
        logger.info("Data collection orchestrator stopped")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'active_tasks': len(self.current_tasks),
            'rate_limit_status': self.rate_limiter.get_status()
        }

# Utility function for external use
async def start_intelligent_collection(config: CollectionConfig = None) -> CollectionOrchestrator:
    """Start intelligent data collection with default or custom configuration."""
    
    orchestrator = CollectionOrchestrator(config)
    
    # Start collection in background task
    collection_task = asyncio.create_task(orchestrator.start_collection())
    
    # Give it a moment to initialize
    await asyncio.sleep(1)
    
    return orchestrator