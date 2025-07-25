"""
Integrated ETL Pipeline with Intelligent Orchestration and Quality Management.
Location: src/fantasy_ai/core/data/etl.py
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .orchestrator import CollectionOrchestrator, CollectionConfig
from .sources.nfl_comprehensive import NFLAPIClient, create_nfl_client, sync_teams_to_database
from .priority_queue import IntelligentPriorityQueue, create_intelligent_queue
from .quality.anomaly_detector import DataQualityValidator, batch_validate_players
from .storage.models import (
    Team, Player, WeeklyStats, CollectionTask, CollectionStatus, 
    CollectionProgress, DataQualityMetric, calculate_fantasy_priority_score
)
# FIXED: Use simple database instead of complex one with proper error handling
from .storage.simple_database import get_simple_session

logger = logging.getLogger(__name__)

class ETLPhase(Enum):
    """ETL Pipeline phases."""
    INITIALIZATION = "initialization"
    EXTRACTION = "extraction"
    TRANSFORMATION = "transformation"
    LOADING = "loading"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"

class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ETLMetrics:
    """ETL pipeline execution metrics."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_players_processed: int = 0
    total_stats_collected: int = 0
    total_api_calls: int = 0
    validation_score: float = 0.0
    errors_encountered: int = 0
    current_phase: ETLPhase = ETLPhase.INITIALIZATION
    status: PipelineStatus = PipelineStatus.NOT_STARTED

class FantasyFootballETL:
    """
    Comprehensive ETL pipeline for Fantasy Football data with intelligent orchestration,
    quality management, and adaptive optimization.
    """
    
    def __init__(self, config: CollectionConfig = None):
        """Initialize ETL pipeline with configuration."""
        
        self.config = config or CollectionConfig()
        
        # Core components
        self.orchestrator: Optional[CollectionOrchestrator] = None
        self.nfl_client: Optional[NFLAPIClient] = None
        self.priority_queue: Optional[IntelligentPriorityQueue] = None
        self.quality_validator: Optional[DataQualityValidator] = None
        
        # Pipeline state
        self.metrics = ETLMetrics(start_time=datetime.now(timezone.utc))
        self.is_running = False
        self.current_batch = []
        
        # Configuration
        self.batch_size = 20
        self.validation_interval = 50  # Validate every 50 processed items
        self.checkpoint_interval = 100  # Save progress every 100 items
        
        # Progress tracking
        self.processed_players = set()
        self.failed_players = set()
        self.quality_issues = []

    async def initialize_pipeline(self) -> bool:
        """Initialize all pipeline components with proper error handling."""
        
        logger.info("Initializing Fantasy Football ETL Pipeline")
        self.metrics.current_phase = ETLPhase.INITIALIZATION
        
        try:
            # FIXED: Test database connection first with proper error handling
            await self._test_database_connection()
            logger.info("Database connection validated")
            
            # Initialize NFL API client
            self.nfl_client = await create_nfl_client()
            logger.info("NFL API client initialized")
            
            # Initialize data quality validator
            self.quality_validator = DataQualityValidator()
            logger.info("Data quality validator initialized")
            
            # Initialize priority queue with error handling
            try:
                self.priority_queue = await create_intelligent_queue()
                logger.info("Intelligent priority queue initialized")
            except Exception as queue_error:
                logger.error(f"Priority queue initialization failed: {queue_error}")
                # Continue without priority queue - use simple processing
                logger.info("Continuing without priority queue - using simple processing")
            
            # Initialize orchestrator
            self.orchestrator = CollectionOrchestrator(self.config)
            logger.info("Collection orchestrator initialized")
            
            # Sync teams to database
            teams_synced = await sync_teams_to_database(self.nfl_client)
            logger.info(f"Synced {teams_synced} teams to database")
            
            # Initialize players
            await self._initialize_players()
            
            self.metrics.status = PipelineStatus.RUNNING
            logger.info("ETL Pipeline initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ETL pipeline: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            self.metrics.status = PipelineStatus.FAILED
            return False

    async def _test_database_connection(self) -> None:
        """Test database connection and handle configuration issues."""
        
        try:
            async with get_simple_session() as session:
                # Simple test query
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                test_value = result.scalar()
                
                if test_value != 1:
                    raise RuntimeError("Database connection test failed")
                    
                logger.info("Database connection test successful")
                
        except Exception as db_error:
            logger.error(f"Database connection failed: {db_error}")
            
            # Check if it's the pool_size error we're trying to fix
            if "pool_size" in str(db_error) or "max_overflow" in str(db_error):
                logger.error("SQLite pooling configuration error detected")
                logger.error("Please check your simple_database.py configuration")
                logger.error("SQLite databases should not use pool_size or max_overflow parameters")
            
            raise RuntimeError(f"Database initialization failed: {db_error}")

    async def _initialize_players(self) -> None:
        """Initialize player data for priority positions with better error handling."""
        
        logger.info("Initializing player data for priority positions")
        
        try:
            async with get_simple_session() as session:
                from sqlalchemy import select
                
                # Get all teams
                result = await session.execute(select(Team))
                teams = result.scalars().all()
                
                if not teams:
                    logger.warning("No teams found in database - may need to sync teams first")
                    return
                
                total_players = 0
                
                for team in teams:
                    try:
                        # Get players for recent season (2023)
                        players_data = await self.nfl_client.get_players_by_team(team.api_id, 2023)
                        
                        for player_data in players_data:
                            player_info = player_data.get('player', {})
                            
                            # Only process priority positions
                            position = player_info.get('position', 'OTHER')
                            if position not in self.config.priority_positions:
                                continue
                            
                            # Check if player exists
                            existing_result = await session.execute(
                                select(Player).where(Player.api_id == player_info.get('id'))
                            )
                            existing_player = existing_result.scalar_one_or_none()
                            
                            if existing_player:
                                # Update existing player
                                existing_player.name = player_info.get('name', '')
                                existing_player.position = position
                                existing_player.team_id = team.id
                                existing_player.updated_at = datetime.now(timezone.utc)
                            else:
                                # Create new player
                                new_player = Player(
                                    api_id=player_info.get('id'),
                                    team_id=team.id,
                                    name=player_info.get('name', ''),
                                    firstname=player_info.get('firstname', ''),
                                    lastname=player_info.get('lastname', ''),
                                    position=position,
                                    number=player_info.get('number'),
                                    age=player_info.get('age'),
                                    height=player_info.get('height'),
                                    weight=player_info.get('weight'),
                                    college=player_info.get('college'),
                                    collection_priority=self._calculate_collection_priority(position),
                                    fantasy_priority_score=0.5  # Will be updated later
                                )
                                session.add(new_player)
                            
                            total_players += 1
                        
                        # Brief pause between teams
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error initializing players for team {team.name}: {e}")
                        # Continue with next team
                        continue
                
                await session.commit()
                logger.info(f"Initialized {total_players} priority position players")
                
        except Exception as e:
            logger.error(f"Player initialization failed: {e}")
            # Don't fail the entire pipeline for player initialization issues
            logger.warning("Continuing pipeline without player initialization")

    def _calculate_collection_priority(self, position: str) -> int:
        """Calculate collection priority for player position."""
        
        priority_map = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 4,
            'K': 8,
            'DEF': 9
        }
        
        return priority_map.get(position, 10)

    async def run_full_pipeline(self) -> ETLMetrics:
        """Execute complete ETL pipeline with comprehensive error handling."""
        
        logger.info("Starting full ETL pipeline execution")
        self.is_running = True
        
        try:
            # Phase 1: Initialization
            if not await self.initialize_pipeline():
                raise RuntimeError("Pipeline initialization failed")
            
            # Phase 2: Extraction (via orchestrator)
            await self.run_extraction_phase()
            
            # Phase 3: Transformation & Loading (integrated)
            await self.run_transformation_phase()
            
            # Phase 4: Validation
            await self.run_validation_phase()
            
            # Phase 5: Optimization
            await self.run_optimization_phase()
            
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.status = PipelineStatus.COMPLETED
            
            logger.info("ETL Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            logger.error(f"Error occurred in phase: {self.metrics.current_phase.value}")
            self.metrics.status = PipelineStatus.FAILED
            self.metrics.errors_encountered += 1
        
        finally:
            self.is_running = False
            await self.cleanup()
        
        return self.metrics

    async def run_extraction_phase(self) -> None:
        """Execute data extraction phase using intelligent orchestration."""
        
        logger.info("Starting extraction phase")
        self.metrics.current_phase = ETLPhase.EXTRACTION
        
        try:
            # Start orchestrator in background
            orchestrator_task = asyncio.create_task(
                self.orchestrator.start_collection()
            )
            
            # Monitor progress
            await self._monitor_extraction_progress(orchestrator_task)
            
        except Exception as e:
            logger.error(f"Extraction phase failed: {e}")
            raise

    async def _monitor_extraction_progress(self, orchestrator_task: asyncio.Task) -> None:
        """Monitor extraction progress and handle completion."""
        
        logger.info("Monitoring extraction progress")
        
        try:
            while not orchestrator_task.done():
                # Get orchestrator stats
                stats = self.orchestrator.get_collection_stats()
                
                self.metrics.total_api_calls = stats.get('api_calls_made', 0)
                
                # Log progress periodically
                if self.metrics.total_api_calls % 50 == 0 and self.metrics.total_api_calls > 0:
                    logger.info(f"Extraction progress: {self.metrics.total_api_calls} API calls made")
                
                # Check if we should stop (rate limit reached or sufficient data)
                try:
                    async with get_simple_session() as session:
                        from sqlalchemy import select, func
                        
                        result = await session.execute(select(func.count()).select_from(WeeklyStats))
                        total_stats = result.scalar()
                        
                        if total_stats >= 5000:  # Sufficient data threshold
                            logger.info(f"Sufficient data collected ({total_stats} stats), stopping extraction")
                            await self.orchestrator.stop_collection()
                            break
                            
                except Exception as db_check_error:
                    logger.warning(f"Error checking database stats: {db_check_error}")
                    # Continue monitoring
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Wait for orchestrator to complete
            try:
                await orchestrator_task
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                
        except Exception as e:
            logger.error(f"Error monitoring extraction progress: {e}")

    async def run_transformation_phase(self) -> None:
        """Execute data transformation and enhancement phase."""
        
        logger.info("Starting transformation phase")
        self.metrics.current_phase = ETLPhase.TRANSFORMATION
        
        try:
            async with get_simple_session() as session:
                from sqlalchemy import select
                
                # Get all players with collected stats
                result = await session.execute(
                    select(Player).join(WeeklyStats).distinct()
                )
                players_with_stats = result.scalars().all()
                
                total_players = len(players_with_stats)
                processed = 0
                
                logger.info(f"Found {total_players} players with stats to transform")
                
                for player in players_with_stats:
                    try:
                        # Transform and enhance player data
                        await self._transform_player_data(session, player)
                        
                        processed += 1
                        self.processed_players.add(player.id)
                        
                        # Periodic progress logging
                        if processed % 50 == 0:
                            logger.info(f"Transformation progress: {processed}/{total_players} players")
                        
                        # Checkpoint saving
                        if processed % self.checkpoint_interval == 0:
                            await session.commit()
                            logger.info(f"Checkpoint saved at {processed} players")
                    
                    except Exception as e:
                        logger.error(f"Error transforming player {player.id}: {e}")
                        self.failed_players.add(player.id)
                        self.metrics.errors_encountered += 1
                        # Continue with next player
                        continue
                
                await session.commit()
                self.metrics.total_players_processed = processed
                
                logger.info(f"Transformation phase completed: {processed} players processed")
                
        except Exception as e:
            logger.error(f"Transformation phase failed: {e}")
            raise

    async def _transform_player_data(self, session, player: Player) -> None:
        """Transform and enhance individual player data."""
        
        from sqlalchemy import select
        
        # Get player's stats
        result = await session.execute(
            select(WeeklyStats).where(WeeklyStats.player_id == player.id)
        )
        stats = result.scalars().all()
        
        if not stats:
            return
        
        # Calculate fantasy priority score
        stats_summary = self._calculate_stats_summary(stats)
        player.fantasy_priority_score = calculate_fantasy_priority_score(
            player.position, stats_summary
        )
        
        # Update last stats update
        latest_stat = max(stats, key=lambda s: (s.season, s.week))
        player.last_stats_update = datetime.now(timezone.utc)
        
        # Calculate enhanced metrics for each stat
        for stat in stats:
            if stat.fantasy_points_ppr is None:
                stat.fantasy_points_ppr = self._calculate_fantasy_points(stat, 'ppr')
            if stat.fantasy_points_standard is None:
                stat.fantasy_points_standard = self._calculate_fantasy_points(stat, 'standard')
            if stat.fantasy_points_half_ppr is None:
                stat.fantasy_points_half_ppr = self._calculate_fantasy_points(stat, 'half_ppr')
        
        self.metrics.total_stats_collected += len(stats)

    def _calculate_stats_summary(self, stats: List[WeeklyStats]) -> Dict[str, Any]:
        """Calculate summary statistics for fantasy priority calculation."""
        
        if not stats:
            return {}
        
        total_fantasy_points = sum(s.fantasy_points_ppr or 0 for s in stats)
        games_with_points = len([s for s in stats if (s.fantasy_points_ppr or 0) > 0])
        
        return {
            'fantasy_points_ppr': total_fantasy_points,
            'games_played': games_with_points,
            'avg_fantasy_points': total_fantasy_points / max(games_with_points, 1),
            'total_games': len(stats)
        }

    def _calculate_fantasy_points(self, stat: WeeklyStats, scoring_type: str) -> float:
        """Calculate fantasy points for a stat entry."""
        
        points = 0.0
        
        # Passing (1 point per 25 yards, 4 points per TD, -2 per INT)
        points += (stat.passing_yards or 0) * 0.04
        points += (stat.passing_touchdowns or 0) * 4
        points -= (stat.interceptions or 0) * 2
        
        # Rushing (1 point per 10 yards, 6 points per TD)
        points += (stat.rushing_yards or 0) * 0.1
        points += (stat.rushing_touchdowns or 0) * 6
        
        # Receiving (1 point per 10 yards, 6 points per TD)
        points += (stat.receiving_yards or 0) * 0.1
        points += (stat.receiving_touchdowns or 0) * 6
        
        # PPR bonuses
        if scoring_type == 'ppr':
            points += (stat.receptions or 0) * 1.0
        elif scoring_type == 'half_ppr':
            points += (stat.receptions or 0) * 0.5
        
        # Fumbles (-2 points per lost fumble)
        points -= (stat.fumbles_lost or 0) * 2
        
        return round(points, 2)

    async def run_validation_phase(self) -> None:
        """Execute comprehensive data validation phase."""
        
        logger.info("Starting validation phase")
        self.metrics.current_phase = ETLPhase.VALIDATION
        
        try:
            async with get_simple_session() as session:
                from sqlalchemy import select
                
                # Get all players to validate
                result = await session.execute(
                    select(Player).where(Player.position.in_(self.config.priority_positions))
                )
                players = result.scalars().all()
                
                total_players = len(players)
                validated = 0
                total_quality_score = 0.0
                
                logger.info(f"Validating {total_players} players")
                
                # Batch validation for efficiency
                batch_size = 10
                
                for i in range(0, total_players, batch_size):
                    batch = players[i:i + batch_size]
                    player_ids = [p.id for p in batch]
                    
                    try:
                        # Validate batch
                        validation_results = await batch_validate_players(
                            player_ids, 2023  # Validate most recent season
                        )
                        
                        for player_id, quality_metrics in validation_results.items():
                            total_quality_score += quality_metrics.overall_score
                            validated += 1
                            
                            # Track quality issues
                            if quality_metrics.overall_score < 0.7:
                                self.quality_issues.append({
                                    'player_id': player_id,
                                    'quality_score': quality_metrics.overall_score,
                                    'anomalies': len(quality_metrics.anomalies)
                                })
                        
                        if validated % 100 == 0:
                            logger.info(f"Validation progress: {validated}/{total_players} players")
                        
                    except Exception as e:
                        logger.error(f"Error validating player batch: {e}")
                        self.metrics.errors_encountered += 1
                        # Continue with next batch
                        continue
                    
                    # Brief pause between batches
                    await asyncio.sleep(1)
                
                # Calculate overall validation score
                if validated > 0:
                    self.metrics.validation_score = total_quality_score / validated
                
                logger.info(f"Validation completed: {validated} players validated, "
                           f"overall quality score: {self.metrics.validation_score:.3f}")
                           
        except Exception as e:
            logger.error(f"Validation phase failed: {e}")
            # Don't fail the entire pipeline for validation issues
            logger.warning("Continuing pipeline without validation")

    async def run_optimization_phase(self) -> None:
        """Execute optimization and finalization phase."""
        
        logger.info("Starting optimization phase")
        self.metrics.current_phase = ETLPhase.OPTIMIZATION
        
        try:
            async with get_simple_session() as session:
                # Update collection priorities based on results
                await self._optimize_collection_priorities(session)
                
                # Generate final statistics
                await self._generate_final_statistics(session)
                
                # Clean up failed tasks
                await self._cleanup_failed_tasks(session)
                
                await session.commit()
            
            logger.info("Optimization phase completed")
            
        except Exception as e:
            logger.error(f"Optimization phase failed: {e}")
            # Don't fail the entire pipeline
            logger.warning("Pipeline completed with optimization errors")

    async def _optimize_collection_priorities(self, session) -> None:
        """Optimize collection priorities based on current data."""
        
        try:
            from sqlalchemy import select
            
            result = await session.execute(
                select(Player).where(Player.position.in_(self.config.priority_positions))
            )
            players = result.scalars().all()
            
            for player in players:
                # Get latest quality metrics
                quality_result = await session.execute(
                    select(DataQualityMetric).where(
                        DataQualityMetric.player_id == player.id,
                        DataQualityMetric.season == 2023
                    )
                )
                quality_metric = quality_result.scalar_one_or_none()
                
                if quality_metric:
                    # Adjust priority based on data quality
                    if quality_metric.overall_quality_score < 0.5:
                        player.collection_priority = max(1, player.collection_priority - 1)
                    elif quality_metric.overall_quality_score > 0.9:
                        player.collection_priority = min(10, player.collection_priority + 1)
                        
        except Exception as e:
            logger.error(f"Error optimizing collection priorities: {e}")

    async def _generate_final_statistics(self, session) -> None:
        """Generate final pipeline statistics."""
        
        try:
            from sqlalchemy import select
            
            # Update collection progress
            result = await session.execute(select(CollectionProgress))
            progress = result.scalar_one_or_none()
            
            if progress:
                progress.players_completed = len(self.processed_players)
                progress.total_api_calls = self.metrics.total_api_calls
                progress.last_updated = datetime.now(timezone.utc)
                
        except Exception as e:
            logger.error(f"Error generating final statistics: {e}")

    async def _cleanup_failed_tasks(self, session) -> None:
        """Clean up failed collection tasks."""
        
        try:
            from sqlalchemy import select
            
            # Mark old failed tasks for retry
            result = await session.execute(
                select(CollectionTask).where(
                    CollectionTask.status == CollectionStatus.FAILED.value,
                    CollectionTask.updated_at < datetime.now(timezone.utc) - timedelta(hours=24)
                )
            )
            old_failed_tasks = result.scalars().all()
            
            for task in old_failed_tasks:
                if task.retry_count < task.max_retries:
                    task.status = CollectionStatus.PENDING.value
                    task.retry_count += 1
                    task.scheduled_at = datetime.now(timezone.utc) + timedelta(hours=1)
                    
        except Exception as e:
            logger.error(f"Error cleaning up failed tasks: {e}")

    async def pause_pipeline(self) -> None:
        """Pause pipeline execution."""
        
        logger.info("Pausing ETL pipeline")
        self.metrics.status = PipelineStatus.PAUSED
        
        if self.orchestrator:
            await self.orchestrator.stop_collection()

    async def resume_pipeline(self) -> None:
        """Resume pipeline execution."""
        
        logger.info("Resuming ETL pipeline")
        self.metrics.status = PipelineStatus.RUNNING
        
        if self.orchestrator:
            orchestrator_task = asyncio.create_task(
                self.orchestrator.start_collection()
            )
            await self._monitor_extraction_progress(orchestrator_task)

    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        
        logger.info("Cleaning up ETL pipeline resources")
        
        try:
            if self.nfl_client:
                await self.nfl_client.close()
            
            if self.orchestrator:
                await self.orchestrator.stop_collection()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        
        status = {
            'metrics': {
                'status': self.metrics.status.value,
                'current_phase': self.metrics.current_phase.value,
                'start_time': self.metrics.start_time.isoformat(),
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'total_players_processed': self.metrics.total_players_processed,
                'total_stats_collected': self.metrics.total_stats_collected,
                'total_api_calls': self.metrics.total_api_calls,
                'validation_score': self.metrics.validation_score,
                'errors_encountered': self.metrics.errors_encountered
            },
            'progress': {
                'processed_players': len(self.processed_players),
                'failed_players': len(self.failed_players),
                'quality_issues': len(self.quality_issues)
            },
            'is_running': self.is_running
        }
        
        # Add component status if available
        if self.orchestrator:
            try:
                status['orchestrator'] = self.orchestrator.get_collection_stats()
            except Exception as e:
                logger.error(f"Error getting orchestrator stats: {e}")
        
        if self.priority_queue:
            try:
                status['priority_queue'] = self.priority_queue.get_queue_stats()
            except Exception as e:
                logger.error(f"Error getting priority queue stats: {e}")
        
        if self.nfl_client:
            try:
                status['nfl_client'] = self.nfl_client.get_stats()
            except Exception as e:
                logger.error(f"Error getting NFL client stats: {e}")
        
        return status

    async def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report."""
        
        report = {
            'execution_summary': {
                'start_time': self.metrics.start_time.isoformat(),
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'total_duration': str(datetime.now(timezone.utc) - self.metrics.start_time),
                'status': self.metrics.status.value,
                'final_phase': self.metrics.current_phase.value
            },
            'data_collection': {
                'total_api_calls': self.metrics.total_api_calls,
                'total_players_processed': self.metrics.total_players_processed,
                'total_stats_collected': self.metrics.total_stats_collected,
                'successful_players': len(self.processed_players),
                'failed_players': len(self.failed_players)
            },
            'quality_assessment': {
                'overall_validation_score': self.metrics.validation_score,
                'quality_issues_found': len(self.quality_issues),
                'quality_issues_details': self.quality_issues
            },
            'errors_and_issues': {
                'total_errors': self.metrics.errors_encountered,
                'failed_player_ids': list(self.failed_players)
            }
        }
        
        # Add database statistics with error handling
        try:
            async with get_simple_session() as session:
                from sqlalchemy import select, func
                
                db_stats = {}
                
                # Get table counts
                for table_name, model in [
                    ('total_teams', Team),
                    ('total_players', Player), 
                    ('total_weekly_stats', WeeklyStats)
                ]:
                    try:
                        result = await session.execute(select(func.count()).select_from(model))
                        db_stats[table_name] = result.scalar()
                    except Exception as e:
                        logger.error(f"Error counting {table_name}: {e}")
                        db_stats[table_name] = 0
                
                # Get task counts
                for status_name, status_value in [
                    ('completed_tasks', CollectionStatus.COMPLETED.value),
                    ('pending_tasks', CollectionStatus.PENDING.value)
                ]:
                    try:
                        result = await session.execute(
                            select(func.count()).select_from(CollectionTask).where(
                                CollectionTask.status == status_value
                            )
                        )
                        db_stats[status_name] = result.scalar()
                    except Exception as e:
                        logger.error(f"Error counting {status_name}: {e}")
                        db_stats[status_name] = 0
                
                report['database_state'] = db_stats
                
        except Exception as e:
            logger.error(f"Error generating database statistics: {e}")
            report['database_state'] = {'error': str(e)}
        
        return report

# Utility functions with better error handling
async def run_quick_etl(max_api_calls: int = 50) -> ETLMetrics:
    """Run a quick ETL pipeline for testing or limited data collection."""
    
    config = CollectionConfig(
        api_calls_per_day=max_api_calls,
        priority_positions=['QB', 'RB'],  # Limited positions
        target_seasons=[2023],  # Recent season only
        max_concurrent_tasks=2
    )
    
    etl = FantasyFootballETL(config)
    return await etl.run_full_pipeline()

async def run_comprehensive_etl() -> ETLMetrics:
    """Run comprehensive ETL pipeline with all data."""
    
    config = CollectionConfig(
        api_calls_per_day=100,
        priority_positions=['QB', 'RB', 'WR', 'TE'],
        target_seasons=[2021, 2022, 2023],
        max_concurrent_tasks=3,
        enable_quality_validation=True
    )
    
    etl = FantasyFootballETL(config)
    return await etl.run_full_pipeline()

async def resume_etl_from_checkpoint() -> ETLMetrics:
    """Resume ETL pipeline from last checkpoint."""
    
    # Load configuration from database or config file
    config = CollectionConfig()
    
    etl = FantasyFootballETL(config)
    
    try:
        # Initialize without full data collection
        if not await etl.initialize_pipeline():
            raise RuntimeError("Failed to initialize pipeline for resume")
        
        # Resume from transformation phase
        await etl.run_transformation_phase()
        await etl.run_validation_phase()
        await etl.run_optimization_phase()
        
    except Exception as e:
        logger.error(f"Error resuming ETL from checkpoint: {e}")
        etl.metrics.status = PipelineStatus.FAILED
    
    return etl.metrics