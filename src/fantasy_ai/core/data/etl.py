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
from .storage.database import get_db_session, ensure_database_exists

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
        """Initialize all pipeline components."""
        
        logger.info("Initializing Fantasy Football ETL Pipeline")
        self.metrics.current_phase = ETLPhase.INITIALIZATION
        
        try:
            # Ensure database exists
            await ensure_database_exists()
            
            # Initialize NFL API client
            self.nfl_client = await create_nfl_client()
            logger.info("NFL API client initialized")
            
            # Initialize data quality validator
            self.quality_validator = DataQualityValidator()
            logger.info("Data quality validator initialized")
            
            # Initialize priority queue
            self.priority_queue = await create_intelligent_queue()
            logger.info("Intelligent priority queue initialized")
            
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
            self.metrics.status = PipelineStatus.FAILED
            return False

    async def _initialize_players(self) -> None:
        """Initialize player data for priority positions."""
        
        logger.info("Initializing player data for priority positions")
        
        async with get_db_session() as session:
            # Get all teams
            teams = session.query(Team).all()
            
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
                        existing_player = session.query(Player).filter(
                            Player.api_id == player_info.get('id')
                        ).first()
                        
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
            
            session.commit()
            logger.info(f"Initialized {total_players} priority position players")

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
        """Execute complete ETL pipeline."""
        
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
        
        # Start orchestrator in background
        orchestrator_task = asyncio.create_task(
            self.orchestrator.start_collection()
        )
        
        # Monitor progress
        await self._monitor_extraction_progress(orchestrator_task)

    async def _monitor_extraction_progress(self, orchestrator_task: asyncio.Task) -> None:
        """Monitor extraction progress and handle completion."""
        
        logger.info("Monitoring extraction progress")
        
        while not orchestrator_task.done():
            # Get orchestrator stats
            stats = self.orchestrator.get_collection_stats()
            
            self.metrics.total_api_calls = stats.get('api_calls_made', 0)
            
            # Log progress periodically
            if self.metrics.total_api_calls % 50 == 0 and self.metrics.total_api_calls > 0:
                logger.info(f"Extraction progress: {self.metrics.total_api_calls} API calls made")
            
            # Check if we should stop (rate limit reached or sufficient data)
            async with get_db_session() as session:
                total_stats = session.query(WeeklyStats).count()
                
                if total_stats >= 5000:  # Sufficient data threshold
                    logger.info(f"Sufficient data collected ({total_stats} stats), stopping extraction")
                    await self.orchestrator.stop_collection()
                    break
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Wait for orchestrator to complete
        try:
            await orchestrator_task
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")

    async def run_transformation_phase(self) -> None:
        """Execute data transformation and enhancement phase."""
        
        logger.info("Starting transformation phase")
        self.metrics.current_phase = ETLPhase.TRANSFORMATION
        
        async with get_db_session() as session:
            # Get all players with collected stats
            players_with_stats = session.query(Player).join(WeeklyStats).distinct().all()
            
            total_players = len(players_with_stats)
            processed = 0
            
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
                        session.commit()
                        logger.info(f"Checkpoint saved at {processed} players")
                
                except Exception as e:
                    logger.error(f"Error transforming player {player.id}: {e}")
                    self.failed_players.add(player.id)
                    self.metrics.errors_encountered += 1
            
            session.commit()
            self.metrics.total_players_processed = processed
            
            logger.info(f"Transformation phase completed: {processed} players processed")

    async def _transform_player_data(self, session: Session, player: Player) -> None:
        """Transform and enhance individual player data."""
        
        # Get player's stats
        stats = session.query(WeeklyStats).filter(
            WeeklyStats.player_id == player.id
        ).all()
        
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
        
        async with get_db_session() as session:
            # Get all players to validate
            players = session.query(Player).filter(
                Player.position.in_(self.config.priority_positions)
            ).all()
            
            total_players = len(players)
            validated = 0
            total_quality_score = 0.0
            
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
                    
                    logger.info(f"Validation progress: {validated}/{total_players} players")
                    
                except Exception as e:
                    logger.error(f"Error validating player batch: {e}")
                    self.metrics.errors_encountered += 1
                
                # Brief pause between batches
                await asyncio.sleep(1)
            
            # Calculate overall validation score
            if validated > 0:
                self.metrics.validation_score = total_quality_score / validated
            
            logger.info(f"Validation completed: {validated} players validated, "
                       f"overall quality score: {self.metrics.validation_score:.3f}")

    async def run_optimization_phase(self) -> None:
        """Execute optimization and finalization phase."""
        
        logger.info("Starting optimization phase")
        self.metrics.current_phase = ETLPhase.OPTIMIZATION
        
        async with get_db_session() as session:
            # Update collection priorities based on results
            await self._optimize_collection_priorities(session)
            
            # Generate final statistics
            await self._generate_final_statistics(session)
            
            # Clean up failed tasks
            await self._cleanup_failed_tasks(session)
            
            session.commit()
        
        logger.info("Optimization phase completed")

    async def _optimize_collection_priorities(self, session: Session) -> None:
        """Optimize collection priorities based on current data."""
        
        players = session.query(Player).filter(
            Player.position.in_(self.config.priority_positions)
        ).all()
        
        for player in players:
            # Get latest quality metrics
            quality_metric = session.query(DataQualityMetric).filter(
                DataQualityMetric.player_id == player.id,
                DataQualityMetric.season == 2023
            ).first()
            
            if quality_metric:
                # Adjust priority based on data quality
                if quality_metric.overall_quality_score < 0.5:
                    player.collection_priority = max(1, player.collection_priority - 1)
                elif quality_metric.overall_quality_score > 0.9:
                    player.collection_priority = min(10, player.collection_priority + 1)

    async def _generate_final_statistics(self, session: Session) -> None:
        """Generate final pipeline statistics."""
        
        # Update collection progress
        progress = session.query(CollectionProgress).first()
        if progress:
            progress.players_completed = len(self.processed_players)
            progress.total_api_calls = self.metrics.total_api_calls
            progress.last_updated = datetime.now(timezone.utc)

    async def _cleanup_failed_tasks(self, session: Session) -> None:
        """Clean up failed collection tasks."""
        
        # Mark old failed tasks for retry
        old_failed_tasks = session.query(CollectionTask).filter(
            CollectionTask.status == CollectionStatus.FAILED.value,
            CollectionTask.updated_at < datetime.now(timezone.utc) - timedelta(hours=24)
        ).all()
        
        for task in old_failed_tasks:
            if task.retry_count < task.max_retries:
                task.status = CollectionStatus.PENDING.value
                task.retry_count += 1
                task.scheduled_at = datetime.now(timezone.utc) + timedelta(hours=1)

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
        
        if self.nfl_client:
            await self.nfl_client.close()
        
        if self.orchestrator:
            await self.orchestrator.stop_collection()

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
            status['orchestrator'] = self.orchestrator.get_collection_stats()
        
        if self.priority_queue:
            status['priority_queue'] = self.priority_queue.get_queue_stats()
        
        if self.nfl_client:
            status['nfl_client'] = self.nfl_client.get_stats()
        
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
        
        # Add database statistics
        async with get_db_session() as session:
            db_stats = {
                'total_teams': session.query(Team).count(),
                'total_players': session.query(Player).count(),
                'total_weekly_stats': session.query(WeeklyStats).count(),
                'completed_tasks': session.query(CollectionTask).filter(
                    CollectionTask.status == CollectionStatus.COMPLETED.value
                ).count(),
                'pending_tasks': session.query(CollectionTask).filter(
                    CollectionTask.status == CollectionStatus.PENDING.value
                ).count()
            }
            
            report['database_state'] = db_stats
        
        return report

# Utility functions
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
    
    # Initialize without full data collection
    await etl.initialize_pipeline()
    
    # Resume from transformation phase
    await etl.run_transformation_phase()
    await etl.run_validation_phase()
    await etl.run_optimization_phase()
    
    return etl.metrics