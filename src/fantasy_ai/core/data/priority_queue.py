"""
Intelligent Priority Queue System for Data Collection Tasks.
Location: src/fantasy_ai/core/data/priority_queue.py
"""

import asyncio
import heapq
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .storage.models import (
    CollectionTask, CollectionStatus, Player, Team, WeeklyStats,
    PlayerPosition, DataQualityMetric, CollectionProgress
)
from .storage.database import get_db_session

logger = logging.getLogger(__name__)

class TaskCategory(Enum):
    """Categories for task prioritization."""
    CRITICAL_MISSING = "critical_missing"      # High-value players missing recent data
    QUALITY_ISSUES = "quality_issues"          # Players with data quality problems
    COMPLETENESS = "completeness"              # Fill gaps in existing data
    ENHANCEMENT = "enhancement"                # Additional data for complete players
    MAINTENANCE = "maintenance"                # Team info updates, etc.

class UrgencyLevel(Enum):
    """Urgency levels for task scheduling."""
    IMMEDIATE = 1    # Process ASAP
    HIGH = 2         # Process within hours
    MEDIUM = 3       # Process within day
    LOW = 4          # Process when convenient
    BACKGROUND = 5   # Process during idle time

@dataclass
class TaskPriorityScore:
    """Comprehensive priority scoring for collection tasks."""
    
    # Core priority factors (0-1 scores)
    fantasy_relevance: float = 0.0    # Based on position and historical performance
    data_freshness: float = 0.0       # How outdated the data is
    quality_impact: float = 0.0       # Impact on overall data quality
    completeness_gap: float = 0.0     # How much this fills data gaps
    
    # Contextual factors
    position_priority: float = 0.0    # QB/RB/WR/TE get higher priority
    season_recency: float = 0.0       # Recent seasons more important
    api_efficiency: float = 0.0       # Batch efficiency considerations
    
    # Calculated fields
    total_score: float = field(init=False)
    urgency: UrgencyLevel = field(init=False)
    category: TaskCategory = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        # Weighted total score
        self.total_score = (
            0.3 * self.fantasy_relevance +
            0.25 * self.data_freshness +
            0.2 * self.quality_impact +
            0.15 * self.completeness_gap +
            0.1 * (self.position_priority + self.season_recency + self.api_efficiency) / 3
        )
        
        # Determine urgency
        if self.total_score >= 0.8:
            self.urgency = UrgencyLevel.IMMEDIATE
        elif self.total_score >= 0.6:
            self.urgency = UrgencyLevel.HIGH
        elif self.total_score >= 0.4:
            self.urgency = UrgencyLevel.MEDIUM
        elif self.total_score >= 0.2:
            self.urgency = UrgencyLevel.LOW
        else:
            self.urgency = UrgencyLevel.BACKGROUND
        
        # Determine category
        if self.quality_impact > 0.7:
            self.category = TaskCategory.QUALITY_ISSUES
        elif self.fantasy_relevance > 0.7 and self.data_freshness > 0.6:
            self.category = TaskCategory.CRITICAL_MISSING
        elif self.completeness_gap > 0.6:
            self.category = TaskCategory.COMPLETENESS
        elif self.fantasy_relevance > 0.5:
            self.category = TaskCategory.ENHANCEMENT
        else:
            self.category = TaskCategory.MAINTENANCE

@dataclass
class PriorityTask:
    """Task with priority information for queue management."""
    
    task_id: int
    priority_score: TaskPriorityScore
    scheduled_time: datetime
    dependencies: Set[int] = field(default_factory=set)
    batch_group: Optional[str] = None
    retry_count: int = 0
    last_attempt: Optional[datetime] = None
    
    def __lt__(self, other):
        """Comparison for heap queue (higher priority = lower value for min-heap)."""
        # Primary: urgency level (lower number = higher priority)
        if self.priority_score.urgency != other.priority_score.urgency:
            return self.priority_score.urgency.value < other.priority_score.urgency.value
        
        # Secondary: total score (higher score = higher priority)
        if abs(self.priority_score.total_score - other.priority_score.total_score) > 0.001:
            return self.priority_score.total_score > other.priority_score.total_score
        
        # Tertiary: scheduled time (earlier = higher priority)
        return self.scheduled_time < other.scheduled_time

class IntelligentPriorityQueue:
    """
    Advanced priority queue with intelligent task scheduling, dependency management,
    and adaptive optimization based on API performance and business rules.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize priority queue with configuration."""
        
        self.max_queue_size = max_queue_size
        
        # Core queue structures
        self._priority_heap: List[PriorityTask] = []
        self._task_lookup: Dict[int, PriorityTask] = {}
        self._batch_groups: Dict[str, List[int]] = defaultdict(list)
        
        # Dependency tracking
        self._dependencies: Dict[int, Set[int]] = defaultdict(set)
        self._dependents: Dict[int, Set[int]] = defaultdict(set)
        
        # Performance analytics
        self._completion_history: deque = deque(maxlen=100)
        self._category_performance: Dict[TaskCategory, Dict[str, float]] = defaultdict(
            lambda: {'avg_time': 0.0, 'success_rate': 1.0, 'count': 0}
        )
        
        # Queue statistics
        self.stats = {
            'tasks_queued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_wait_time': 0.0,
            'queue_size_history': deque(maxlen=24)  # 24 hours of hourly snapshots
        }
        
        # Configuration
        self.position_weights = {
            'QB': 1.0,
            'RB': 0.95,
            'WR': 0.90,
            'TE': 0.85,
            'K': 0.3,
            'DEF': 0.2
        }
        
        self.season_weights = {
            2023: 1.0,
            2022: 0.8,
            2021: 0.6
        }

    async def analyze_and_queue_tasks(self, limit: Optional[int] = None) -> int:
        """
        Analyze database state and intelligently queue high-priority collection tasks.
        """
        
        logger.info("Analyzing database state for intelligent task prioritization")
        
        async with get_db_session() as session:
            # Get comprehensive data for analysis
            analysis_data = await self._gather_analysis_data(session)
            
            # Generate priority tasks
            priority_tasks = await self._generate_priority_tasks(session, analysis_data)
            
            # Optimize and queue tasks
            queued_count = await self._optimize_and_queue(session, priority_tasks, limit)
            
            logger.info(f"Queued {queued_count} intelligently prioritized tasks")
            return queued_count

    async def _gather_analysis_data(self, session: Session) -> Dict[str, Any]:
        """Gather comprehensive data for intelligent task prioritization."""
        
        # Priority players (QB, RB, WR, TE)
        priority_positions = ['QB', 'RB', 'WR', 'TE']
        
        priority_players = session.query(Player).filter(
            Player.position.in_(priority_positions),
            Player.is_active == True
        ).all()
        
        # Current data coverage analysis
        data_coverage = {}
        quality_issues = {}
        
        for player in priority_players:
            # Analyze data coverage by season
            coverage_by_season = {}
            quality_by_season = {}
            
            for season in [2021, 2022, 2023]:
                # Get existing weeks
                existing_weeks = session.query(WeeklyStats.week).filter(
                    WeeklyStats.player_id == player.id,
                    WeeklyStats.season == season
                ).all()
                
                existing_week_numbers = {week[0] for week in existing_weeks}
                expected_weeks = set(range(1, 18))  # Weeks 1-17
                missing_weeks = expected_weeks - existing_week_numbers
                
                coverage_by_season[season] = {
                    'existing_weeks': len(existing_week_numbers),
                    'missing_weeks': list(missing_weeks),
                    'coverage_ratio': len(existing_week_numbers) / 17
                }
                
                # Get quality metrics
                quality_metric = session.query(DataQualityMetric).filter(
                    DataQualityMetric.player_id == player.id,
                    DataQualityMetric.season == season
                ).first()
                
                if quality_metric:
                    quality_by_season[season] = {
                        'overall_score': quality_metric.overall_quality_score,
                        'anomaly_score': quality_metric.anomaly_score,
                        'completeness_score': quality_metric.completeness_score,
                        'needs_validation': quality_metric.overall_quality_score < 0.7
                    }
                else:
                    quality_by_season[season] = {
                        'overall_score': 0.0,
                        'anomaly_score': 1.0,
                        'completeness_score': 0.0,
                        'needs_validation': True
                    }
            
            data_coverage[player.id] = coverage_by_season
            quality_issues[player.id] = quality_by_season
        
        # API and performance context
        recent_tasks = session.query(CollectionTask).filter(
            CollectionTask.created_at >= datetime.now(timezone.utc) - timedelta(days=7)
        ).all()
        
        api_performance = self._analyze_recent_performance(recent_tasks)
        
        return {
            'priority_players': priority_players,
            'data_coverage': data_coverage,
            'quality_issues': quality_issues,
            'api_performance': api_performance,
            'current_time': datetime.now(timezone.utc)
        }

    async def _generate_priority_tasks(self, session: Session, analysis_data: Dict[str, Any]) -> List[Tuple[CollectionTask, TaskPriorityScore]]:
        """Generate prioritized collection tasks based on analysis."""
        
        priority_tasks = []
        
        for player in analysis_data['priority_players']:
            player_coverage = analysis_data['data_coverage'][player.id]
            player_quality = analysis_data['quality_issues'][player.id]
            
            for season in [2023, 2022, 2021]:  # Process recent seasons first
                season_coverage = player_coverage[season]
                season_quality = player_quality[season]
                
                # Skip if data is complete and high quality
                if (season_coverage['coverage_ratio'] > 0.95 and 
                    season_quality['overall_score'] > 0.8):
                    continue
                
                # Generate tasks for missing weeks
                missing_weeks = season_coverage['missing_weeks']
                
                # Prioritize recent weeks and high-impact weeks
                sorted_weeks = self._sort_weeks_by_priority(missing_weeks, season)
                
                # Limit weeks per player to prevent queue overflow
                max_weeks = 5 if season == 2023 else 3
                
                for week in sorted_weeks[:max_weeks]:
                    # Calculate priority score
                    priority_score = self._calculate_task_priority(
                        player, season, week, 
                        season_coverage, season_quality,
                        analysis_data['api_performance']
                    )
                    
                    # Create collection task
                    task = CollectionTask(
                        task_type='player_stats',
                        player_id=player.id,
                        team_id=player.team_id,
                        season=season,
                        week=week,
                        priority=priority_score.urgency.value,
                        scheduled_at=self._calculate_optimal_schedule_time(priority_score)
                    )
                    
                    priority_tasks.append((task, priority_score))
        
        # Sort by priority score
        priority_tasks.sort(key=lambda x: x[1], reverse=True)
        
        return priority_tasks

    def _calculate_task_priority(self, player: Player, season: int, week: int,
                               coverage_data: Dict[str, Any], quality_data: Dict[str, Any],
                               api_performance: Dict[str, Any]) -> TaskPriorityScore:
        """Calculate comprehensive priority score for a collection task."""
        
        # Fantasy relevance (based on position and player performance)
        position_weight = self.position_weights.get(player.position, 0.1)
        fantasy_priority = getattr(player, 'fantasy_priority_score', 0.5)
        fantasy_relevance = (position_weight + fantasy_priority) / 2
        
        # Data freshness (how outdated/missing the data is)
        if week in coverage_data['missing_weeks']:
            data_freshness = 1.0  # Missing data = highest freshness need
        else:
            # Existing data freshness based on last update
            data_freshness = 0.3  # Lower priority for existing data
        
        # Quality impact (how much this improves overall quality)
        current_quality = quality_data['overall_score']
        quality_impact = max(0, (0.8 - current_quality))  # Impact of bringing to 0.8 quality
        
        # Completeness gap (how much this fills overall picture)
        coverage_ratio = coverage_data['coverage_ratio']
        completeness_gap = 1.0 - coverage_ratio
        
        # Position priority (key fantasy positions get boost)
        position_priority = position_weight
        
        # Season recency (recent seasons more important)
        season_recency = self.season_weights.get(season, 0.5)
        
        # API efficiency (consider batching opportunities)
        api_efficiency = 0.7  # Default, can be adjusted based on batch potential
        
        # Recent week bonus (weeks 10-17 are more important for fantasy)
        if week >= 10:
            fantasy_relevance = min(1.0, fantasy_relevance * 1.2)
        
        return TaskPriorityScore(
            fantasy_relevance=fantasy_relevance,
            data_freshness=data_freshness,
            quality_impact=quality_impact,
            completeness_gap=completeness_gap,
            position_priority=position_priority,
            season_recency=season_recency,
            api_efficiency=api_efficiency
        )

    def _sort_weeks_by_priority(self, weeks: List[int], season: int) -> List[int]:
        """Sort weeks by collection priority."""
        
        def week_priority(week: int) -> float:
            # Recent weeks in season get higher priority
            if season == 2023:
                if week >= 15:  # Playoff weeks
                    return week + 10
                elif week >= 10:  # Late season
                    return week + 5
                else:
                    return week
            else:
                # For historical seasons, prioritize key weeks
                if week in [1, 8, 15, 17]:  # Season start, mid, playoff, end
                    return week + 5
                else:
                    return week
        
        return sorted(weeks, key=week_priority, reverse=True)

    def _calculate_optimal_schedule_time(self, priority_score: TaskPriorityScore) -> datetime:
        """Calculate optimal scheduling time based on priority."""
        
        base_time = datetime.now(timezone.utc)
        
        # Schedule based on urgency
        if priority_score.urgency == UrgencyLevel.IMMEDIATE:
            return base_time
        elif priority_score.urgency == UrgencyLevel.HIGH:
            return base_time + timedelta(minutes=30)
        elif priority_score.urgency == UrgencyLevel.MEDIUM:
            return base_time + timedelta(hours=2)
        elif priority_score.urgency == UrgencyLevel.LOW:
            return base_time + timedelta(hours=12)
        else:  # BACKGROUND
            return base_time + timedelta(days=1)

    async def _optimize_and_queue(self, session: Session, 
                                priority_tasks: List[Tuple[CollectionTask, TaskPriorityScore]],
                                limit: Optional[int] = None) -> int:
        """Optimize task order and add to queue with dependency management."""
        
        # Apply limit if specified
        if limit:
            priority_tasks = priority_tasks[:limit]
        
        # Group tasks for batch optimization
        batch_groups = self._create_batch_groups(priority_tasks)
        
        queued_count = 0
        
        for (task, priority_score) in priority_tasks:
            try:
                # Add task to database
                session.add(task)
                session.flush()  # Get task ID
                
                # Create priority task for queue
                priority_task = PriorityTask(
                    task_id=task.id,
                    priority_score=priority_score,
                    scheduled_time=task.scheduled_at,
                    batch_group=batch_groups.get(task.id)
                )
                
                # Add to queue
                await self._add_to_queue(priority_task)
                queued_count += 1
                
            except Exception as e:
                logger.error(f"Error queuing task for player {task.player_id}: {e}")
                continue
        
        session.commit()
        
        # Optimize queue order
        await self._optimize_queue_order()
        
        return queued_count

    def _create_batch_groups(self, priority_tasks: List[Tuple[CollectionTask, TaskPriorityScore]]) -> Dict[int, str]:
        """Create batch groups for efficient API usage."""
        
        batch_groups = {}
        
        # Group by team/season for efficient batching
        team_season_groups = defaultdict(list)
        
        for task, priority_score in priority_tasks:
            group_key = f"team_{task.team_id}_season_{task.season}"
            team_season_groups[group_key].append(task.id)
        
        # Assign batch group IDs
        for group_key, task_ids in team_season_groups.items():
            if len(task_ids) >= 3:  # Only create batches for 3+ tasks
                for task_id in task_ids:
                    batch_groups[task_id] = group_key
        
        return batch_groups

    async def _add_to_queue(self, priority_task: PriorityTask) -> None:
        """Add task to priority queue with proper ordering."""
        
        if len(self._priority_heap) >= self.max_queue_size:
            # Remove lowest priority task if queue is full
            lowest_priority_task = max(self._priority_heap)
            await self._remove_from_queue(lowest_priority_task.task_id)
        
        # Add to heap and lookup
        heapq.heappush(self._priority_heap, priority_task)
        self._task_lookup[priority_task.task_id] = priority_task
        
        # Update batch groups
        if priority_task.batch_group:
            self._batch_groups[priority_task.batch_group].append(priority_task.task_id)
        
        # Update statistics
        self.stats['tasks_queued'] += 1

    async def _remove_from_queue(self, task_id: int) -> bool:
        """Remove task from queue."""
        
        if task_id not in self._task_lookup:
            return False
        
        priority_task = self._task_lookup[task_id]
        
        # Remove from lookup
        del self._task_lookup[task_id]
        
        # Remove from batch groups
        if priority_task.batch_group:
            self._batch_groups[priority_task.batch_group].remove(task_id)
            if not self._batch_groups[priority_task.batch_group]:
                del self._batch_groups[priority_task.batch_group]
        
        # Mark as removed in heap (will be filtered out during pop)
        priority_task.task_id = -1
        
        return True

    async def get_next_tasks(self, count: int = 1, 
                           consider_batches: bool = True) -> List[PriorityTask]:
        """Get next highest priority tasks for processing."""
        
        if not self._priority_heap:
            return []
        
        next_tasks = []
        
        if consider_batches:
            # Try to get batch of related tasks
            batch_tasks = await self._get_batch_tasks(count)
            if batch_tasks:
                return batch_tasks
        
        # Get individual high-priority tasks
        while len(next_tasks) < count and self._priority_heap:
            # Get highest priority task
            priority_task = heapq.heappop(self._priority_heap)
            
            # Skip removed tasks
            if priority_task.task_id == -1:
                continue
            
            # Check if task is ready (dependencies met, scheduled time passed)
            if await self._is_task_ready(priority_task):
                next_tasks.append(priority_task)
                # Remove from lookup since it's being processed
                del self._task_lookup[priority_task.task_id]
            else:
                # Put back in queue if not ready
                heapq.heappush(self._priority_heap, priority_task)
                break
        
        return next_tasks

    async def _get_batch_tasks(self, max_count: int) -> List[PriorityTask]:
        """Get tasks that can be efficiently batched together."""
        
        # Look for ready batch groups
        for batch_group, task_ids in self._batch_groups.items():
            if len(task_ids) >= 2:  # Minimum batch size
                batch_tasks = []
                
                for task_id in task_ids[:max_count]:
                    if task_id in self._task_lookup:
                        priority_task = self._task_lookup[task_id]
                        if await self._is_task_ready(priority_task):
                            batch_tasks.append(priority_task)
                
                if len(batch_tasks) >= 2:
                    # Remove from queue structures
                    for task in batch_tasks:
                        await self._remove_from_queue(task.task_id)
                    
                    return batch_tasks
        
        return []

    async def _is_task_ready(self, priority_task: PriorityTask) -> bool:
        """Check if task is ready for processing."""
        
        # Check scheduled time
        if priority_task.scheduled_time > datetime.now(timezone.utc):
            return False
        
        # Check dependencies
        if priority_task.dependencies:
            # Check if all dependencies are completed
            async with get_db_session() as session:
                completed_deps = session.query(CollectionTask.id).filter(
                    CollectionTask.id.in_(priority_task.dependencies),
                    CollectionTask.status == CollectionStatus.COMPLETED.value
                ).all()
                
                completed_dep_ids = {dep[0] for dep in completed_deps}
                
                if not priority_task.dependencies.issubset(completed_dep_ids):
                    return False
        
        return True

    async def _optimize_queue_order(self) -> None:
        """Optimize queue order based on current performance data."""
        
        # Re-heapify to ensure proper ordering
        heapq.heapify(self._priority_heap)
        
        # Update queue size statistics
        current_size = len(self._priority_heap)
        self.stats['queue_size_history'].append({
            'timestamp': datetime.now(timezone.utc),
            'size': current_size
        })

    def _analyze_recent_performance(self, recent_tasks: List[CollectionTask]) -> Dict[str, Any]:
        """Analyze recent task performance for optimization."""
        
        if not recent_tasks:
            return {'avg_response_time': 5.0, 'success_rate': 1.0}
        
        completed_tasks = [t for t in recent_tasks if t.status == CollectionStatus.COMPLETED.value]
        failed_tasks = [t for t in recent_tasks if t.status == CollectionStatus.FAILED.value]
        
        success_rate = len(completed_tasks) / len(recent_tasks) if recent_tasks else 1.0
        
        response_times = [t.api_response_time for t in completed_tasks if t.api_response_time]
        avg_response_time = statistics.mean(response_times) if response_times else 5.0
        
        return {
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'total_tasks': len(recent_tasks),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks)
        }

    async def mark_task_completed(self, task_id: int, success: bool, 
                                execution_time: float) -> None:
        """Mark task as completed and update performance analytics."""
        
        completion_data = {
            'task_id': task_id,
            'success': success,
            'execution_time': execution_time,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self._completion_history.append(completion_data)
        
        # Update statistics
        if success:
            self.stats['tasks_completed'] += 1
        else:
            self.stats['tasks_failed'] += 1
        
        # Update wait time statistics
        if task_id in self._task_lookup:
            priority_task = self._task_lookup[task_id]
            wait_time = (datetime.now(timezone.utc) - priority_task.scheduled_time).total_seconds()
            
            current_avg = self.stats['avg_wait_time']
            total_completed = self.stats['tasks_completed'] + self.stats['tasks_failed']
            
            self.stats['avg_wait_time'] = (
                (current_avg * (total_completed - 1) + wait_time) / total_completed
            )

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        
        stats = self.stats.copy()
        
        # Current queue state
        stats.update({
            'current_queue_size': len(self._priority_heap),
            'batch_groups': len(self._batch_groups),
            'tasks_by_urgency': self._count_tasks_by_urgency(),
            'tasks_by_category': self._count_tasks_by_category()
        })
        
        # Performance metrics
        if self._completion_history:
            recent_completions = list(self._completion_history)[-20:]  # Last 20 tasks
            
            stats['recent_performance'] = {
                'success_rate': sum(1 for c in recent_completions if c['success']) / len(recent_completions),
                'avg_execution_time': statistics.mean(c['execution_time'] for c in recent_completions),
                'throughput_per_hour': len(recent_completions)  # Approximate
            }
        
        return stats

    def _count_tasks_by_urgency(self) -> Dict[str, int]:
        """Count tasks by urgency level."""
        
        urgency_counts = defaultdict(int)
        
        for priority_task in self._priority_heap:
            if priority_task.task_id != -1:  # Skip removed tasks
                urgency_counts[priority_task.priority_score.urgency.name] += 1
        
        return dict(urgency_counts)

    def _count_tasks_by_category(self) -> Dict[str, int]:
        """Count tasks by category."""
        
        category_counts = defaultdict(int)
        
        for priority_task in self._priority_heap:
            if priority_task.task_id != -1:  # Skip removed tasks
                category_counts[priority_task.priority_score.category.value] += 1
        
        return dict(category_counts)

    async def clear_queue(self) -> None:
        """Clear all tasks from queue."""
        
        self._priority_heap.clear()
        self._task_lookup.clear()
        self._batch_groups.clear()
        self._dependencies.clear()
        self._dependents.clear()
        
        logger.info("Priority queue cleared")

    async def add_dependency(self, task_id: int, depends_on: int) -> bool:
        """Add dependency relationship between tasks."""
        
        if task_id in self._task_lookup and depends_on in self._task_lookup:
            self._dependencies[task_id].add(depends_on)
            self._dependents[depends_on].add(task_id)
            
            # Update the task in queue
            self._task_lookup[task_id].dependencies.add(depends_on)
            return True
        
        return False

# Utility functions
async def create_intelligent_queue() -> IntelligentPriorityQueue:
    """Create and initialize intelligent priority queue."""
    
    queue = IntelligentPriorityQueue()
    
    # Analyze and populate initial tasks
    await queue.analyze_and_queue_tasks(limit=100)
    
    return queue

async def optimize_collection_schedule() -> Dict[str, Any]:
    """Analyze and optimize the overall collection schedule."""
    
    logger.info("Optimizing collection schedule")
    
    async with get_db_session() as session:
        # Get pending tasks
        pending_tasks = session.query(CollectionTask).filter(
            CollectionTask.status == CollectionStatus.PENDING.value
        ).all()
        
        # Get collection progress
        progress = session.query(CollectionProgress).first()
        
        # Calculate optimization recommendations
        recommendations = {
            'total_pending_tasks': len(pending_tasks),
            'estimated_completion_days': 0,
            'priority_adjustments': [],
            'batch_opportunities': [],
            'resource_optimization': {}
        }
        
        if progress and progress.api_calls_remaining:
            # Estimate completion time
            avg_tasks_per_day = progress.api_calls_remaining
            recommendations['estimated_completion_days'] = len(pending_tasks) / max(avg_tasks_per_day, 1)
        
        # Identify batch opportunities
        team_season_groups = defaultdict(list)
        for task in pending_tasks:
            if task.task_type == 'player_stats':
                key = f"{task.team_id}_{task.season}"
                team_season_groups[key].append(task)
        
        batch_opportunities = [
            {'group': key, 'task_count': len(tasks)}
            for key, tasks in team_season_groups.items()
            if len(tasks) >= 3
        ]
        
        recommendations['batch_opportunities'] = batch_opportunities
        
        logger.info(f"Schedule optimization completed: {len(batch_opportunities)} batch opportunities identified")
        
        return recommendations