"""
Enhanced database models with collection tracking and data quality metrics.
Location: src/fantasy_ai/core/data/storage/models.py
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Boolean, Text, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class CollectionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class DataQualityStatus(Enum):
    UNKNOWN = "unknown"
    VALID = "valid"
    ANOMALY = "anomaly"
    INVALID = "invalid"

class PlayerPosition(Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"
    OTHER = "OTHER"

# Core Data Models
class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), nullable=False)
    city = Column(String(100))
    logo = Column(String(255))
    conference = Column(String(50))
    division = Column(String(50))
    stadium = Column(String(100))
    coach = Column(String(100))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    players = relationship("Player", back_populates="team")
    collection_tasks = relationship("CollectionTask", back_populates="team")

class Player(Base):
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, nullable=False, index=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    
    # Player Info
    name = Column(String(100), nullable=False)
    firstname = Column(String(50))
    lastname = Column(String(50)) 
    position = Column(String(10), nullable=False, index=True)
    number = Column(Integer)
    age = Column(Integer)
    height = Column(String(20))
    weight = Column(String(20))
    experience = Column(Integer)
    college = Column(String(100))
    
    # Fantasy Metrics
    fantasy_priority_score = Column(Float, default=0.0, index=True)
    collection_priority = Column(Integer, default=5, index=True)  # 1=highest, 10=lowest
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_stats_update = Column(DateTime(timezone=True))
    
    # Relationships
    team = relationship("Team", back_populates="players")
    weekly_stats = relationship("WeeklyStats", back_populates="player")
    collection_tasks = relationship("CollectionTask", back_populates="player")
    quality_metrics = relationship("DataQualityMetric", back_populates="player")
    
    # Indexes
    __table_args__ = (
        Index('idx_player_position_priority', 'position', 'collection_priority'),
        Index('idx_player_team_position', 'team_id', 'position'),
    )

class WeeklyStats(Base):
    __tablename__ = 'weekly_stats'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    game_id = Column(String(50), index=True)
    
    # Game Context
    opponent_team = Column(String(10))
    is_home = Column(Boolean)
    game_date = Column(DateTime(timezone=True))
    
    # Offensive Stats
    passing_attempts = Column(Integer, default=0)
    passing_completions = Column(Integer, default=0)
    passing_yards = Column(Integer, default=0)
    passing_touchdowns = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    rushing_attempts = Column(Integer, default=0)
    rushing_yards = Column(Integer, default=0)
    rushing_touchdowns = Column(Integer, default=0)
    receiving_targets = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    receiving_yards = Column(Integer, default=0)
    receiving_touchdowns = Column(Integer, default=0)
    fumbles = Column(Integer, default=0)
    fumbles_lost = Column(Integer, default=0)
    
    # Fantasy Points
    fantasy_points_standard = Column(Float, default=0.0)
    fantasy_points_ppr = Column(Float, default=0.0)
    fantasy_points_half_ppr = Column(Float, default=0.0)
    
    # Data Quality
    data_quality_score = Column(Float, default=1.0)
    is_validated = Column(Boolean, default=False)
    
    # Raw API Data
    raw_api_data = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="weekly_stats")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('player_id', 'season', 'week', name='uq_player_season_week'),
        Index('idx_stats_season_week', 'season', 'week'),
        Index('idx_stats_fantasy_points', 'fantasy_points_ppr'),
    )

# Collection Tracking Models
class CollectionTask(Base):
    __tablename__ = 'collection_tasks'
    
    id = Column(Integer, primary_key=True)
    task_type = Column(String(50), nullable=False)  # 'player_stats', 'player_info', 'team_info'
    
    # Target Information
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=True)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=True)
    season = Column(Integer, nullable=True)
    week = Column(Integer, nullable=True)
    
    # Collection Status
    status = Column(String(20), default=CollectionStatus.PENDING.value, index=True)
    priority = Column(Integer, default=5, index=True)  # 1=highest
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timing
    scheduled_at = Column(DateTime(timezone=True), default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    next_retry_at = Column(DateTime(timezone=True))
    
    # Results
    api_calls_made = Column(Integer, default=0)
    records_collected = Column(Integer, default=0)
    error_message = Column(Text)
    api_response_time = Column(Float)  # seconds
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="collection_tasks")
    player = relationship("Player", back_populates="collection_tasks")
    
    # Indexes
    __table_args__ = (
        Index('idx_task_status_priority', 'status', 'priority'),
        Index('idx_task_scheduled', 'scheduled_at'),
        Index('idx_task_type_status', 'task_type', 'status'),
    )

class ApiRateLimit(Base):
    __tablename__ = 'api_rate_limits'
    
    id = Column(Integer, primary_key=True)
    api_name = Column(String(50), nullable=False, index=True)  # 'nfl_api'
    endpoint = Column(String(100), nullable=False)
    
    # Rate Limiting
    requests_made = Column(Integer, default=0)
    requests_limit = Column(Integer, nullable=False)
    reset_time = Column(DateTime(timezone=True), nullable=False)
    
    # Performance Tracking
    avg_response_time = Column(Float, default=0.0)
    last_request_time = Column(DateTime(timezone=True))
    consecutive_errors = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('api_name', 'endpoint', name='uq_api_endpoint'),
    )

class DataQualityMetric(Base):
    __tablename__ = 'data_quality_metrics'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    season = Column(Integer, nullable=False)
    
    # Quality Metrics
    completeness_score = Column(Float, default=0.0)  # 0-1
    consistency_score = Column(Float, default=0.0)   # 0-1
    anomaly_score = Column(Float, default=0.0)       # 0-1 (higher = more anomalous)
    overall_quality_score = Column(Float, default=0.0) # 0-1
    
    # Specific Checks
    missing_weeks_count = Column(Integer, default=0)
    zero_stat_weeks = Column(Integer, default=0)
    outlier_weeks = Column(Integer, default=0)
    
    # Status
    quality_status = Column(String(20), default=DataQualityStatus.UNKNOWN.value)
    last_validation = Column(DateTime(timezone=True))
    validation_details = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="quality_metrics")
    
    __table_args__ = (
        UniqueConstraint('player_id', 'season', name='uq_player_season_quality'),
        Index('idx_quality_score', 'overall_quality_score'),
        Index('idx_quality_status', 'quality_status'),
    )

class CollectionProgress(Base):
    __tablename__ = 'collection_progress'
    
    id = Column(Integer, primary_key=True)
    
    # Progress Tracking
    total_players = Column(Integer, default=0)
    players_completed = Column(Integer, default=0)
    total_seasons = Column(Integer, default=0)
    seasons_completed = Column(Integer, default=0)
    total_weeks = Column(Integer, default=0)
    weeks_completed = Column(Integer, default=0)
    
    # API Usage
    total_api_calls = Column(Integer, default=0)
    api_calls_today = Column(Integer, default=0)
    api_calls_remaining = Column(Integer, default=100)
    
    # Performance
    avg_collection_time = Column(Float, default=0.0)
    estimated_completion = Column(DateTime(timezone=True))
    
    # Current State
    current_priority_position = Column(String(10))  # QB, RB, WR, TE
    current_team_id = Column(Integer)
    current_season = Column(Integer, default=2023)
    
    # Metadata
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    collection_start_date = Column(DateTime(timezone=True))

# Utility function for priority scoring
def calculate_fantasy_priority_score(position: str, stats_data: Dict[str, Any]) -> float:
    """Calculate fantasy priority score based on position and historical performance."""
    
    position_weights = {
        'QB': 1.0,
        'RB': 1.0, 
        'WR': 1.0,
        'TE': 1.0,
        'K': 0.3,
        'DEF': 0.2,
        'OTHER': 0.1
    }
    
    base_score = position_weights.get(position, 0.1)
    
    # Add performance-based scoring if stats available
    if stats_data:
        fantasy_points = stats_data.get('fantasy_points_ppr', 0)
        games_played = stats_data.get('games_played', 1)
        
        if games_played > 0:
            ppg = fantasy_points / games_played
            # Normalize to 0-1 scale (assuming 30+ PPG is elite)
            performance_score = min(ppg / 30.0, 1.0)
            base_score = base_score * (0.5 + 0.5 * performance_score)
    
    return round(base_score, 3)