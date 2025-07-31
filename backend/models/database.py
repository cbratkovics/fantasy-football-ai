"""
Production Database Models for Fantasy Football AI
- Async SQLAlchemy for high performance
- Comprehensive indexes for query optimization
- JSONB fields for flexible data storage
- Audit trails with timestamps
- Type hints for better code quality
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean,
    ForeignKey, UniqueConstraint, Index, Text, DECIMAL, create_engine
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import enum

Base = declarative_base()


class SubscriptionTier(enum.Enum):
    """User subscription tiers"""
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class Player(Base):
    """
    Core player information from Sleeper API
    Optimized for quick lookups and joins
    """
    __tablename__ = 'players'
    
    player_id = Column(String, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    position = Column(String, index=True)
    team = Column(String, index=True)
    fantasy_positions = Column(JSONB)  # ['QB', 'SUPER_FLEX']
    age = Column(Integer)
    years_exp = Column(Integer)
    status = Column(String)  # Active, Injured, etc.
    meta_data = Column(JSONB)  # Full API response for flexibility
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    stats = relationship("PlayerStats", back_populates="player")
    predictions = relationship("Prediction", back_populates="player")
    tier_assignments = relationship("DraftTier", back_populates="player")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_player_position_team', position, team),
        Index('idx_player_name', first_name, last_name),
    )
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class PlayerStats(Base):
    """
    Historical player statistics by week
    Stores raw stats and calculated fantasy points
    """
    __tablename__ = 'player_stats'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String, ForeignKey('players.player_id'))
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    
    # Raw statistics as JSONB for flexibility
    stats = Column(JSONB, nullable=False)
    
    # Pre-calculated fantasy points for quick access
    fantasy_points_std = Column(DECIMAL(5, 2))
    fantasy_points_ppr = Column(DECIMAL(5, 2))
    fantasy_points_half = Column(DECIMAL(5, 2))
    
    # Game context
    opponent = Column(String)
    is_home = Column(Boolean)
    game_date = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="stats")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('player_id', 'season', 'week'),
        Index('idx_stats_season_week', season, week),
        Index('idx_stats_player_season', player_id, season),
    )


class Prediction(Base):
    """
    ML model predictions for player performance
    Includes confidence intervals and model versioning
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String, ForeignKey('players.player_id'))
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    
    # Predictions with confidence
    predicted_points = Column(DECIMAL(5, 2), nullable=False)
    confidence_interval = Column(JSONB)  # {"low": 5.2, "high": 15.8}
    prediction_std = Column(DECIMAL(4, 2))  # Standard deviation
    
    # Model metadata
    model_version = Column(String, nullable=False)
    model_type = Column(String)  # 'neural_network', 'ensemble'
    features_used = Column(JSONB)  # For explainability
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_predictions_player_week', player_id, season, week),
        Index('idx_predictions_created', created_at.desc()),
    )


class User(Base):
    """User accounts with subscription management"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    
    # Subscription info
    subscription_tier = Column(String, default='free')
    stripe_customer_id = Column(String, unique=True)
    
    # Tracking
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    prediction_usage = relationship("PredictionUsage", back_populates="user")


class Subscription(Base):
    """User subscription details"""
    __tablename__ = 'subscriptions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), unique=True)
    stripe_subscription_id = Column(String, unique=True)
    
    status = Column(String)  # active, trialing, past_due, canceled
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    trial_end = Column(DateTime)
    canceled_at = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="subscription")


class PredictionUsage(Base):
    """Track prediction usage for rate limiting"""
    __tablename__ = 'prediction_usage'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    week_start = Column(DateTime, nullable=False)
    predictions_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="prediction_usage")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'week_start'),
        Index('idx_usage_user_week', user_id, week_start),
    )


class DraftTier(Base):
    """
    GMM clustering results for draft optimization
    Stores tier assignments with probability scores
    """
    __tablename__ = 'draft_tiers'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String, ForeignKey('players.player_id'))
    season = Column(Integer, nullable=False)
    
    # Tier assignment (1-16)
    tier = Column(Integer, nullable=False, index=True)
    probability = Column(DECIMAL(4, 3))  # Confidence in tier assignment
    
    # Cluster characteristics
    cluster_features = Column(JSONB)  # Mean values of cluster
    tier_label = Column(String)  # "Elite QB", "RB2", etc.
    
    # Alternative tier probabilities
    alt_tiers = Column(JSONB)  # {"tier_2": 0.25, "tier_3": 0.15}
    
    created_at = Column(DateTime, default=func.now())
    model_version = Column(String)
    
    # Relationships
    player = relationship("Player", back_populates="tier_assignments")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('player_id', 'season', 'model_version'),
        Index('idx_tier_season_tier', season, tier),
    )




class UserLeague(Base):
    """
    User's fantasy leagues for personalized predictions
    Supports multiple platforms (Sleeper, ESPN, Yahoo)
    """
    __tablename__ = 'user_leagues'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    
    # League identification
    platform = Column(String, default='sleeper')  # sleeper, espn, yahoo
    league_id = Column(String, nullable=False)
    league_name = Column(String)
    
    # League settings
    scoring_settings = Column(JSONB)  # Custom scoring rules
    roster_positions = Column(JSONB)  # ['QB', 'RB', 'RB', 'WR', ...]
    team_count = Column(Integer)
    
    # User's team in this league
    roster = Column(JSONB)  # Current player IDs
    draft_position = Column(Integer)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="leagues")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'platform', 'league_id'),
    )


class ModelPerformance(Base):
    """
    Track ML model performance for continuous improvement
    Essential for showcasing model reliability
    """
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    
    # Performance metrics
    season = Column(Integer)
    week = Column(Integer)
    position = Column(String)
    
    # Accuracy metrics
    mae = Column(DECIMAL(4, 2))  # Mean Absolute Error
    rmse = Column(DECIMAL(4, 2))  # Root Mean Square Error
    mape = Column(DECIMAL(4, 2))  # Mean Absolute Percentage Error
    r_squared = Column(DECIMAL(4, 3))
    
    # Additional metrics
    sample_size = Column(Integer)
    metrics_detail = Column(JSONB)  # Position-specific metrics
    
    evaluated_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_model_performance', model_version, season, week),
    )


# Database connection and session management
class DatabaseManager:
    """Async database manager with connection pooling"""
    
    def __init__(self, database_url: str):
        # Convert to async URL
        if database_url.startswith('postgresql://'):
            database_url = database_url.replace(
                'postgresql://', 
                'postgresql+asyncpg://'
            )
        
        self.engine = create_async_engine(
            database_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections
            echo=False  # Set True for SQL logging
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Create all tables in database"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all tables (use with caution!)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        async with self.async_session() as session:
            yield session


# Create engine for database connection
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")
engine = create_engine(DATABASE_URL)

# Create SessionLocal for dependency injection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Example usage
async def example_usage():
    """Demonstrate database operations"""
    db = DatabaseManager("postgresql://user:pass@localhost/fantasy_football")
    
    # Create tables
    await db.create_tables()
    
    # Example: Add a player
    async with db.async_session() as session:
        player = Player(
            player_id="1234",
            first_name="Patrick",
            last_name="Mahomes",
            position="QB",
            team="KC",
            fantasy_positions=["QB"],
            metadata={"espn_id": "3139477"}
        )
        session.add(player)
        await session.commit()
        
    print("Database initialized successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())