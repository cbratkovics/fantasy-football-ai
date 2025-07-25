"""
Simplified Database Configuration for SQLite Only.
Location: src/fantasy_ai/core/data/storage/simple_database.py
"""

import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

logger = logging.getLogger(__name__)

class SimpleDatabaseManager:
    """Simplified database manager for SQLite only."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default database location
            project_root = Path(__file__).parent.parent.parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "fantasy_football.db"
        
        self.db_path = Path(db_path)
        self.database_url = f"sqlite+aiosqlite:///{self.db_path}"
        
        # Simple async engine for SQLite - CORRECTLY CONFIGURED
        self.engine = create_async_engine(
            self.database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False,  # Set to True for debugging
            future=True
        )
        
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Track initialization
        self._tables_created = False
    
    async def create_tables(self, drop_existing: bool = False):
        """Create database tables."""
        try:
            async with self.engine.begin() as conn:
                if drop_existing:
                    logger.info("Dropping existing tables")
                    await conn.run_sync(Base.metadata.drop_all)
                
                logger.info("Creating database tables")
                await conn.run_sync(Base.metadata.create_all)
                
            self._tables_created = True
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def ensure_tables_exist(self):
        """Ensure tables exist, create if they don't."""
        if not self._tables_created:
            try:
                # Test if tables exist by trying a simple query
                async with self.session_factory() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                    self._tables_created = True
                    logger.info("Database tables already exist")
            except Exception:
                logger.info("Database tables don't exist, creating them")
                await self.create_tables()
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            async with self.session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                test_value = result.scalar()
                return test_value == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic table creation."""
        # Ensure tables exist before providing session
        await self.ensure_tables_exist()
        
        async with self.session_factory() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                logger.error(f"Database error: {e}")
                await session.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await session.rollback()
                raise
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")

# Global instance
_simple_db_manager = None

def get_simple_db_manager() -> SimpleDatabaseManager:
    """Get global simple database manager."""
    global _simple_db_manager
    if _simple_db_manager is None:
        _simple_db_manager = SimpleDatabaseManager()
    return _simple_db_manager

@asynccontextmanager
async def get_simple_session() -> AsyncGenerator[AsyncSession, None]:
    """Get simple database session with automatic setup."""
    db_manager = get_simple_db_manager()
    async with db_manager.get_session() as session:
        yield session

# Initialization function for the ETL
async def initialize_simple_database():
    """Initialize the simple database (create tables, etc.)."""
    db_manager = get_simple_db_manager()
    
    # Test connection
    if not await db_manager.test_connection():
        logger.info("Database connection failed, creating tables")
        await db_manager.create_tables()
    
    # Ensure tables exist
    await db_manager.ensure_tables_exist()
    
    logger.info("Simple database initialization completed")
    return db_manager