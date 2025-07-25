"""
Enhanced Database Connection and Session Management with Async Support.
Location: src/fantasy_ai/core/data/storage/database.py
"""

import os
import asyncio
import logging
from typing import Optional, AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with both sync and async support."""
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """Initialize database manager with configuration."""
        
        # Determine database URL
        if database_url:
            self.database_url = database_url
        else:
            # Default to project database
            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = project_root / "data" / "fantasy_football.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_path}"
        
        # Create async URL for async operations
        if self.database_url.startswith('sqlite:'):
            self.async_database_url = self.database_url.replace('sqlite:', 'sqlite+aiosqlite:', 1)
        else:
            self.async_database_url = self.database_url
        
        self.echo = echo
        
        # Initialize engines
        self._sync_engine = None
        self._async_engine = None
        self._sync_session_factory = None
        self._async_session_factory = None
        
        # FIXED: Configure pool settings based on database type
        self.is_sqlite = 'sqlite' in self.database_url.lower()
        
        if self.is_sqlite:
            # SQLite-specific settings (no pooling parameters)
            self.pool_settings = {
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30,
                    'isolation_level': None  # Enable autocommit mode
                }
            }
        else:
            # PostgreSQL/MySQL settings (with pooling)
            self.pool_settings = {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_pre_ping': True,
                'pool_recycle': 3600
            }

    def get_sync_engine(self):
        """Get or create synchronous database engine."""
        if self._sync_engine is None:
            self._sync_engine = create_engine(
                self.database_url,
                echo=self.echo,
                **self.pool_settings
            )
            
            # Configure SQLite specific settings
            if self.is_sqlite:
                @event.listens_for(self._sync_engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    # Enable foreign key constraints
                    cursor.execute("PRAGMA foreign_keys=ON")
                    # Set journal mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL")
                    # Optimize for performance
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=10000")
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    cursor.close()
        
        return self._sync_engine

    def get_async_engine(self):
        """Get or create asynchronous database engine."""
        if self._async_engine is None:
            # FIXED: Use appropriate settings for each database type
            if self.is_sqlite:
                # SQLite async settings (no pooling)
                async_pool_settings = {
                    'poolclass': StaticPool,
                    'connect_args': {
                        'timeout': 30,
                        'check_same_thread': False
                    }
                }
            else:
                # PostgreSQL/MySQL async settings (with pooling)
                async_pool_settings = {
                    'pool_size': 10,
                    'max_overflow': 20,
                    'pool_pre_ping': True,
                    'pool_recycle': 3600
                }
            
            self._async_engine = create_async_engine(
                self.async_database_url,
                echo=self.echo,
                **async_pool_settings
            )
        
        return self._async_engine

    def get_sync_session_factory(self):
        """Get or create synchronous session factory."""
        if self._sync_session_factory is None:
            self._sync_session_factory = sessionmaker(
                bind=self.get_sync_engine(),
                expire_on_commit=False,
                autoflush=True
            )
        return self._sync_session_factory

    def get_async_session_factory(self):
        """Get or create asynchronous session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True
            )
        return self._async_session_factory

    async def create_tables(self, drop_existing: bool = False):
        """Create database tables using async engine."""
        async_engine = self.get_async_engine()
        
        async with async_engine.begin() as conn:
            if drop_existing:
                logger.info("Dropping existing tables")
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.info("Creating database tables")
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created successfully")

    def create_tables_sync(self, drop_existing: bool = False):
        """Create database tables using sync engine."""
        sync_engine = self.get_sync_engine()
        
        if drop_existing:
            logger.info("Dropping existing tables")
            Base.metadata.drop_all(sync_engine)
        
        logger.info("Creating database tables")
        Base.metadata.create_all(sync_engine)
        logger.info("Database tables created successfully")

    async def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            async with self.get_async_session() as session:
                # Simple query to test connection
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                await session.commit()
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        session_factory = self.get_async_session_factory()
        
        async with session_factory() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                logger.error(f"Database error in async session: {e}")
                await session.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error in async session: {e}")
                await session.rollback()
                raise

    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """Get sync database session with automatic cleanup."""
        session_factory = self.get_sync_session_factory()
        
        with session_factory() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                logger.error(f"Database error in sync session: {e}")
                session.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error in sync session: {e}")
                session.rollback()
                raise

    async def close_connections(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database connections closed")
        
        if self._sync_engine:
            self._sync_engine.dispose()
            logger.info("Sync database connections closed")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def initialize_database(database_url: Optional[str] = None, echo: bool = False):
    """Initialize global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    return _db_manager

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Convenience functions for common operations
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session (convenience function)."""
    db_manager = get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session

@contextmanager
def get_sync_db_session() -> Generator[Session, None, None]:
    """Get sync database session (convenience function)."""
    db_manager = get_database_manager()
    with db_manager.get_sync_session() as session:
        yield session

async def ensure_database_exists():
    """Ensure database and tables exist."""
    db_manager = get_database_manager()
    
    # Test connection first
    if not await db_manager.test_connection():
        logger.info("Database connection failed, creating tables")
        await db_manager.create_tables()
    else:
        logger.info("Database connection successful")

def ensure_database_exists_sync():
    """Ensure database and tables exist (synchronous version)."""
    db_manager = get_database_manager()
    
    try:
        # Test with a simple table creation (will fail if already exists)
        db_manager.create_tables_sync(drop_existing=False)
    except Exception as e:
        if "already exists" not in str(e).lower():
            logger.error(f"Database initialization error: {e}")
            raise

# Database maintenance utilities
async def optimize_database():
    """Optimize database performance (SQLite specific)."""
    db_manager = get_database_manager()
    
    if not db_manager.is_sqlite:
        logger.info("Database optimization only supported for SQLite")
        return
    
    async with get_db_session() as session:
        try:
            from sqlalchemy import text
            # SQLite optimization commands
            await session.execute(text("PRAGMA optimize"))
            await session.execute(text("VACUUM"))
            await session.execute(text("ANALYZE"))
            await session.commit()
            logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")


async def get_database_stats() -> dict:
    """Get database statistics and health metrics."""
    db_manager = get_database_manager()
    
    stats = {
        'database_url': db_manager.database_url,
        'database_type': 'sqlite' if db_manager.is_sqlite else 'other',
        'engine_type': 'async' if db_manager._async_engine else 'sync',
        'connection_pool_size': 'N/A (SQLite)' if db_manager.is_sqlite else db_manager.pool_settings.get('pool_size', 'N/A'),
        'tables': {}
    }
    
    try:
        async with get_db_session() as session:
            # Get table row counts - FIXED: Use SQLAlchemy 2.0 syntax
            from .models import Team, Player, WeeklyStats, CollectionTask, DataQualityMetric
            from sqlalchemy import select, func
            
            table_models = {
                'teams': Team,
                'players': Player,
                'weekly_stats': WeeklyStats,
                'collection_tasks': CollectionTask,
                'data_quality_metrics': DataQualityMetric
            }
            
            for table_name, model in table_models.items():
                try:
                    # FIXED: Use modern SQLAlchemy 2.0 syntax
                    result = await session.execute(select(func.count()).select_from(model))
                    count = result.scalar()
                    stats['tables'][table_name] = count
                except Exception as e:
                    stats['tables'][table_name] = f"Error: {e}"
            
            await session.commit()
            
    except Exception as e:
        stats['error'] = str(e)
    
    return stats

# Migration utilities
async def backup_database(backup_path: Optional[str] = None):
    """Create database backup (SQLite specific)."""
    db_manager = get_database_manager()
    
    if not db_manager.is_sqlite:
        logger.error("Database backup only supported for SQLite")
        return False
    
    try:
        import shutil
        from datetime import datetime
        
        # Extract database path from URL
        db_path = db_manager.database_url.replace('sqlite:///', '')
        
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{db_path}.backup_{timestamp}"
        
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False

# Testing utilities
async def reset_test_database():
    """Reset database for testing (drops and recreates all tables)."""
    db_manager = get_database_manager()
    
    logger.warning("Resetting test database - all data will be lost!")
    await db_manager.create_tables(drop_existing=True)
    logger.info("Test database reset completed")

# Connection health monitoring
class DatabaseHealthMonitor:
    """Monitor database connection health and performance."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.health_stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'last_health_check': None
        }
    
    async def health_check(self) -> dict:
        """Perform comprehensive database health check."""
        import time
        from datetime import datetime
        
        start_time = time.time()
        health_report = {
            'status': 'unknown',
            'response_time': 0.0,
            'connection_pool': {},
            'table_status': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Test basic connectivity
            async with self.db_manager.get_async_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
                await session.commit()
            
            response_time = time.time() - start_time
            health_report['response_time'] = round(response_time, 3)
            health_report['status'] = 'healthy'
            
            # Check connection pool if available (not for SQLite)
            if not self.db_manager.is_sqlite and hasattr(self.db_manager.get_async_engine(), 'pool'):
                pool = self.db_manager.get_async_engine().pool
                health_report['connection_pool'] = {
                    'size': getattr(pool, 'size', 'N/A'),
                    'checked_in': getattr(pool, 'checkedin', 'N/A'),
                    'checked_out': getattr(pool, 'checkedout', 'N/A'),
                    'overflow': getattr(pool, 'overflow', 'N/A')
                }
            else:
                health_report['connection_pool'] = {'type': 'SQLite (no pooling)'}
            
            # Update stats
            self.health_stats['total_queries'] += 1
            self.health_stats['avg_response_time'] = (
                (self.health_stats['avg_response_time'] * (self.health_stats['total_queries'] - 1) + response_time) 
                / self.health_stats['total_queries']
            )
            self.health_stats['last_health_check'] = datetime.now()
            
        except Exception as e:
            health_report['status'] = 'unhealthy'
            health_report['error'] = str(e)
            self.health_stats['failed_queries'] += 1
            
        return health_report

# Startup function for application initialization
async def startup_database():
    """Initialize database for application startup."""
    logger.info("Initializing database for application startup")
    
    # Initialize database manager
    db_manager = get_database_manager()
    
    # Ensure tables exist
    await ensure_database_exists()
    
    # Test connection
    if not await db_manager.test_connection():
        raise RuntimeError("Failed to establish database connection")
    
    # Log statistics
    stats = await get_database_stats()
    logger.info(f"Database startup completed: {stats}")
    
    return db_manager

# Shutdown function for application cleanup
async def shutdown_database():
    """Clean shutdown of database connections."""
    logger.info("Shutting down database connections")
    
    db_manager = get_database_manager()
    await db_manager.close_connections()
    
    logger.info("Database shutdown completed")