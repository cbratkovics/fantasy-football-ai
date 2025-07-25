"""
Simplified Database Configuration for SQLite Only.
Location: src/fantasy_ai/core/data/storage/simple_database.py
"""

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

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
        
        # Simple async engine for SQLite
        self.engine = create_async_engine(
            self.database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False  # Set to True for debugging
        )
        
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self, drop_existing: bool = False):
        """Create database tables."""
        async with self.engine.begin() as conn:
            if drop_existing:
                await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            async with self.session_factory() as session:
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()

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
    """Get simple database session."""
    db_manager = get_simple_db_manager()
    async with db_manager.get_session() as session:
        yield session