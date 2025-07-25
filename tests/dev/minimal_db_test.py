#!/usr/bin/env python3
"""
Minimal database test to isolate SQLite connection issues.
Run this to test basic SQLite functionality.
"""

import asyncio
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from datetime import datetime

Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'test_table'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

async def test_minimal_db():
    """Test minimal SQLite async setup."""
    
    print("🧪 Testing minimal SQLite async connection...")
    
    # Create database path
    db_path = Path("./test_minimal.db")
    db_url = f"sqlite+aiosqlite:///{db_path}"
    
    try:
        # Create engine with minimal settings
        engine = create_async_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=True
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Test session
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession)
        
        async with session_factory() as session:
            # Insert test record
            test_record = TestTable(name="test")
            session.add(test_record)
            await session.commit()
            
            print("✅ Minimal database test successful!")
            
        await engine.dispose()
        
        # Clean up
        if db_path.exists():
            db_path.unlink()
            
    except Exception as e:
        print(f"❌ Minimal database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_minimal_db())