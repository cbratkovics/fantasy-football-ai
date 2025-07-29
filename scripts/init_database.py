#!/usr/bin/env python3
"""Initialize database schema and create tables for Fantasy Football AI"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime

# Import from backend/models/
from backend.models.database import Base, Player, PlayerStats, Prediction, User, DraftTier, UserLeague, ModelPerformance

# Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handle database initialization and schema creation"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Using DATABASE_URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
        
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create all database tables"""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False
    
    def create_indexes(self) -> bool:
        """Create performance indexes based on actual column names"""
        try:
            logger.info("Creating database indexes...")
            
            # Only create indexes that don't already exist
            index_definitions = [
                # Player indexes (player_id is already primary key)
                "CREATE INDEX IF NOT EXISTS idx_player_status ON players(status);",
                
                # Player stats indexes
                "CREATE INDEX IF NOT EXISTS idx_player_stats_player_id ON player_stats(player_id);",
                "CREATE INDEX IF NOT EXISTS idx_player_stats_season_week ON player_stats(season, week);",
                "CREATE INDEX IF NOT EXISTS idx_player_stats_composite ON player_stats(player_id, season, week);",
                
                # Predictions indexes
                "CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);",
                "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_predictions_season_week ON predictions(season, week);",
                
                # User indexes
                "CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);",
                
                # User league indexes
                "CREATE INDEX IF NOT EXISTS idx_user_league_user_id ON user_leagues(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_user_league_league_id ON user_leagues(league_id);",
                
                # Draft tier indexes
                "CREATE INDEX IF NOT EXISTS idx_draft_tier_player_id ON draft_tiers(player_id);",
                "CREATE INDEX IF NOT EXISTS idx_draft_tier_season ON draft_tiers(season);"
            ]
            
            with self.engine.connect() as conn:
                for index_sql in index_definitions:
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                    except Exception as e:
                        # Skip if index already exists
                        if "already exists" not in str(e):
                            logger.warning(f"Failed to create index: {index_sql} - {str(e)}")
                
            logger.info("Index creation completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            return False
    
    def verify_schema(self) -> dict:
        """Verify database schema was created correctly"""
        try:
            with self.engine.connect() as conn:
                # Get table information
                result = conn.execute(text("""
                    SELECT 
                        table_name,
                        (SELECT COUNT(*) 
                         FROM information_schema.columns 
                         WHERE table_name = t.table_name 
                         AND table_schema = 'public') as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """))
                
                tables = {}
                for table_name, column_count in result:
                    tables[table_name] = column_count
                
                logger.info("Database schema verification:")
                for table, col_count in tables.items():
                    logger.info(f"  - {table}: {col_count} columns")
                
                return {
                    'tables': tables,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Failed to verify schema: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def initialize(self) -> bool:
        """Run complete database initialization"""
        logger.info("Starting database initialization...")
        
        # Test connection
        if not self.test_connection():
            return False
        
        # Tables already created, just create indexes
        if not self.create_indexes():
            logger.warning("Some indexes may have failed, but continuing...")
        
        # Verify schema
        verification = self.verify_schema()
        if not verification['success']:
            logger.error("Schema verification failed")
            return False
        
        logger.info("Database initialization completed successfully!")
        return True


def main():
    """Main entry point"""
    initializer = DatabaseInitializer()
    
    if initializer.initialize():
        logger.info("Database is ready for use")
        sys.exit(0)
    else:
        logger.error("Database initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()