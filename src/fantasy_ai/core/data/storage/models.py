"""
SQLite database models for Fantasy Football AI Assistant.

This module defines the database schema and provides ORM-like functionality
for storing and retrieving fantasy football data.
"""

import sqlite3
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Player model representing an NFL player."""
    player_id: int
    name: str
    position: str
    team: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class WeeklyStats:
    """Weekly statistics for a player in a specific season/week."""
    id: Optional[int] = None
    player_id: int = 0
    season: int = 0
    week: int = 0
    fantasy_points: float = 0.0
    projected_points: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class Team:
    """NFL team information."""
    team_code: str
    team_name: str
    conference: Optional[str] = None
    division: Optional[str] = None


class DatabaseManager:
    """Manages SQLite database operations for fantasy football data."""
    
    def __init__(self, db_path: str = "data/fantasy_football.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    team_code TEXT PRIMARY KEY,
                    team_name TEXT NOT NULL,
                    conference TEXT,
                    division TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    position TEXT NOT NULL,
                    team TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team) REFERENCES teams (team_code)
                )
            """)
            
            # Weekly stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    fantasy_points REAL DEFAULT 0.0,
                    projected_points REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id),
                    UNIQUE(player_id, season, week)
                )
            """)
            
            # Seasons tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seasons (
                    season INTEGER PRIMARY KEY,
                    status TEXT DEFAULT 'active',
                    weeks_completed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_season_week ON weekly_stats(player_id, season, week)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_season_week ON weekly_stats(season, week)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_position ON players(position)")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def insert_player(self, player: Player) -> bool:
        """Insert or update a player record."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO players 
                    (player_id, name, position, team, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (player.player_id, player.name, player.position, player.team))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error inserting player {player.name}: {e}")
            return False
    
    def insert_weekly_stats(self, stats: WeeklyStats) -> bool:
        """Insert or update weekly statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO weekly_stats 
                    (player_id, season, week, fantasy_points, projected_points)
                    VALUES (?, ?, ?, ?, ?)
                """, (stats.player_id, stats.season, stats.week, 
                     stats.fantasy_points, stats.projected_points))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error inserting weekly stats: {e}")
            return False
    
    def bulk_insert_weekly_stats(self, stats_list: List[WeeklyStats]) -> int:
        """Bulk insert weekly statistics for better performance."""
        success_count = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                data = [(s.player_id, s.season, s.week, s.fantasy_points, s.projected_points) 
                       for s in stats_list]
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO weekly_stats 
                    (player_id, season, week, fantasy_points, projected_points)
                    VALUES (?, ?, ?, ?, ?)
                """, data)
                
                success_count = cursor.rowcount
                conn.commit()
                logger.info(f"Bulk inserted {success_count} weekly stats records")
                
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            
        return success_count
    
    def get_player_stats(self, player_id: int, season: Optional[int] = None) -> pd.DataFrame:
        """Get historical stats for a specific player."""
        query = """
            SELECT ws.*, p.name, p.position, p.team
            FROM weekly_stats ws
            JOIN players p ON ws.player_id = p.player_id
            WHERE ws.player_id = ?
        """
        params = [player_id]
        
        if season:
            query += " AND ws.season = ?"
            params.append(season)
            
        query += " ORDER BY ws.season, ws.week"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_season_stats(self, season: int, position: Optional[str] = None) -> pd.DataFrame:
        """Get all stats for a specific season, optionally filtered by position."""
        query = """
            SELECT ws.*, p.name, p.position, p.team
            FROM weekly_stats ws
            JOIN players p ON ws.player_id = p.player_id
            WHERE ws.season = ?
        """
        params = [season]
        
        if position:
            query += " AND p.position = ?"
            params.append(position)
            
        query += " ORDER BY p.position, ws.week"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_training_data(self, seasons: List[int], positions: Optional[List[str]] = None) -> pd.DataFrame:
        """Get formatted training data for ML models."""
        query = """
            SELECT 
                ws.player_id,
                p.name,
                p.position,
                p.team,
                ws.season,
                ws.week,
                ws.fantasy_points,
                ws.projected_points,
                ws.fantasy_points - ws.projected_points as points_vs_projection
            FROM weekly_stats ws
            JOIN players p ON ws.player_id = p.player_id
            WHERE ws.season IN ({})
        """.format(','.join('?' * len(seasons)))
        
        params = seasons
        
        if positions:
            query += " AND p.position IN ({})".format(','.join('?' * len(positions)))
            params.extend(positions)
        
        query += " ORDER BY ws.season, ws.week, p.position"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records in each table
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM players")
            stats['total_players'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM weekly_stats")
            stats['total_weekly_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT season) FROM weekly_stats")
            stats['seasons_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(season), MAX(season) FROM weekly_stats")
            min_season, max_season = cursor.fetchone()
            stats['season_range'] = f"{min_season}-{max_season}" if min_season else "No data"
            
            cursor.execute("""
                SELECT position, COUNT(*) 
                FROM players 
                GROUP BY position 
                ORDER BY COUNT(*) DESC
            """)
            stats['players_by_position'] = dict(cursor.fetchall())
            
            return stats
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate database integrity and return issues found."""
        issues = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check for orphaned weekly stats
            cursor.execute("""
                SELECT COUNT(*) FROM weekly_stats ws
                LEFT JOIN players p ON ws.player_id = p.player_id
                WHERE p.player_id IS NULL
            """)
            issues['orphaned_stats'] = cursor.fetchone()[0]
            
            # Check for missing projected points
            cursor.execute("""
                SELECT COUNT(*) FROM weekly_stats 
                WHERE projected_points IS NULL OR projected_points = 0
            """)
            issues['missing_projections'] = cursor.fetchone()[0]
            
            # Check for duplicate records
            cursor.execute("""
                SELECT player_id, season, week, COUNT(*) as count
                FROM weekly_stats 
                GROUP BY player_id, season, week
                HAVING count > 1
            """)
            duplicates = cursor.fetchall()
            issues['duplicate_records'] = len(duplicates)
            
            return issues


# Convenience function for easy database access
def get_database(db_path: str = "data/fantasy_football.db") -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager(db_path)


if __name__ == "__main__":
    # Example usage and testing
    db = get_database()
    
    # Print database statistics
    stats = db.get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check data integrity
    issues = db.validate_data_integrity()
    print("\nData Integrity Check:")
    for key, value in issues.items():
        print(f"  {key}: {value}")