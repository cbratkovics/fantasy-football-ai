"""
Enhanced Scoring Engine for Multiple Fantasy Formats
Supports Standard, PPR, Half-PPR, and custom league settings
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import redis
import json
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.database import PlayerStats

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


@dataclass
class ScoringSettings:
    """Configurable scoring settings for different league formats"""
    # Passing
    pass_yd_per_point: float = 25.0
    pass_td: float = 4.0
    pass_int: float = -2.0
    pass_2pt: float = 2.0
    pass_bonus_300: float = 0.0
    pass_bonus_400: float = 0.0
    
    # Rushing
    rush_yd_per_point: float = 10.0
    rush_td: float = 6.0
    rush_2pt: float = 2.0
    rush_bonus_100: float = 0.0
    rush_bonus_200: float = 0.0
    
    # Receiving
    rec_yd_per_point: float = 10.0
    rec_td: float = 6.0
    rec_2pt: float = 2.0
    rec_bonus_100: float = 0.0
    rec_bonus_200: float = 0.0
    
    # Reception points (PPR settings)
    reception_points: float = 1.0  # 1.0 for PPR, 0.5 for Half-PPR, 0 for Standard
    
    # Special teams
    ret_td: float = 6.0
    
    # Turnovers
    fum_lost: float = -2.0
    fum_rec_td: float = 6.0
    
    # Kicking
    fg_0_39: float = 3.0
    fg_40_49: float = 4.0
    fg_50_plus: float = 5.0
    fg_miss: float = -1.0
    xp_made: float = 1.0
    xp_miss: float = -1.0
    
    # Team Defense (if needed)
    def_td: float = 6.0
    def_int: float = 2.0
    def_fum_rec: float = 2.0
    def_sack: float = 1.0
    def_safety: float = 2.0
    
    @classmethod
    def standard(cls):
        """Standard scoring (no PPR)"""
        return cls(reception_points=0.0)
    
    @classmethod
    def ppr(cls):
        """Full PPR scoring"""
        return cls(reception_points=1.0)
    
    @classmethod
    def half_ppr(cls):
        """Half PPR scoring"""
        return cls(reception_points=0.5)
    
    @classmethod
    def custom(cls, settings: Dict[str, float]):
        """Custom scoring from dict"""
        base = cls()
        for key, value in settings.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base


class ScoringEngine:
    """
    Comprehensive scoring engine with caching and custom format support
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize Redis for caching
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis caching enabled")
        except Exception as e:
            logger.warning(f"Redis not available, caching disabled: {str(e)}")
            self.redis_client = None
            self.cache_enabled = False
        
        # Common scoring formats
        self.scoring_formats = {
            'standard': ScoringSettings.standard(),
            'ppr': ScoringSettings.ppr(),
            'half_ppr': ScoringSettings.half_ppr()
        }
    
    def calculate_points(
        self,
        stats: Dict[str, Any],
        scoring_settings: ScoringSettings
    ) -> float:
        """Calculate fantasy points for given stats and scoring settings"""
        points = 0.0
        
        # Passing
        if 'pass_yd' in stats:
            points += stats['pass_yd'] / scoring_settings.pass_yd_per_point
        if 'pass_td' in stats:
            points += stats['pass_td'] * scoring_settings.pass_td
        if 'pass_int' in stats:
            points += stats['pass_int'] * scoring_settings.pass_int
        if 'pass_2pt' in stats:
            points += stats['pass_2pt'] * scoring_settings.pass_2pt
        
        # Passing bonuses
        if 'pass_yd' in stats:
            if stats['pass_yd'] >= 400:
                points += scoring_settings.pass_bonus_400
            elif stats['pass_yd'] >= 300:
                points += scoring_settings.pass_bonus_300
        
        # Rushing
        if 'rush_yd' in stats:
            points += stats['rush_yd'] / scoring_settings.rush_yd_per_point
        if 'rush_td' in stats:
            points += stats['rush_td'] * scoring_settings.rush_td
        if 'rush_2pt' in stats:
            points += stats['rush_2pt'] * scoring_settings.rush_2pt
        
        # Rushing bonuses
        if 'rush_yd' in stats:
            if stats['rush_yd'] >= 200:
                points += scoring_settings.rush_bonus_200
            elif stats['rush_yd'] >= 100:
                points += scoring_settings.rush_bonus_100
        
        # Receiving
        if 'rec' in stats:
            points += stats['rec'] * scoring_settings.reception_points
        if 'rec_yd' in stats:
            points += stats['rec_yd'] / scoring_settings.rec_yd_per_point
        if 'rec_td' in stats:
            points += stats['rec_td'] * scoring_settings.rec_td
        if 'rec_2pt' in stats:
            points += stats['rec_2pt'] * scoring_settings.rec_2pt
        
        # Receiving bonuses
        if 'rec_yd' in stats:
            if stats['rec_yd'] >= 200:
                points += scoring_settings.rec_bonus_200
            elif stats['rec_yd'] >= 100:
                points += scoring_settings.rec_bonus_100
        
        # Return touchdowns
        if 'ret_td' in stats:
            points += stats['ret_td'] * scoring_settings.ret_td
        
        # Fumbles
        if 'fum_lost' in stats:
            points += stats['fum_lost'] * scoring_settings.fum_lost
        if 'fum_rec_td' in stats:
            points += stats['fum_rec_td'] * scoring_settings.fum_rec_td
        
        # Kicking
        if 'fgm_0_19' in stats:
            points += stats['fgm_0_19'] * scoring_settings.fg_0_39
        if 'fgm_20_29' in stats:
            points += stats['fgm_20_29'] * scoring_settings.fg_0_39
        if 'fgm_30_39' in stats:
            points += stats['fgm_30_39'] * scoring_settings.fg_0_39
        if 'fgm_40_49' in stats:
            points += stats['fgm_40_49'] * scoring_settings.fg_40_49
        if 'fgm_50p' in stats:
            points += stats['fgm_50p'] * scoring_settings.fg_50_plus
        if 'fgmiss' in stats:
            points += stats['fgmiss'] * scoring_settings.fg_miss
        if 'xpm' in stats:
            points += stats['xpm'] * scoring_settings.xp_made
        if 'xpmiss' in stats:
            points += stats['xpmiss'] * scoring_settings.xp_miss
        
        return round(points, 2)
    
    def calculate_all_formats(
        self,
        stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate points for all standard formats"""
        return {
            'standard': self.calculate_points(stats, self.scoring_formats['standard']),
            'ppr': self.calculate_points(stats, self.scoring_formats['ppr']),
            'half_ppr': self.calculate_points(stats, self.scoring_formats['half_ppr'])
        }
    
    def calculate_player_week(
        self,
        player_id: str,
        season: int,
        week: int,
        scoring_format: str = 'ppr'
    ) -> Optional[float]:
        """Calculate points for a specific player/week with caching"""
        # Check cache first
        cache_key = f"scoring:{player_id}:{season}:{week}:{scoring_format}"
        
        if self.cache_enabled:
            cached = self.redis_client.get(cache_key)
            if cached:
                return float(cached)
        
        # Calculate from database
        with self.SessionLocal() as db:
            player_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week == week
            ).first()
            
            if not player_stats or not player_stats.stats:
                return None
            
            # Get scoring settings
            if scoring_format in self.scoring_formats:
                settings = self.scoring_formats[scoring_format]
            else:
                # Try to parse as custom settings
                try:
                    custom_settings = json.loads(scoring_format)
                    settings = ScoringSettings.custom(custom_settings)
                except:
                    settings = self.scoring_formats['ppr']
            
            # Calculate points
            points = self.calculate_points(player_stats.stats, settings)
            
            # Cache result (expire in 1 hour)
            if self.cache_enabled:
                self.redis_client.setex(cache_key, 3600, str(points))
            
            return points
    
    def calculate_season_total(
        self,
        player_id: str,
        season: int,
        scoring_format: str = 'ppr',
        weeks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate season totals with weekly breakdown"""
        with self.SessionLocal() as db:
            query = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season
            )
            
            if weeks:
                query = query.filter(PlayerStats.week.in_(weeks))
            
            player_stats = query.all()
        
        if not player_stats:
            return {
                'total': 0.0,
                'games': 0,
                'average': 0.0,
                'weekly': {}
            }
        
        weekly_scores = {}
        total = 0.0
        
        for stat in player_stats:
            if stat.stats:
                points = self.calculate_player_week(
                    player_id, season, stat.week, scoring_format
                )
                if points is not None:
                    weekly_scores[stat.week] = points
                    total += points
        
        games = len(weekly_scores)
        average = total / games if games > 0 else 0.0
        
        return {
            'total': round(total, 2),
            'games': games,
            'average': round(average, 2),
            'weekly': weekly_scores,
            'high': max(weekly_scores.values()) if weekly_scores else 0.0,
            'low': min(weekly_scores.values()) if weekly_scores else 0.0
        }
    
    def compare_formats(
        self,
        player_id: str,
        season: int,
        week: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compare player value across different scoring formats"""
        if week:
            # Single week comparison
            with self.SessionLocal() as db:
                player_stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season,
                    PlayerStats.week == week
                ).first()
                
                if not player_stats or not player_stats.stats:
                    return {}
                
                return self.calculate_all_formats(player_stats.stats)
        else:
            # Season comparison
            results = {}
            for format_name in ['standard', 'ppr', 'half_ppr']:
                season_data = self.calculate_season_total(
                    player_id, season, format_name
                )
                results[format_name] = {
                    'total': season_data['total'],
                    'average': season_data['average'],
                    'games': season_data['games']
                }
            
            return results
    
    def get_custom_settings(
        self,
        league_id: str
    ) -> Optional[ScoringSettings]:
        """Retrieve custom scoring settings for a league"""
        # In production, this would fetch from database
        # For now, return None to use default
        return None
    
    def cache_weekly_scores(
        self,
        season: int,
        week: int
    ):
        """Pre-calculate and cache scores for all players in a week"""
        if not self.cache_enabled:
            return
        
        with self.SessionLocal() as db:
            all_stats = db.query(PlayerStats).filter(
                PlayerStats.season == season,
                PlayerStats.week == week
            ).all()
        
        cached_count = 0
        for stat in all_stats:
            if stat.stats:
                for format_name in ['standard', 'ppr', 'half_ppr']:
                    self.calculate_player_week(
                        stat.player_id, season, week, format_name
                    )
                    cached_count += 1
        
        logger.info(f"Cached {cached_count} scores for season {season}, week {week}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cached scores"""
        if not self.cache_enabled:
            return
        
        if pattern:
            keys = self.redis_client.keys(f"scoring:{pattern}*")
        else:
            keys = self.redis_client.keys("scoring:*")
        
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached scores")


# Example usage
if __name__ == "__main__":
    engine = ScoringEngine()
    
    # Test with a player
    player_id = "6783"
    season = 2023
    week = 10
    
    # Calculate for all formats
    print(f"Calculating scores for player {player_id}, season {season}, week {week}")
    
    for format_name in ['standard', 'ppr', 'half_ppr']:
        points = engine.calculate_player_week(player_id, season, week, format_name)
        if points is not None:
            print(f"{format_name}: {points} points")
    
    # Season totals
    print(f"\nSeason totals for {season}:")
    season_data = engine.calculate_season_total(player_id, season, 'ppr')
    print(f"Total: {season_data['total']} points")
    print(f"Average: {season_data['average']} points/game")
    print(f"Games: {season_data['games']}")
    
    # Format comparison
    print(f"\nFormat comparison for full season:")
    comparison = engine.compare_formats(player_id, season)
    for format_name, data in comparison.items():
        print(f"{format_name}: {data['average']} ppg")