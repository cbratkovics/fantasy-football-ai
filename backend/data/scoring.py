"""
Fantasy Football Scoring Calculator
Supports Standard, PPR, Half-PPR, and Custom Scoring Systems
Handles all positions: QB, RB, WR, TE, K, DEF
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import json


@dataclass
class ScoringSettings:
    """
    Comprehensive scoring settings for all positions
    Default values match standard ESPN/Yahoo scoring
    """
    # Passing
    passing_yards_per_point: float = 25.0
    passing_td: float = 4.0
    passing_int: float = -2.0
    passing_2pt: float = 2.0
    
    # Rushing
    rushing_yards_per_point: float = 10.0
    rushing_td: float = 6.0
    rushing_2pt: float = 2.0
    
    # Receiving
    receiving_yards_per_point: float = 10.0
    receiving_td: float = 6.0
    receiving_2pt: float = 2.0
    reception: float = 0.0  # 0 for standard, 1 for PPR, 0.5 for half-PPR
    
    # Turnovers
    fumble_lost: float = -2.0
    fumble_recovered_td: float = 6.0
    
    # Kicking
    fg_made_0_19: float = 3.0
    fg_made_20_29: float = 3.0
    fg_made_30_39: float = 3.0
    fg_made_40_49: float = 4.0
    fg_made_50_plus: float = 5.0
    fg_missed: float = -1.0
    pat_made: float = 1.0
    pat_missed: float = -1.0
    
    # Defense/Special Teams
    dst_sack: float = 1.0
    dst_int: float = 2.0
    dst_fumble_rec: float = 2.0
    dst_fumble_forced: float = 1.0
    dst_safety: float = 2.0
    dst_td: float = 6.0
    dst_blocked_kick: float = 2.0
    dst_return_td: float = 6.0
    
    # Points allowed (DST)
    dst_points_allowed_0: float = 10.0
    dst_points_allowed_1_6: float = 7.0
    dst_points_allowed_7_13: float = 4.0
    dst_points_allowed_14_20: float = 1.0
    dst_points_allowed_21_27: float = 0.0
    dst_points_allowed_28_34: float = -1.0
    dst_points_allowed_35_plus: float = -4.0
    
    # Yards allowed (DST)
    dst_yards_allowed_0_99: float = 10.0
    dst_yards_allowed_100_199: float = 5.0
    dst_yards_allowed_200_299: float = 3.0
    dst_yards_allowed_300_399: float = 0.0
    dst_yards_allowed_400_449: float = -1.0
    dst_yards_allowed_450_499: float = -3.0
    dst_yards_allowed_500_plus: float = -5.0
    
    # Bonuses (optional)
    passing_300_yard_bonus: float = 0.0
    passing_400_yard_bonus: float = 0.0
    rushing_100_yard_bonus: float = 0.0
    rushing_200_yard_bonus: float = 0.0
    receiving_100_yard_bonus: float = 0.0
    receiving_200_yard_bonus: float = 0.0
    
    @classmethod
    def standard(cls) -> 'ScoringSettings':
        """Standard scoring (no PPR)"""
        return cls()
    
    @classmethod
    def ppr(cls) -> 'ScoringSettings':
        """Full PPR scoring"""
        return cls(reception=1.0)
    
    @classmethod
    def half_ppr(cls) -> 'ScoringSettings':
        """Half PPR scoring"""
        return cls(reception=0.5)
    
    @classmethod
    def from_dict(cls, settings: Dict[str, float]) -> 'ScoringSettings':
        """Create from dictionary (for custom leagues)"""
        return cls(**settings)


class FantasyScorer:
    """
    Calculate fantasy points from raw NFL statistics
    Handles all positions and scoring formats
    """
    
    def __init__(self, settings: Optional[ScoringSettings] = None):
        """
        Initialize scorer with settings
        
        Args:
            settings: Scoring settings (default: standard scoring)
        """
        self.settings = settings or ScoringSettings.standard()
    
    def calculate_points(
        self, 
        stats: Dict[str, Any], 
        position: str
    ) -> Decimal:
        """
        Calculate fantasy points for a player
        
        Args:
            stats: Raw statistics dictionary
            position: Player position (QB, RB, WR, TE, K, DEF)
            
        Returns:
            Fantasy points as Decimal for precision
        """
        points = Decimal('0')
        
        # Quarterback, Running Back, Wide Receiver, Tight End
        if position in ['QB', 'RB', 'WR', 'TE']:
            points += self._calculate_offensive_points(stats)
        
        # Kicker
        elif position == 'K':
            points += self._calculate_kicker_points(stats)
        
        # Defense/Special Teams
        elif position in ['DEF', 'DST']:
            points += self._calculate_defense_points(stats)
        
        # Round to 2 decimal places
        return points.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _calculate_offensive_points(self, stats: Dict[str, Any]) -> Decimal:
        """Calculate points for offensive players"""
        points = Decimal('0')
        
        # Passing
        passing_yards = Decimal(str(stats.get('passing_yards', 0)))
        if passing_yards > 0:
            points += passing_yards / Decimal(str(self.settings.passing_yards_per_point))
        
        points += Decimal(str(stats.get('passing_tds', 0))) * Decimal(str(self.settings.passing_td))
        points += Decimal(str(stats.get('passing_int', 0))) * Decimal(str(self.settings.passing_int))
        points += Decimal(str(stats.get('passing_2pt', 0))) * Decimal(str(self.settings.passing_2pt))
        
        # Rushing
        rushing_yards = Decimal(str(stats.get('rushing_yards', 0)))
        if rushing_yards > 0:
            points += rushing_yards / Decimal(str(self.settings.rushing_yards_per_point))
        
        points += Decimal(str(stats.get('rushing_tds', 0))) * Decimal(str(self.settings.rushing_td))
        points += Decimal(str(stats.get('rushing_2pt', 0))) * Decimal(str(self.settings.rushing_2pt))
        
        # Receiving
        receiving_yards = Decimal(str(stats.get('receiving_yards', 0)))
        if receiving_yards > 0:
            points += receiving_yards / Decimal(str(self.settings.receiving_yards_per_point))
        
        points += Decimal(str(stats.get('receiving_tds', 0))) * Decimal(str(self.settings.receiving_td))
        points += Decimal(str(stats.get('receiving_2pt', 0))) * Decimal(str(self.settings.receiving_2pt))
        points += Decimal(str(stats.get('receptions', 0))) * Decimal(str(self.settings.reception))
        
        # Turnovers
        points += Decimal(str(stats.get('fumbles_lost', 0))) * Decimal(str(self.settings.fumble_lost))
        points += Decimal(str(stats.get('fumble_rec_tds', 0))) * Decimal(str(self.settings.fumble_recovered_td))
        
        # Bonuses
        if self.settings.passing_300_yard_bonus and passing_yards >= 300:
            points += Decimal(str(self.settings.passing_300_yard_bonus))
        if self.settings.passing_400_yard_bonus and passing_yards >= 400:
            points += Decimal(str(self.settings.passing_400_yard_bonus))
        
        if self.settings.rushing_100_yard_bonus and rushing_yards >= 100:
            points += Decimal(str(self.settings.rushing_100_yard_bonus))
        if self.settings.rushing_200_yard_bonus and rushing_yards >= 200:
            points += Decimal(str(self.settings.rushing_200_yard_bonus))
        
        if self.settings.receiving_100_yard_bonus and receiving_yards >= 100:
            points += Decimal(str(self.settings.receiving_100_yard_bonus))
        if self.settings.receiving_200_yard_bonus and receiving_yards >= 200:
            points += Decimal(str(self.settings.receiving_200_yard_bonus))
        
        return points
    
    def _calculate_kicker_points(self, stats: Dict[str, Any]) -> Decimal:
        """Calculate points for kickers"""
        points = Decimal('0')
        
        # Field goals by distance
        fg_0_19 = stats.get('fg_made_0_19', 0)
        fg_20_29 = stats.get('fg_made_20_29', 0)
        fg_30_39 = stats.get('fg_made_30_39', 0)
        fg_40_49 = stats.get('fg_made_40_49', 0)
        fg_50_plus = stats.get('fg_made_50_plus', 0)
        
        points += Decimal(str(fg_0_19)) * Decimal(str(self.settings.fg_made_0_19))
        points += Decimal(str(fg_20_29)) * Decimal(str(self.settings.fg_made_20_29))
        points += Decimal(str(fg_30_39)) * Decimal(str(self.settings.fg_made_30_39))
        points += Decimal(str(fg_40_49)) * Decimal(str(self.settings.fg_made_40_49))
        points += Decimal(str(fg_50_plus)) * Decimal(str(self.settings.fg_made_50_plus))
        
        # Field goals missed
        points += Decimal(str(stats.get('fg_missed', 0))) * Decimal(str(self.settings.fg_missed))
        
        # Extra points
        points += Decimal(str(stats.get('pat_made', 0))) * Decimal(str(self.settings.pat_made))
        points += Decimal(str(stats.get('pat_missed', 0))) * Decimal(str(self.settings.pat_missed))
        
        return points
    
    def _calculate_defense_points(self, stats: Dict[str, Any]) -> Decimal:
        """Calculate points for defense/special teams"""
        points = Decimal('0')
        
        # Defensive stats
        points += Decimal(str(stats.get('sacks', 0))) * Decimal(str(self.settings.dst_sack))
        points += Decimal(str(stats.get('interceptions', 0))) * Decimal(str(self.settings.dst_int))
        points += Decimal(str(stats.get('fumble_recoveries', 0))) * Decimal(str(self.settings.dst_fumble_rec))
        points += Decimal(str(stats.get('forced_fumbles', 0))) * Decimal(str(self.settings.dst_fumble_forced))
        points += Decimal(str(stats.get('safeties', 0))) * Decimal(str(self.settings.dst_safety))
        points += Decimal(str(stats.get('defensive_tds', 0))) * Decimal(str(self.settings.dst_td))
        points += Decimal(str(stats.get('blocked_kicks', 0))) * Decimal(str(self.settings.dst_blocked_kick))
        points += Decimal(str(stats.get('return_tds', 0))) * Decimal(str(self.settings.dst_return_td))
        
        # Points allowed
        points_allowed = stats.get('points_allowed', 0)
        if points_allowed == 0:
            points += Decimal(str(self.settings.dst_points_allowed_0))
        elif points_allowed <= 6:
            points += Decimal(str(self.settings.dst_points_allowed_1_6))
        elif points_allowed <= 13:
            points += Decimal(str(self.settings.dst_points_allowed_7_13))
        elif points_allowed <= 20:
            points += Decimal(str(self.settings.dst_points_allowed_14_20))
        elif points_allowed <= 27:
            points += Decimal(str(self.settings.dst_points_allowed_21_27))
        elif points_allowed <= 34:
            points += Decimal(str(self.settings.dst_points_allowed_28_34))
        else:
            points += Decimal(str(self.settings.dst_points_allowed_35_plus))
        
        # Yards allowed
        yards_allowed = stats.get('yards_allowed', 0)
        if yards_allowed < 100:
            points += Decimal(str(self.settings.dst_yards_allowed_0_99))
        elif yards_allowed < 200:
            points += Decimal(str(self.settings.dst_yards_allowed_100_199))
        elif yards_allowed < 300:
            points += Decimal(str(self.settings.dst_yards_allowed_200_299))
        elif yards_allowed < 400:
            points += Decimal(str(self.settings.dst_yards_allowed_300_399))
        elif yards_allowed < 450:
            points += Decimal(str(self.settings.dst_yards_allowed_400_449))
        elif yards_allowed < 500:
            points += Decimal(str(self.settings.dst_yards_allowed_450_499))
        else:
            points += Decimal(str(self.settings.dst_yards_allowed_500_plus))
        
        return points
    
    def calculate_season_totals(
        self,
        weekly_stats: list[Dict[str, Any]],
        position: str
    ) -> Dict[str, Decimal]:
        """
        Calculate season totals and averages
        
        Args:
            weekly_stats: List of weekly stat dictionaries
            position: Player position
            
        Returns:
            Dictionary with total points, average, std dev, etc.
        """
        weekly_points = [
            self.calculate_points(week, position) 
            for week in weekly_stats
        ]
        
        if not weekly_points:
            return {
                'total': Decimal('0'),
                'average': Decimal('0'),
                'games': 0
            }
        
        total = sum(weekly_points)
        games = len(weekly_points)
        average = total / games
        
        # Calculate standard deviation
        variance = sum((p - average) ** 2 for p in weekly_points) / games
        std_dev = variance.sqrt()
        
        return {
            'total': total,
            'average': average.quantize(Decimal('0.01')),
            'std_dev': std_dev.quantize(Decimal('0.01')),
            'games': games,
            'min': min(weekly_points),
            'max': max(weekly_points),
            'consistency_score': (average / (std_dev + 1)).quantize(Decimal('0.01'))
        }


# Example usage and testing
def example_usage():
    """Demonstrate scoring calculations"""
    
    # Example QB stats
    qb_stats = {
        'passing_yards': 285,
        'passing_tds': 2,
        'passing_int': 1,
        'rushing_yards': 25,
        'rushing_tds': 0,
        'fumbles_lost': 0
    }
    
    # Example RB stats
    rb_stats = {
        'rushing_yards': 95,
        'rushing_tds': 1,
        'receptions': 4,
        'receiving_yards': 35,
        'receiving_tds': 0,
        'fumbles_lost': 0
    }
    
    # Calculate for different scoring systems
    scorers = {
        'Standard': FantasyScorer(ScoringSettings.standard()),
        'PPR': FantasyScorer(ScoringSettings.ppr()),
        'Half-PPR': FantasyScorer(ScoringSettings.half_ppr())
    }
    
    print("QB Stats:", qb_stats)
    for name, scorer in scorers.items():
        points = scorer.calculate_points(qb_stats, 'QB')
        print(f"{name}: {points} points")
    
    print("\nRB Stats:", rb_stats)
    for name, scorer in scorers.items():
        points = scorer.calculate_points(rb_stats, 'RB')
        print(f"{name}: {points} points")
    
    # Custom scoring example
    custom_settings = ScoringSettings(
        passing_td=6.0,  # 6 point passing TDs
        reception=0.5,   # Half PPR
        passing_300_yard_bonus=3.0  # Bonus points
    )
    custom_scorer = FantasyScorer(custom_settings)
    print(f"\nCustom scoring QB: {custom_scorer.calculate_points(qb_stats, 'QB')} points")


if __name__ == "__main__":
    example_usage()