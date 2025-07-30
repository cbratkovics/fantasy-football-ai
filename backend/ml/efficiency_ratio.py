"""
Efficiency Ratio Calculator - Proprietary Metric for Fantasy Football AI
Measures how efficiently a player converts opportunities into fantasy production
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats
from backend.ml.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)

# Database connection
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics"""
    overall_efficiency: float
    opportunity_efficiency: float
    matchup_efficiency: float
    game_script_efficiency: float
    red_zone_efficiency: Optional[float]
    third_down_efficiency: Optional[float]
    target_share_efficiency: Optional[float]
    components: Dict[str, float]
    percentile_rank: float
    comparison_to_position: float


class EfficiencyRatioCalculator:
    """
    Calculate proprietary Efficiency Ratio that measures:
    1. How well a player converts opportunities into points
    2. Performance relative to game conditions
    3. Matchup-adjusted production efficiency
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.scoring_engine = ScoringEngine()
        
        # Position-specific efficiency thresholds
        self.position_baselines = {
            'QB': {
                'yards_per_attempt': 7.0,
                'td_rate': 0.05,
                'completion_rate': 0.62,
                'yards_per_game': 250
            },
            'RB': {
                'yards_per_carry': 4.2,
                'yards_per_reception': 7.5,
                'td_rate': 0.06,
                'catch_rate': 0.75
            },
            'WR': {
                'yards_per_reception': 12.0,
                'yards_per_target': 8.0,
                'td_rate': 0.08,
                'catch_rate': 0.65
            },
            'TE': {
                'yards_per_reception': 10.5,
                'yards_per_target': 7.0,
                'td_rate': 0.07,
                'catch_rate': 0.70
            }
        }
    
    def calculate_player_efficiency(
        self,
        player_id: str,
        season: int,
        weeks: Optional[List[int]] = None,
        include_components: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive efficiency ratio for a player
        
        Returns efficiency metrics and detailed breakdown
        """
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Get player stats
            query = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season
            )
            
            if weeks:
                query = query.filter(PlayerStats.week.in_(weeks))
            
            stats_list = query.all()
        
        if not stats_list:
            return {"error": "No stats found for player"}
        
        # Calculate position-specific efficiency
        if player.position == 'QB':
            efficiency = self._calculate_qb_efficiency(stats_list)
        elif player.position == 'RB':
            efficiency = self._calculate_rb_efficiency(stats_list)
        elif player.position in ['WR', 'TE']:
            efficiency = self._calculate_receiver_efficiency(stats_list, player.position)
        else:
            return {"error": f"Position {player.position} not supported"}
        
        # Calculate percentile rank among position
        percentile = self._calculate_percentile_rank(
            efficiency.overall_efficiency,
            player.position,
            season
        )
        
        # Build response
        response = {
            "player_id": player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "season": season,
            "games_analyzed": len(stats_list),
            "efficiency_ratio": round(efficiency.overall_efficiency, 3),
            "percentile_rank": round(percentile, 1),
            "efficiency_grade": self._get_efficiency_grade(efficiency.overall_efficiency),
            "comparison_to_average": f"{efficiency.comparison_to_position:+.1%}"
        }
        
        if include_components:
            response["components"] = {
                "opportunity_efficiency": round(efficiency.opportunity_efficiency, 3),
                "matchup_efficiency": round(efficiency.matchup_efficiency, 3),
                "game_script_efficiency": round(efficiency.game_script_efficiency, 3),
                "red_zone_efficiency": round(efficiency.red_zone_efficiency, 3) if efficiency.red_zone_efficiency else None,
                "detailed_metrics": efficiency.components
            }
            
            response["insights"] = self._generate_efficiency_insights(efficiency, player.position)
        
        return response
    
    def _calculate_qb_efficiency(self, stats_list: List[PlayerStats]) -> EfficiencyMetrics:
        """Calculate QB-specific efficiency metrics"""
        total_games = len(stats_list)
        
        # Aggregate stats
        total_attempts = 0
        total_completions = 0
        total_yards = 0
        total_tds = 0
        total_ints = 0
        total_rush_yards = 0
        total_rush_tds = 0
        total_fantasy_points = 0
        
        for stat in stats_list:
            if stat.stats:
                total_attempts += stat.stats.get('pass_att', 0)
                total_completions += stat.stats.get('pass_cmp', 0)
                total_yards += stat.stats.get('pass_yd', 0)
                total_tds += stat.stats.get('pass_td', 0)
                total_ints += stat.stats.get('pass_int', 0)
                total_rush_yards += stat.stats.get('rush_yd', 0)
                total_rush_tds += stat.stats.get('rush_td', 0)
                
                # Calculate fantasy points
                fp = self.scoring_engine.calculate_points(
                    stat.stats,
                    self.scoring_engine.scoring_formats['ppr']
                )
                total_fantasy_points += fp
        
        # Calculate efficiency metrics
        yards_per_attempt = total_yards / total_attempts if total_attempts > 0 else 0
        completion_rate = total_completions / total_attempts if total_attempts > 0 else 0
        td_rate = total_tds / total_attempts if total_attempts > 0 else 0
        int_rate = total_ints / total_attempts if total_attempts > 0 else 0
        
        # Opportunity efficiency
        baseline = self.position_baselines['QB']
        opp_efficiency = (
            (yards_per_attempt / baseline['yards_per_attempt']) * 0.3 +
            (td_rate / baseline['td_rate']) * 0.3 +
            (completion_rate / baseline['completion_rate']) * 0.2 +
            (1 - int_rate / 0.025) * 0.2  # Lower is better for INT rate
        )
        
        # Game script efficiency (rushing contribution)
        rush_efficiency = (total_rush_yards + total_rush_tds * 60) / total_games / 15  # 15 rush yards/game is good
        
        # Matchup efficiency (simplified - would use opponent data in production)
        matchup_efficiency = total_fantasy_points / total_games / 20  # 20 PPG is baseline
        
        # Overall efficiency
        overall = (
            opp_efficiency * 0.5 +
            matchup_efficiency * 0.3 +
            rush_efficiency * 0.2
        )
        
        return EfficiencyMetrics(
            overall_efficiency=overall,
            opportunity_efficiency=opp_efficiency,
            matchup_efficiency=matchup_efficiency,
            game_script_efficiency=rush_efficiency,
            red_zone_efficiency=None,  # Would calculate if we had red zone data
            third_down_efficiency=None,
            target_share_efficiency=None,
            components={
                'yards_per_attempt': round(yards_per_attempt, 2),
                'completion_rate': round(completion_rate, 3),
                'td_rate': round(td_rate, 3),
                'int_rate': round(int_rate, 3),
                'rush_yards_per_game': round(total_rush_yards / total_games, 1),
                'fantasy_ppg': round(total_fantasy_points / total_games, 1)
            },
            percentile_rank=0,  # Calculated separately
            comparison_to_position=overall - 1.0
        )
    
    def _calculate_rb_efficiency(self, stats_list: List[PlayerStats]) -> EfficiencyMetrics:
        """Calculate RB-specific efficiency metrics"""
        total_games = len(stats_list)
        
        # Aggregate stats
        total_rush_att = 0
        total_rush_yards = 0
        total_rush_tds = 0
        total_rec = 0
        total_rec_yards = 0
        total_rec_tds = 0
        total_targets = 0
        total_fantasy_points = 0
        
        for stat in stats_list:
            if stat.stats:
                total_rush_att += stat.stats.get('rush_att', 0)
                total_rush_yards += stat.stats.get('rush_yd', 0)
                total_rush_tds += stat.stats.get('rush_td', 0)
                total_rec += stat.stats.get('rec', 0)
                total_rec_yards += stat.stats.get('rec_yd', 0)
                total_rec_tds += stat.stats.get('rec_td', 0)
                total_targets += stat.stats.get('rec_tgt', 0)
                
                fp = self.scoring_engine.calculate_points(
                    stat.stats,
                    self.scoring_engine.scoring_formats['ppr']
                )
                total_fantasy_points += fp
        
        # Calculate efficiency metrics
        yards_per_carry = total_rush_yards / total_rush_att if total_rush_att > 0 else 0
        yards_per_reception = total_rec_yards / total_rec if total_rec > 0 else 0
        catch_rate = total_rec / total_targets if total_targets > 0 else 0
        td_rate = (total_rush_tds + total_rec_tds) / (total_rush_att + total_targets) if (total_rush_att + total_targets) > 0 else 0
        
        # Opportunity efficiency
        baseline = self.position_baselines['RB']
        opp_efficiency = (
            (yards_per_carry / baseline['yards_per_carry']) * 0.4 +
            (yards_per_reception / baseline['yards_per_reception']) * 0.2 +
            (catch_rate / baseline['catch_rate']) * 0.2 +
            (td_rate / baseline['td_rate']) * 0.2
        )
        
        # Target share efficiency
        target_efficiency = total_targets / total_games / 4  # 4 targets/game is good for RB
        
        # Matchup efficiency
        matchup_efficiency = total_fantasy_points / total_games / 15  # 15 PPG baseline for RB
        
        # Overall efficiency
        overall = (
            opp_efficiency * 0.5 +
            matchup_efficiency * 0.3 +
            target_efficiency * 0.2
        )
        
        return EfficiencyMetrics(
            overall_efficiency=overall,
            opportunity_efficiency=opp_efficiency,
            matchup_efficiency=matchup_efficiency,
            game_script_efficiency=target_efficiency,
            red_zone_efficiency=None,
            third_down_efficiency=None,
            target_share_efficiency=target_efficiency,
            components={
                'yards_per_carry': round(yards_per_carry, 2),
                'yards_per_reception': round(yards_per_reception, 2),
                'catch_rate': round(catch_rate, 3),
                'td_rate': round(td_rate, 3),
                'touches_per_game': round((total_rush_att + total_rec) / total_games, 1),
                'fantasy_ppg': round(total_fantasy_points / total_games, 1)
            },
            percentile_rank=0,
            comparison_to_position=overall - 1.0
        )
    
    def _calculate_receiver_efficiency(self, stats_list: List[PlayerStats], position: str) -> EfficiencyMetrics:
        """Calculate WR/TE-specific efficiency metrics"""
        total_games = len(stats_list)
        
        # Aggregate stats
        total_rec = 0
        total_rec_yards = 0
        total_rec_tds = 0
        total_targets = 0
        total_fantasy_points = 0
        
        for stat in stats_list:
            if stat.stats:
                total_rec += stat.stats.get('rec', 0)
                total_rec_yards += stat.stats.get('rec_yd', 0)
                total_rec_tds += stat.stats.get('rec_td', 0)
                total_targets += stat.stats.get('rec_tgt', 0)
                
                fp = self.scoring_engine.calculate_points(
                    stat.stats,
                    self.scoring_engine.scoring_formats['ppr']
                )
                total_fantasy_points += fp
        
        # Calculate efficiency metrics
        yards_per_reception = total_rec_yards / total_rec if total_rec > 0 else 0
        yards_per_target = total_rec_yards / total_targets if total_targets > 0 else 0
        catch_rate = total_rec / total_targets if total_targets > 0 else 0
        td_rate = total_rec_tds / total_targets if total_targets > 0 else 0
        
        # Opportunity efficiency
        baseline = self.position_baselines[position]
        opp_efficiency = (
            (yards_per_reception / baseline['yards_per_reception']) * 0.3 +
            (yards_per_target / baseline['yards_per_target']) * 0.3 +
            (catch_rate / baseline['catch_rate']) * 0.2 +
            (td_rate / baseline['td_rate']) * 0.2
        )
        
        # Target share efficiency (targets per game)
        target_efficiency = total_targets / total_games / (8 if position == 'WR' else 6)
        
        # Matchup efficiency
        baseline_ppg = 12 if position == 'WR' else 10
        matchup_efficiency = total_fantasy_points / total_games / baseline_ppg
        
        # Overall efficiency
        overall = (
            opp_efficiency * 0.5 +
            matchup_efficiency * 0.3 +
            target_efficiency * 0.2
        )
        
        return EfficiencyMetrics(
            overall_efficiency=overall,
            opportunity_efficiency=opp_efficiency,
            matchup_efficiency=matchup_efficiency,
            game_script_efficiency=target_efficiency,
            red_zone_efficiency=None,
            third_down_efficiency=None,
            target_share_efficiency=target_efficiency,
            components={
                'yards_per_reception': round(yards_per_reception, 2),
                'yards_per_target': round(yards_per_target, 2),
                'catch_rate': round(catch_rate, 3),
                'td_rate': round(td_rate, 3),
                'targets_per_game': round(total_targets / total_games, 1),
                'fantasy_ppg': round(total_fantasy_points / total_games, 1)
            },
            percentile_rank=0,
            comparison_to_position=overall - 1.0
        )
    
    def _calculate_percentile_rank(
        self,
        efficiency_score: float,
        position: str,
        season: int
    ) -> float:
        """Calculate percentile rank among position"""
        # In production, this would query all players at position
        # For now, use approximation based on score
        if efficiency_score >= 1.5:
            return 95.0
        elif efficiency_score >= 1.3:
            return 85.0
        elif efficiency_score >= 1.1:
            return 70.0
        elif efficiency_score >= 1.0:
            return 50.0
        elif efficiency_score >= 0.9:
            return 30.0
        elif efficiency_score >= 0.8:
            return 15.0
        else:
            return 5.0
    
    def _get_efficiency_grade(self, efficiency_score: float) -> str:
        """Convert efficiency score to letter grade"""
        if efficiency_score >= 1.4:
            return "A+"
        elif efficiency_score >= 1.3:
            return "A"
        elif efficiency_score >= 1.2:
            return "A-"
        elif efficiency_score >= 1.15:
            return "B+"
        elif efficiency_score >= 1.1:
            return "B"
        elif efficiency_score >= 1.05:
            return "B-"
        elif efficiency_score >= 1.0:
            return "C+"
        elif efficiency_score >= 0.95:
            return "C"
        elif efficiency_score >= 0.9:
            return "C-"
        elif efficiency_score >= 0.85:
            return "D+"
        elif efficiency_score >= 0.8:
            return "D"
        else:
            return "F"
    
    def _generate_efficiency_insights(
        self,
        metrics: EfficiencyMetrics,
        position: str
    ) -> List[str]:
        """Generate insights based on efficiency metrics"""
        insights = []
        
        # Overall efficiency insight
        if metrics.overall_efficiency >= 1.3:
            insights.append("Elite efficiency - consistently outperforms expectations")
        elif metrics.overall_efficiency >= 1.1:
            insights.append("Above-average efficiency in converting opportunities")
        elif metrics.overall_efficiency >= 0.9:
            insights.append("Average efficiency relative to opportunities")
        else:
            insights.append("Below-average efficiency - underperforming relative to opportunities")
        
        # Opportunity efficiency
        if metrics.opportunity_efficiency >= 1.2:
            insights.append("Maximizing opportunities with excellent per-touch production")
        elif metrics.opportunity_efficiency <= 0.8:
            insights.append("Struggling to convert opportunities into production")
        
        # Position-specific insights
        if position == 'QB' and 'td_rate' in metrics.components:
            if metrics.components['td_rate'] >= 0.06:
                insights.append("Excellent touchdown efficiency in the red zone")
            elif metrics.components['td_rate'] <= 0.03:
                insights.append("Low touchdown rate despite passing volume")
        
        elif position == 'RB' and 'yards_per_carry' in metrics.components:
            if metrics.components['yards_per_carry'] >= 5.0:
                insights.append("Explosive runner averaging 5+ yards per carry")
            elif metrics.components['yards_per_carry'] <= 3.5:
                insights.append("Inefficient rushing - possible offensive line issues")
        
        elif position in ['WR', 'TE'] and 'catch_rate' in metrics.components:
            if metrics.components['catch_rate'] >= 0.75:
                insights.append("Highly reliable target with excellent hands")
            elif metrics.components['catch_rate'] <= 0.55:
                insights.append("Struggling with drops or difficult targets")
        
        # Target share insight for receivers
        if metrics.target_share_efficiency and position in ['WR', 'TE']:
            if metrics.target_share_efficiency >= 1.2:
                insights.append("Heavy target share indicates strong QB trust")
            elif metrics.target_share_efficiency <= 0.7:
                insights.append("Limited target share caps upside potential")
        
        return insights
    
    def calculate_weekly_efficiency_trend(
        self,
        player_id: str,
        season: int,
        last_n_weeks: int = 5
    ) -> Dict[str, Any]:
        """Calculate efficiency trend over recent weeks"""
        with self.SessionLocal() as db:
            # Get recent stats
            recent_stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season
            ).order_by(PlayerStats.week.desc()).limit(last_n_weeks).all()
        
        if not recent_stats:
            return {"error": "No recent stats found"}
        
        # Calculate efficiency for each week
        weekly_efficiencies = []
        
        for stat in reversed(recent_stats):  # Process in chronological order
            week_eff = self.calculate_player_efficiency(
                player_id=player_id,
                season=season,
                weeks=[stat.week],
                include_components=False
            )
            
            if "error" not in week_eff:
                weekly_efficiencies.append({
                    'week': stat.week,
                    'efficiency': week_eff['efficiency_ratio'],
                    'grade': week_eff['efficiency_grade']
                })
        
        if not weekly_efficiencies:
            return {"error": "Could not calculate weekly efficiencies"}
        
        # Calculate trend
        efficiencies = [w['efficiency'] for w in weekly_efficiencies]
        avg_efficiency = np.mean(efficiencies)
        
        # Simple linear regression for trend
        weeks = np.array([w['week'] for w in weekly_efficiencies])
        eff_values = np.array(efficiencies)
        
        if len(weeks) > 1:
            slope = np.polyfit(weeks, eff_values, 1)[0]
            trend = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
        else:
            slope = 0
            trend = "insufficient_data"
        
        return {
            "player_id": player_id,
            "season": season,
            "weekly_efficiencies": weekly_efficiencies,
            "average_efficiency": round(avg_efficiency, 3),
            "trend": trend,
            "trend_slope": round(slope, 4),
            "recent_direction": "up" if len(efficiencies) > 1 and efficiencies[-1] > efficiencies[-2] else "down"
        }
    
    def get_position_efficiency_rankings(
        self,
        position: str,
        season: int,
        min_games: int = 5
    ) -> List[Dict[str, Any]]:
        """Get efficiency rankings for all players at a position"""
        with self.SessionLocal() as db:
            # Get all players at position with sufficient games
            players = db.query(Player).filter(
                Player.position == position
            ).all()
        
        rankings = []
        
        for player in players:
            # Check if player has enough games
            with self.SessionLocal() as db:
                game_count = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player.player_id,
                    PlayerStats.season == season
                ).count()
            
            if game_count >= min_games:
                efficiency = self.calculate_player_efficiency(
                    player.player_id,
                    season,
                    include_components=False
                )
                
                if "error" not in efficiency:
                    rankings.append({
                        'player_id': player.player_id,
                        'player_name': efficiency['player_name'],
                        'team': player.team,
                        'efficiency_ratio': efficiency['efficiency_ratio'],
                        'grade': efficiency['efficiency_grade'],
                        'games': efficiency['games_analyzed']
                    })
        
        # Sort by efficiency ratio
        rankings.sort(key=lambda x: x['efficiency_ratio'], reverse=True)
        
        # Add rank
        for i, player in enumerate(rankings):
            player['rank'] = i + 1
        
        return rankings[:50]  # Top 50 players


# Example usage
if __name__ == "__main__":
    calculator = EfficiencyRatioCalculator()
    
    # Test with a player
    result = calculator.calculate_player_efficiency(
        player_id="6783",
        season=2023,
        include_components=True
    )
    
    if "error" not in result:
        print(f"Player: {result['player_name']}")
        print(f"Efficiency Ratio: {result['efficiency_ratio']}")
        print(f"Grade: {result['efficiency_grade']}")
        print(f"Percentile: {result['percentile_rank']}%")
        print(f"\nComponents:")
        for key, value in result['components'].items():
            print(f"  {key}: {value}")
        print(f"\nInsights:")
        for insight in result['insights']:
            print(f"  - {insight}")
    else:
        print(f"Error: {result['error']}")