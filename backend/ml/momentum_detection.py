"""
Momentum Detection System with Rolling Averages
Identifies player performance trends and hot/cold streaks
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from scipy import stats

from backend.models.database import Player, PlayerStats
from backend.ml.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)

# Database connection
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


@dataclass
class MomentumIndicators:
    """Container for momentum analysis results"""
    # Core momentum scores
    momentum_3w: float
    momentum_5w: float
    momentum_10w: float
    
    # Trend indicators
    trend_strength: float
    trend_direction: str  # 'strong_up', 'up', 'neutral', 'down', 'strong_down'
    consistency_score: float
    
    # Statistical measures
    rolling_avg_3w: float
    rolling_avg_5w: float
    rolling_avg_10w: float
    volatility: float
    
    # Streak information
    current_streak: int  # positive for hot, negative for cold
    streak_type: str  # 'hot', 'cold', 'neutral'
    
    # Performance vs expectations
    performance_vs_avg: float
    performance_vs_tier: float
    
    # Predictive indicators
    breakout_probability: float
    regression_probability: float


class MomentumDetector:
    """
    Advanced momentum detection system that identifies:
    1. Short-term and long-term performance trends
    2. Hot and cold streaks
    3. Breakout and regression candidates
    4. Consistency patterns
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.scoring_engine = ScoringEngine()
        
        # Thresholds for momentum detection
        self.momentum_thresholds = {
            'strong_positive': 0.20,  # 20% above average
            'positive': 0.10,         # 10% above average
            'neutral_high': 0.05,     # 5% above average
            'neutral_low': -0.05,     # 5% below average
            'negative': -0.10,        # 10% below average
            'strong_negative': -0.20  # 20% below average
        }
        
        # Streak thresholds
        self.streak_thresholds = {
            'hot': 1.15,   # 15% above average
            'cold': 0.85   # 15% below average
        }
    
    def analyze_player_momentum(
        self,
        player_id: str,
        season: int,
        current_week: int,
        lookback_weeks: int = 15
    ) -> Dict[str, Any]:
        """
        Comprehensive momentum analysis for a player
        
        Returns detailed momentum indicators and predictions
        """
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Get historical stats
            stats = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < current_week
            ).order_by(PlayerStats.week.desc()).limit(lookback_weeks).all()
            
            if len(stats) < 3:
                # Try previous season
                stats = db.query(PlayerStats).filter(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season - 1
                ).order_by(PlayerStats.week.desc()).limit(lookback_weeks).all()
        
        if len(stats) < 3:
            return {"error": "Insufficient data for momentum analysis"}
        
        # Calculate momentum indicators
        momentum = self._calculate_momentum_indicators(stats, player.position)
        
        # Analyze streaks
        streak_info = self._analyze_streaks(stats)
        momentum.current_streak = streak_info['current_streak']
        momentum.streak_type = streak_info['streak_type']
        
        # Calculate breakout/regression probabilities
        probabilities = self._calculate_probabilities(momentum, stats)
        momentum.breakout_probability = probabilities['breakout']
        momentum.regression_probability = probabilities['regression']
        
        # Build response
        response = {
            "player_id": player_id,
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "season": season,
            "analysis_week": current_week,
            "games_analyzed": len(stats),
            "momentum_score": round(momentum.momentum_3w, 3),
            "trend": momentum.trend_direction,
            "indicators": {
                "momentum_3w": round(momentum.momentum_3w, 3),
                "momentum_5w": round(momentum.momentum_5w, 3),
                "momentum_10w": round(momentum.momentum_10w, 3),
                "trend_strength": round(momentum.trend_strength, 3),
                "consistency": round(momentum.consistency_score, 3),
                "volatility": round(momentum.volatility, 3)
            },
            "rolling_averages": {
                "3_week": round(momentum.rolling_avg_3w, 2),
                "5_week": round(momentum.rolling_avg_5w, 2),
                "10_week": round(momentum.rolling_avg_10w, 2)
            },
            "streak": {
                "current": momentum.current_streak,
                "type": momentum.streak_type,
                "description": self._get_streak_description(momentum.current_streak)
            },
            "performance": {
                "vs_average": f"{momentum.performance_vs_avg:+.1%}",
                "vs_tier": f"{momentum.performance_vs_tier:+.1%}"
            },
            "predictions": {
                "breakout_probability": round(momentum.breakout_probability, 3),
                "regression_probability": round(momentum.regression_probability, 3),
                "recommendation": self._get_recommendation(momentum)
            },
            "insights": self._generate_insights(momentum, player.position)
        }
        
        return response
    
    def _calculate_momentum_indicators(
        self,
        stats: List[PlayerStats],
        position: str
    ) -> MomentumIndicators:
        """Calculate comprehensive momentum indicators"""
        # Extract fantasy points
        points = []
        for stat in reversed(stats):  # Chronological order
            if stat.fantasy_points_ppr is not None:
                points.append(float(stat.fantasy_points_ppr))
        
        points = np.array(points)
        n_games = len(points)
        
        # Calculate rolling averages
        rolling_3w = np.mean(points[-3:]) if n_games >= 3 else np.mean(points)
        rolling_5w = np.mean(points[-5:]) if n_games >= 5 else np.mean(points)
        rolling_10w = np.mean(points[-10:]) if n_games >= 10 else np.mean(points)
        
        # Calculate overall average (excluding most recent games for comparison)
        if n_games > 5:
            baseline_avg = np.mean(points[:-3])
        else:
            baseline_avg = np.mean(points)
        
        # Calculate momentum scores
        momentum_3w = (rolling_3w - baseline_avg) / (baseline_avg + 1e-6)
        momentum_5w = (rolling_5w - baseline_avg) / (baseline_avg + 1e-6)
        momentum_10w = (rolling_10w - baseline_avg) / (baseline_avg + 1e-6)
        
        # Calculate trend using linear regression
        if n_games >= 3:
            x = np.arange(n_games)
            slope, intercept, r_value, _, _ = stats.linregress(x, points)
            trend_strength = abs(r_value)  # How strong is the trend
            
            # Normalize slope by average points
            normalized_slope = slope / (np.mean(points) + 1e-6)
            
            # Determine trend direction
            if normalized_slope > 0.05 and trend_strength > 0.5:
                trend_direction = "strong_up"
            elif normalized_slope > 0.02:
                trend_direction = "up"
            elif normalized_slope < -0.05 and trend_strength > 0.5:
                trend_direction = "strong_down"
            elif normalized_slope < -0.02:
                trend_direction = "down"
            else:
                trend_direction = "neutral"
        else:
            trend_strength = 0
            trend_direction = "neutral"
        
        # Calculate consistency (inverse of coefficient of variation)
        consistency = 1 - (np.std(points) / (np.mean(points) + 1e-6))
        
        # Calculate volatility
        if n_games > 1:
            returns = np.diff(points) / points[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0
        
        # Performance comparisons
        performance_vs_avg = (rolling_3w - baseline_avg) / (baseline_avg + 1e-6)
        
        # Position-based tier expectations
        tier_expectations = {
            'QB': {'elite': 25, 'good': 20, 'average': 15},
            'RB': {'elite': 20, 'good': 15, 'average': 10},
            'WR': {'elite': 18, 'good': 14, 'average': 9},
            'TE': {'elite': 15, 'good': 11, 'average': 7}
        }
        
        position_tier = tier_expectations.get(position, tier_expectations['WR'])
        tier_avg = position_tier['good']
        performance_vs_tier = (rolling_3w - tier_avg) / tier_avg
        
        return MomentumIndicators(
            momentum_3w=momentum_3w,
            momentum_5w=momentum_5w,
            momentum_10w=momentum_10w,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            consistency_score=consistency,
            rolling_avg_3w=rolling_3w,
            rolling_avg_5w=rolling_5w,
            rolling_avg_10w=rolling_10w,
            volatility=volatility,
            current_streak=0,  # Set by streak analysis
            streak_type="neutral",
            performance_vs_avg=performance_vs_avg,
            performance_vs_tier=performance_vs_tier,
            breakout_probability=0,  # Set by probability calculation
            regression_probability=0
        )
    
    def _analyze_streaks(self, stats: List[PlayerStats]) -> Dict[str, Any]:
        """Analyze hot/cold streaks"""
        # Get recent games in chronological order
        recent_games = []
        for stat in reversed(stats[:10]):  # Last 10 games
            if stat.fantasy_points_ppr is not None:
                recent_games.append(float(stat.fantasy_points_ppr))
        
        if len(recent_games) < 3:
            return {"current_streak": 0, "streak_type": "neutral"}
        
        # Calculate baseline (games 4-10)
        if len(recent_games) >= 7:
            baseline = np.mean(recent_games[3:])
        else:
            baseline = np.mean(recent_games)
        
        # Analyze current streak
        current_streak = 0
        streak_type = "neutral"
        
        # Check last 3 games
        for i in range(min(3, len(recent_games))):
            game_idx = -(i + 1)
            performance_ratio = recent_games[game_idx] / (baseline + 1e-6)
            
            if performance_ratio >= self.streak_thresholds['hot']:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    break
            elif performance_ratio <= self.streak_thresholds['cold']:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    break
            else:
                break
        
        # Determine streak type
        if current_streak >= 3:
            streak_type = "hot"
        elif current_streak <= -3:
            streak_type = "cold"
        elif current_streak > 0:
            streak_type = "warm"
        elif current_streak < 0:
            streak_type = "cool"
        
        return {
            "current_streak": current_streak,
            "streak_type": streak_type
        }
    
    def _calculate_probabilities(
        self,
        momentum: MomentumIndicators,
        stats: List[PlayerStats]
    ) -> Dict[str, float]:
        """Calculate breakout and regression probabilities"""
        # Base probabilities
        breakout_prob = 0.2  # 20% base
        regression_prob = 0.2  # 20% base
        
        # Adjust based on momentum
        if momentum.momentum_3w > 0.15:
            breakout_prob += 0.2
        elif momentum.momentum_3w > 0.05:
            breakout_prob += 0.1
        elif momentum.momentum_3w < -0.15:
            regression_prob += 0.2
        elif momentum.momentum_3w < -0.05:
            regression_prob += 0.1
        
        # Adjust based on trend
        if momentum.trend_direction == "strong_up":
            breakout_prob += 0.15
            regression_prob -= 0.1
        elif momentum.trend_direction == "up":
            breakout_prob += 0.05
        elif momentum.trend_direction == "strong_down":
            regression_prob += 0.15
            breakout_prob -= 0.1
        elif momentum.trend_direction == "down":
            regression_prob += 0.05
        
        # Adjust based on consistency
        if momentum.consistency_score > 0.8:
            # Consistent players less likely to break out or regress dramatically
            breakout_prob *= 0.8
            regression_prob *= 0.8
        elif momentum.consistency_score < 0.5:
            # Volatile players more likely to have big swings
            breakout_prob *= 1.2
            regression_prob *= 1.2
        
        # Adjust based on current performance vs tier
        if momentum.performance_vs_tier > 0.3:
            # Already performing well above tier
            regression_prob += 0.1
            breakout_prob -= 0.05
        elif momentum.performance_vs_tier < -0.3:
            # Performing well below tier
            breakout_prob += 0.1
            regression_prob -= 0.05
        
        # Cap probabilities
        breakout_prob = min(0.8, max(0.05, breakout_prob))
        regression_prob = min(0.8, max(0.05, regression_prob))
        
        return {
            "breakout": breakout_prob,
            "regression": regression_prob
        }
    
    def _get_streak_description(self, streak: int) -> str:
        """Get human-readable streak description"""
        if streak >= 5:
            return f"On fire! {streak}-game hot streak"
        elif streak >= 3:
            return f"Hot streak - {streak} games above average"
        elif streak >= 1:
            return f"Warming up - {streak} game(s) above average"
        elif streak <= -5:
            return f"Ice cold! {abs(streak)}-game cold streak"
        elif streak <= -3:
            return f"Cold streak - {abs(streak)} games below average"
        elif streak <= -1:
            return f"Cooling off - {abs(streak)} game(s) below average"
        else:
            return "No significant streak"
    
    def _get_recommendation(self, momentum: MomentumIndicators) -> str:
        """Generate actionable recommendation based on momentum"""
        if momentum.breakout_probability > 0.6 and momentum.momentum_3w > 0.1:
            return "Strong Buy - High breakout potential with positive momentum"
        elif momentum.breakout_probability > 0.4 and momentum.trend_direction in ["up", "strong_up"]:
            return "Buy - Positive trends indicate upside potential"
        elif momentum.regression_probability > 0.6 and momentum.momentum_3w < -0.1:
            return "Strong Sell - High regression risk with negative momentum"
        elif momentum.regression_probability > 0.4 and momentum.trend_direction in ["down", "strong_down"]:
            return "Sell - Negative trends suggest downside risk"
        elif momentum.consistency_score > 0.8 and abs(momentum.momentum_3w) < 0.1:
            return "Hold - Consistent performer at expected levels"
        else:
            return "Monitor - Mixed signals require further observation"
    
    def _generate_insights(
        self,
        momentum: MomentumIndicators,
        position: str
    ) -> List[str]:
        """Generate insights based on momentum analysis"""
        insights = []
        
        # Momentum insight
        if momentum.momentum_3w > 0.2:
            insights.append("Significant positive momentum - performing 20%+ above baseline")
        elif momentum.momentum_3w > 0.1:
            insights.append("Positive momentum trending above recent averages")
        elif momentum.momentum_3w < -0.2:
            insights.append("Significant negative momentum - performing 20%+ below baseline")
        elif momentum.momentum_3w < -0.1:
            insights.append("Negative momentum trending below recent averages")
        
        # Trend insight
        if momentum.trend_direction == "strong_up":
            insights.append("Strong upward trend with high correlation")
        elif momentum.trend_direction == "strong_down":
            insights.append("Strong downward trend with high correlation")
        
        # Consistency insight
        if momentum.consistency_score > 0.85:
            insights.append("Highly consistent performer with low volatility")
        elif momentum.consistency_score < 0.5:
            insights.append("Volatile performer with high week-to-week variance")
        
        # Streak insight
        if momentum.current_streak >= 3:
            insights.append(f"Currently on a {momentum.current_streak}-game hot streak")
        elif momentum.current_streak <= -3:
            insights.append(f"Currently in a {abs(momentum.current_streak)}-game cold streak")
        
        # Rolling average comparison
        if momentum.rolling_avg_3w > momentum.rolling_avg_10w * 1.15:
            insights.append("Recent performance significantly exceeds season average")
        elif momentum.rolling_avg_3w < momentum.rolling_avg_10w * 0.85:
            insights.append("Recent performance significantly below season average")
        
        # Position-specific insights
        if position in ['RB', 'WR'] and momentum.volatility > 0.3:
            insights.append("High volatility typical of committee/rotational usage")
        
        return insights
    
    def batch_analyze_momentum(
        self,
        player_ids: List[str],
        season: int,
        current_week: int
    ) -> pd.DataFrame:
        """Analyze momentum for multiple players"""
        results = []
        
        for player_id in player_ids:
            try:
                analysis = self.analyze_player_momentum(player_id, season, current_week)
                if "error" not in analysis:
                    results.append({
                        'player_id': player_id,
                        'player_name': analysis['player_name'],
                        'position': analysis['position'],
                        'momentum_score': analysis['momentum_score'],
                        'trend': analysis['trend'],
                        'streak_type': analysis['streak']['type'],
                        'breakout_prob': analysis['predictions']['breakout_probability'],
                        'regression_prob': analysis['predictions']['regression_probability'],
                        'recommendation': analysis['predictions']['recommendation']
                    })
            except Exception as e:
                logger.error(f"Error analyzing momentum for {player_id}: {str(e)}")
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    detector = MomentumDetector()
    
    # Test with a player
    result = detector.analyze_player_momentum(
        player_id="6783",
        season=2024,
        current_week=10
    )
    
    if "error" not in result:
        print(f"Player: {result['player_name']}")
        print(f"Position: {result['position']}")
        print(f"Momentum Score: {result['momentum_score']}")
        print(f"Trend: {result['trend']}")
        print(f"\nRolling Averages:")
        for period, avg in result['rolling_averages'].items():
            print(f"  {period}: {avg}")
        print(f"\nStreak: {result['streak']['description']}")
        print(f"\nPredictions:")
        print(f"  Breakout Probability: {result['predictions']['breakout_probability']:.1%}")
        print(f"  Regression Probability: {result['predictions']['regression_probability']:.1%}")
        print(f"  Recommendation: {result['predictions']['recommendation']}")
        print(f"\nInsights:")
        for insight in result['insights']:
            print(f"  - {insight}")
    else:
        print(f"Error: {result['error']}")