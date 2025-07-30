"""
Player Trend Analysis Module
Analyzes historical performance patterns to identify trends, consistency, and predictive signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging
import os

from sqlalchemy import create_engine, func, and_
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class PlayerTrendAnalyzer:
    """Analyzes player trends and patterns from historical data"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def analyze_player_trends(
        self, 
        player_id: str,
        seasons: Optional[List[int]] = None,
        min_games: int = 6
    ) -> Dict[str, Any]:
        """
        Comprehensive trend analysis for a player
        
        Returns:
            - Performance trends (improving/declining)
            - Consistency metrics
            - Seasonal patterns
            - Injury impact analysis
            - Matchup-based trends
        """
        with self.SessionLocal() as db:
            # Get player info
            player = db.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return {"error": "Player not found"}
            
            # Get historical stats
            query = db.query(PlayerStats).filter(
                PlayerStats.player_id == player_id
            ).order_by(PlayerStats.season, PlayerStats.week)
            
            if seasons:
                query = query.filter(PlayerStats.season.in_(seasons))
            
            stats_df = pd.read_sql(query.statement, db.bind)
            
        if len(stats_df) < min_games:
            return {"error": "Insufficient data for analysis"}
        
        # Perform various analyses
        analysis = {
            "player_name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "games_analyzed": len(stats_df),
            "performance_trend": self._analyze_performance_trend(stats_df),
            "consistency_metrics": self._analyze_consistency(stats_df),
            "seasonal_patterns": self._analyze_seasonal_patterns(stats_df),
            "hot_cold_streaks": self._identify_streaks(stats_df),
            "injury_impact": self._analyze_injury_impact(stats_df),
            "matchup_trends": self._analyze_matchup_trends(stats_df),
            "scoring_distribution": self._analyze_scoring_distribution(stats_df),
            "projected_trajectory": self._project_future_performance(stats_df)
        }
        
        return analysis
    
    def _analyze_performance_trend(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall performance trends"""
        # Overall trend line
        x = np.arange(len(stats_df))
        y = stats_df['fantasy_points_ppr'].values
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Recent vs early performance
        early_games = stats_df.head(len(stats_df) // 3)['fantasy_points_ppr'].mean()
        recent_games = stats_df.tail(len(stats_df) // 3)['fantasy_points_ppr'].mean()
        
        # Trend classification
        if slope > 0.1 and p_value < 0.05:
            trend = "improving"
        elif slope < -0.1 and p_value < 0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        # Year-over-year comparison
        yoy_trends = {}
        for season in stats_df['season'].unique():
            season_avg = stats_df[stats_df['season'] == season]['fantasy_points_ppr'].mean()
            yoy_trends[int(season)] = round(season_avg, 2)
        
        return {
            "overall_trend": trend,
            "trend_slope": round(slope, 3),
            "trend_confidence": round(1 - p_value, 3),
            "early_season_avg": round(early_games, 2),
            "recent_avg": round(recent_games, 2),
            "improvement_rate": round((recent_games - early_games) / early_games * 100, 1) if early_games > 0 else 0,
            "year_over_year": yoy_trends
        }
    
    def _analyze_consistency(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency metrics"""
        points = stats_df['fantasy_points_ppr']
        
        # Basic stats
        mean_points = points.mean()
        std_points = points.std()
        cv = std_points / mean_points if mean_points > 0 else 0
        
        # Floor and ceiling
        floor = np.percentile(points, 20)
        ceiling = np.percentile(points, 80)
        
        # Consistency score (lower CV is better)
        if cv < 0.3:
            consistency_rating = "Very Consistent"
        elif cv < 0.5:
            consistency_rating = "Consistent"
        elif cv < 0.7:
            consistency_rating = "Moderately Consistent"
        else:
            consistency_rating = "Volatile"
        
        # Weekly reliability
        reliable_games = ((points >= mean_points * 0.8) & (points <= mean_points * 1.2)).sum()
        reliability_pct = reliable_games / len(points) * 100
        
        return {
            "average_points": round(mean_points, 2),
            "standard_deviation": round(std_points, 2),
            "coefficient_of_variation": round(cv, 3),
            "consistency_rating": consistency_rating,
            "floor_20th_percentile": round(floor, 2),
            "ceiling_80th_percentile": round(ceiling, 2),
            "reliable_game_percentage": round(reliability_pct, 1),
            "boom_games": int((points > mean_points * 1.5).sum()),
            "bust_games": int((points < mean_points * 0.5).sum())
        }
    
    def _analyze_seasonal_patterns(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns within seasons"""
        # Group by week across all seasons
        weekly_avg = stats_df.groupby('week')['fantasy_points_ppr'].mean()
        
        # Early, mid, late season splits
        early_season = stats_df[stats_df['week'] <= 6]['fantasy_points_ppr'].mean()
        mid_season = stats_df[(stats_df['week'] > 6) & (stats_df['week'] <= 13)]['fantasy_points_ppr'].mean()
        late_season = stats_df[stats_df['week'] > 13]['fantasy_points_ppr'].mean()
        
        # Best and worst weeks
        best_weeks = weekly_avg.nlargest(3)
        worst_weeks = weekly_avg.nsmallest(3)
        
        # Monthly trends (approximate)
        monthly_trends = {
            "September": stats_df[stats_df['week'].isin([1, 2, 3, 4])]['fantasy_points_ppr'].mean(),
            "October": stats_df[stats_df['week'].isin([5, 6, 7, 8])]['fantasy_points_ppr'].mean(),
            "November": stats_df[stats_df['week'].isin([9, 10, 11, 12])]['fantasy_points_ppr'].mean(),
            "December": stats_df[stats_df['week'].isin([13, 14, 15, 16])]['fantasy_points_ppr'].mean(),
            "Playoffs": stats_df[stats_df['week'] >= 17]['fantasy_points_ppr'].mean()
        }
        
        return {
            "early_season_avg": round(early_season, 2),
            "mid_season_avg": round(mid_season, 2),
            "late_season_avg": round(late_season, 2),
            "best_weeks": {int(k): round(v, 2) for k, v in best_weeks.items()},
            "worst_weeks": {int(k): round(v, 2) for k, v in worst_weeks.items()},
            "monthly_trends": {k: round(v, 2) for k, v in monthly_trends.items() if not np.isnan(v)}
        }
    
    def _identify_streaks(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify hot and cold streaks"""
        points = stats_df['fantasy_points_ppr'].values
        mean_points = np.mean(points)
        
        # Define hot/cold thresholds
        hot_threshold = mean_points * 1.2
        cold_threshold = mean_points * 0.8
        
        # Track streaks
        current_streak = 0
        streak_type = None
        all_streaks = []
        
        for i, pts in enumerate(points):
            if pts >= hot_threshold:
                if streak_type == 'hot':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        all_streaks.append((streak_type, current_streak))
                    streak_type = 'hot'
                    current_streak = 1
            elif pts <= cold_threshold:
                if streak_type == 'cold':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        all_streaks.append((streak_type, current_streak))
                    streak_type = 'cold'
                    current_streak = 1
            else:
                if current_streak > 0:
                    all_streaks.append((streak_type, current_streak))
                current_streak = 0
                streak_type = None
        
        # Add final streak
        if current_streak > 0:
            all_streaks.append((streak_type, current_streak))
        
        # Analyze streaks
        hot_streaks = [s[1] for s in all_streaks if s[0] == 'hot']
        cold_streaks = [s[1] for s in all_streaks if s[0] == 'cold']
        
        # Current form (last 5 games)
        recent_form = points[-5:] if len(points) >= 5 else points
        recent_avg = np.mean(recent_form)
        
        if recent_avg > mean_points * 1.2:
            current_form = "Hot"
        elif recent_avg < mean_points * 0.8:
            current_form = "Cold"
        else:
            current_form = "Average"
        
        return {
            "longest_hot_streak": max(hot_streaks) if hot_streaks else 0,
            "longest_cold_streak": max(cold_streaks) if cold_streaks else 0,
            "total_hot_streaks": len(hot_streaks),
            "total_cold_streaks": len(cold_streaks),
            "current_form": current_form,
            "last_5_games_avg": round(recent_avg, 2),
            "streak_volatility": len(all_streaks) / len(points) if len(points) > 0 else 0
        }
    
    def _analyze_injury_impact(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of injuries on performance"""
        # Extract injury info from stats
        injury_games = []
        return_games = []
        
        for i in range(1, len(stats_df)):
            prev_week = stats_df.iloc[i-1]['week']
            curr_week = stats_df.iloc[i]['week']
            
            # If gap in weeks (same season), player likely missed games
            if (stats_df.iloc[i]['season'] == stats_df.iloc[i-1]['season'] and 
                curr_week - prev_week > 1):
                return_games.append(i)
        
        # Performance after returns
        post_return_performance = []
        for idx in return_games:
            if idx < len(stats_df) - 3:  # Need at least 3 games after return
                post_return_avg = stats_df.iloc[idx:idx+3]['fantasy_points_ppr'].mean()
                pre_injury_avg = stats_df.iloc[max(0, idx-5):idx]['fantasy_points_ppr'].mean()
                post_return_performance.append({
                    'games_missed': int(stats_df.iloc[idx]['week'] - stats_df.iloc[idx-1]['week'] - 1),
                    'performance_drop': round((pre_injury_avg - post_return_avg) / pre_injury_avg * 100, 1) if pre_injury_avg > 0 else 0
                })
        
        return {
            "games_missed_total": len(return_games),
            "injury_returns": post_return_performance,
            "avg_performance_drop_post_injury": round(
                np.mean([p['performance_drop'] for p in post_return_performance]), 1
            ) if post_return_performance else 0
        }
    
    def _analyze_matchup_trends(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance against different types of defenses"""
        # This would require defensive ranking data
        # For now, analyze performance variance
        
        # Group by opponent
        opp_performance = stats_df.groupby('opponent')['fantasy_points_ppr'].agg(['mean', 'count'])
        
        # Best and worst matchups
        best_matchups = opp_performance[opp_performance['count'] >= 2].nlargest(5, 'mean')
        worst_matchups = opp_performance[opp_performance['count'] >= 2].nsmallest(5, 'mean')
        
        # Home vs away
        home_games = stats_df[stats_df['is_home'] == True]['fantasy_points_ppr'].mean() if 'is_home' in stats_df else 0
        away_games = stats_df[stats_df['is_home'] == False]['fantasy_points_ppr'].mean() if 'is_home' in stats_df else 0
        
        return {
            "best_matchups": {idx: round(row['mean'], 2) for idx, row in best_matchups.iterrows()},
            "worst_matchups": {idx: round(row['mean'], 2) for idx, row in worst_matchups.iterrows()},
            "home_avg": round(home_games, 2),
            "away_avg": round(away_games, 2),
            "home_away_split": round(home_games - away_games, 2)
        }
    
    def _analyze_scoring_distribution(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how points are distributed"""
        points = stats_df['fantasy_points_ppr']
        
        # Percentile distribution
        percentiles = {
            "10th": np.percentile(points, 10),
            "25th": np.percentile(points, 25),
            "50th": np.percentile(points, 50),
            "75th": np.percentile(points, 75),
            "90th": np.percentile(points, 90)
        }
        
        # Scoring ranges
        ranges = {
            "0-5 points": ((points >= 0) & (points < 5)).sum(),
            "5-10 points": ((points >= 5) & (points < 10)).sum(),
            "10-15 points": ((points >= 10) & (points < 15)).sum(),
            "15-20 points": ((points >= 15) & (points < 20)).sum(),
            "20-25 points": ((points >= 20) & (points < 25)).sum(),
            "25+ points": (points >= 25).sum()
        }
        
        # Calculate skewness
        skewness = stats.skew(points)
        
        return {
            "percentiles": {k: round(v, 2) for k, v in percentiles.items()},
            "scoring_ranges": ranges,
            "skewness": round(skewness, 3),
            "distribution_type": "Right-skewed (boom potential)" if skewness > 0.5 else "Normal" if abs(skewness) <= 0.5 else "Left-skewed"
        }
    
    def _project_future_performance(self, stats_df: pd.DataFrame) -> Dict[str, Any]:
        """Project future performance based on trends"""
        # Simple projection using recent trend
        recent_games = min(10, len(stats_df))
        recent_data = stats_df.tail(recent_games)
        
        x = np.arange(recent_games).reshape(-1, 1)
        y = recent_data['fantasy_points_ppr'].values
        
        # Fit model
        model = LinearRegression()
        model.fit(x, y)
        
        # Project next 5 games
        future_x = np.arange(recent_games, recent_games + 5).reshape(-1, 1)
        projections = model.predict(future_x)
        
        # Confidence based on recent consistency
        recent_std = recent_data['fantasy_points_ppr'].std()
        confidence = "High" if recent_std < 5 else "Medium" if recent_std < 10 else "Low"
        
        return {
            "next_game_projection": round(projections[0], 2),
            "5_game_projection": [round(p, 2) for p in projections],
            "projection_confidence": confidence,
            "trend_direction": "up" if model.coef_[0] > 0 else "down",
            "recent_volatility": round(recent_std, 2)
        }
    
    def compare_players(
        self,
        player_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple players across key metrics"""
        if not metrics:
            metrics = [
                'average_points', 'consistency_rating', 'floor_20th_percentile',
                'ceiling_80th_percentile', 'boom_games', 'bust_games'
            ]
        
        comparison_data = []
        
        for player_id in player_ids:
            analysis = self.analyze_player_trends(player_id)
            if "error" not in analysis:
                player_data = {
                    'player_name': analysis['player_name'],
                    'position': analysis['position'],
                    'games': analysis['games_analyzed']
                }
                
                # Extract requested metrics
                for metric in metrics:
                    if metric in analysis['consistency_metrics']:
                        player_data[metric] = analysis['consistency_metrics'][metric]
                    elif metric in analysis['performance_trend']:
                        player_data[metric] = analysis['performance_trend'][metric]
                
                comparison_data.append(player_data)
        
        return pd.DataFrame(comparison_data)


# Example usage
if __name__ == "__main__":
    import os
    
    analyzer = PlayerTrendAnalyzer()
    
    # Analyze a specific player
    player_id = "6783"  # Example player ID
    analysis = analyzer.analyze_player_trends(player_id)
    
    if "error" not in analysis:
        print(f"Player: {analysis['player_name']}")
        print(f"Position: {analysis['position']}")
        print(f"Games Analyzed: {analysis['games_analyzed']}")
        print(f"\nPerformance Trend: {analysis['performance_trend']['overall_trend']}")
        print(f"Average Points: {analysis['consistency_metrics']['average_points']}")
        print(f"Consistency: {analysis['consistency_metrics']['consistency_rating']}")
        print(f"Current Form: {analysis['hot_cold_streaks']['current_form']}")
    else:
        print(f"Error: {analysis['error']}")