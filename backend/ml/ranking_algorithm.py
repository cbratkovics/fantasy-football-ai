"""
Accurate Player Ranking Algorithm
Combines ML predictions, historical performance, and advanced metrics for comprehensive rankings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from scipy import stats
import os

from sqlalchemy import create_engine, func, and_, desc
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats, DraftTier
from backend.ml.predictions import PredictionEngine
from backend.ml.trend_analysis import PlayerTrendAnalyzer
from backend.ml.gmm_clustering import GMMDraftOptimizer

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class PlayerRankingSystem:
    """
    Comprehensive player ranking system using multiple data sources:
    - ML predictions for future performance
    - Historical consistency and trends
    - Position scarcity and replacement value
    - Draft tier clustering
    - Injury risk assessment
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.prediction_engine = PredictionEngine()
        self.trend_analyzer = PlayerTrendAnalyzer()
        
        # Position-based VBD baselines (Value Based Drafting)
        self.vbd_baselines = {
            'QB': 12,   # QB12 is replacement level
            'RB': 24,   # RB24 is replacement level
            'WR': 30,   # WR30 is replacement level
            'TE': 12,   # TE12 is replacement level
            'K': 12,    # K12 is replacement level
        }
    
    def generate_rankings(
        self,
        season: int,
        week: Optional[int] = None,
        scoring_format: str = 'ppr',
        positions: Optional[List[str]] = None,
        include_rookies: bool = True,
        min_games_played: int = 6
    ) -> pd.DataFrame:
        """
        Generate comprehensive player rankings
        
        Args:
            season: Season for rankings
            week: Specific week (None for season-long)
            scoring_format: 'ppr', 'standard', or 'half_ppr'
            positions: List of positions to rank (None for all)
            include_rookies: Whether to include rookies
            min_games_played: Minimum games for established players
            
        Returns:
            DataFrame with comprehensive rankings
        """
        if not positions:
            positions = ['QB', 'RB', 'WR', 'TE', 'K']
        
        all_rankings = []
        
        with self.SessionLocal() as db:
            for position in positions:
                # Get eligible players
                players = self._get_eligible_players(
                    db, position, season, include_rookies, min_games_played
                )
                
                # Rank players within position
                position_rankings = self._rank_position(
                    players, season, week, scoring_format
                )
                
                all_rankings.extend(position_rankings)
        
        # Create DataFrame
        rankings_df = pd.DataFrame(all_rankings)
        
        # Add overall rankings
        rankings_df = self._add_overall_rankings(rankings_df, scoring_format)
        
        # Sort by overall rank
        rankings_df = rankings_df.sort_values('overall_rank')
        
        return rankings_df
    
    def _get_eligible_players(
        self,
        db,
        position: str,
        season: int,
        include_rookies: bool,
        min_games_played: int
    ) -> List[Dict[str, Any]]:
        """Get eligible players for ranking"""
        # Base query
        query = db.query(
            Player.player_id,
            Player.first_name,
            Player.last_name,
            Player.position,
            Player.team,
            Player.age,
            Player.years_exp,
            func.count(PlayerStats.id).label('career_games'),
            func.avg(PlayerStats.fantasy_points_ppr).label('career_ppg_ppr'),
            func.avg(PlayerStats.fantasy_points_std).label('career_ppg_std'),
            func.avg(PlayerStats.fantasy_points_half).label('career_ppg_half')
        ).outerjoin(
            PlayerStats, Player.player_id == PlayerStats.player_id
        ).filter(
            Player.position == position,
            Player.status.in_(['Active', 'Injured Reserve'])
        ).group_by(
            Player.player_id,
            Player.first_name,
            Player.last_name,
            Player.position,
            Player.team,
            Player.age,
            Player.years_exp
        )
        
        # Filter by experience
        if not include_rookies:
            query = query.having(func.count(PlayerStats.id) >= min_games_played)
        
        players = query.all()
        
        # Convert to dict format
        player_list = []
        for p in players:
            player_dict = {
                'player_id': p.player_id,
                'name': f"{p.first_name} {p.last_name}",
                'position': p.position,
                'team': p.team,
                'age': p.age,
                'years_exp': p.years_exp,
                'career_games': p.career_games or 0,
                'career_ppg_ppr': p.career_ppg_ppr or 0,
                'career_ppg_std': p.career_ppg_std or 0,
                'career_ppg_half': p.career_ppg_half or 0
            }
            player_list.append(player_dict)
        
        return player_list
    
    def _rank_position(
        self,
        players: List[Dict[str, Any]],
        season: int,
        week: Optional[int],
        scoring_format: str
    ) -> List[Dict[str, Any]]:
        """Rank players within a position"""
        ranked_players = []
        
        for player in players:
            # Calculate ranking score
            ranking_data = self._calculate_player_score(
                player, season, week, scoring_format
            )
            
            # Merge with player info
            player_ranking = {**player, **ranking_data}
            ranked_players.append(player_ranking)
        
        # Sort by composite score
        ranked_players.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add position rank
        for i, player in enumerate(ranked_players):
            player['position_rank'] = i + 1
            player['tier'] = self._assign_tier(player['position'], i + 1)
        
        return ranked_players
    
    def _calculate_player_score(
        self,
        player: Dict[str, Any],
        season: int,
        week: Optional[int],
        scoring_format: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive ranking score for a player"""
        scores = {}
        weights = {}
        
        # 1. Prediction Score (40% weight)
        prediction_score = self._get_prediction_score(
            player['player_id'], season, week, scoring_format
        )
        scores['prediction'] = prediction_score
        weights['prediction'] = 0.40
        
        # 2. Historical Performance Score (25% weight)
        historical_score = self._get_historical_score(
            player, scoring_format
        )
        scores['historical'] = historical_score
        weights['historical'] = 0.25
        
        # 3. Consistency Score (20% weight)
        consistency_score = self._get_consistency_score(
            player['player_id']
        )
        scores['consistency'] = consistency_score
        weights['consistency'] = 0.20
        
        # 4. Trend Score (10% weight)
        trend_score = self._get_trend_score(
            player['player_id']
        )
        scores['trend'] = trend_score
        weights['trend'] = 0.10
        
        # 5. Availability Score (5% weight)
        availability_score = self._get_availability_score(
            player['player_id']
        )
        scores['availability'] = availability_score
        weights['availability'] = 0.05
        
        # Calculate weighted composite score
        composite_score = sum(
            scores[key] * weights[key] 
            for key in scores 
            if scores[key] is not None
        )
        
        # Normalize weights if some scores are missing
        total_weight = sum(
            weights[key] 
            for key in scores 
            if scores[key] is not None
        )
        
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        return {
            'composite_score': round(composite_score, 3),
            'prediction_score': round(scores['prediction'] or 0, 3),
            'historical_score': round(scores['historical'] or 0, 3),
            'consistency_score': round(scores['consistency'] or 0, 3),
            'trend_score': round(scores['trend'] or 0, 3),
            'availability_score': round(scores['availability'] or 0, 3),
            'score_breakdown': scores,
            'weights_used': weights
        }
    
    def _get_prediction_score(
        self,
        player_id: str,
        season: int,
        week: Optional[int],
        scoring_format: str
    ) -> Optional[float]:
        """Get normalized prediction score"""
        try:
            if week:
                # Single week prediction
                prediction = self.prediction_engine.predict_player_week(
                    player_id, season, week,
                    include_confidence=False,
                    include_factors=False
                )
            else:
                # Season-long projection (average next 5 weeks)
                predictions = []
                for w in range(1, 6):
                    pred = self.prediction_engine.predict_player_week(
                        player_id, season, w,
                        include_confidence=False,
                        include_factors=False
                    )
                    if "error" not in pred:
                        predictions.append(pred)
                
                if not predictions:
                    return None
                
                # Average predictions
                prediction = {
                    'predictions': {
                        scoring_format: {
                            'point_estimate': np.mean([
                                p['predictions'][scoring_format]['point_estimate'] 
                                for p in predictions
                            ])
                        }
                    }
                }
            
            if "error" in prediction:
                return None
            
            # Normalize by position
            points = prediction['predictions'][scoring_format]['point_estimate']
            
            # Position-based normalization (rough ranges)
            position_max = {
                'QB': 30, 'RB': 25, 'WR': 25, 'TE': 20, 'K': 15
            }
            
            with self.SessionLocal() as db:
                player = db.query(Player).filter(
                    Player.player_id == player_id
                ).first()
                
                if player:
                    max_points = position_max.get(player.position, 20)
                    return min(1.0, points / max_points)
            
            return 0.5  # Default
            
        except Exception as e:
            logger.error(f"Failed to get prediction score: {str(e)}")
            return None
    
    def _get_historical_score(
        self,
        player: Dict[str, Any],
        scoring_format: str
    ) -> float:
        """Get normalized historical performance score"""
        # Map scoring format to column
        ppg_column = {
            'ppr': 'career_ppg_ppr',
            'standard': 'career_ppg_std',
            'half_ppr': 'career_ppg_half'
        }
        
        ppg = player.get(ppg_column[scoring_format], 0)
        
        # Position-based normalization
        position_benchmarks = {
            'QB': {'elite': 25, 'good': 18, 'average': 15},
            'RB': {'elite': 20, 'good': 15, 'average': 10},
            'WR': {'elite': 18, 'good': 14, 'average': 10},
            'TE': {'elite': 15, 'good': 10, 'average': 7},
            'K': {'elite': 10, 'good': 8, 'average': 6}
        }
        
        benchmarks = position_benchmarks.get(
            player['position'], 
            {'elite': 15, 'good': 10, 'average': 7}
        )
        
        # Calculate score
        if ppg >= benchmarks['elite']:
            score = 0.9 + (ppg - benchmarks['elite']) / benchmarks['elite'] * 0.1
        elif ppg >= benchmarks['good']:
            score = 0.7 + (ppg - benchmarks['good']) / (benchmarks['elite'] - benchmarks['good']) * 0.2
        elif ppg >= benchmarks['average']:
            score = 0.5 + (ppg - benchmarks['average']) / (benchmarks['good'] - benchmarks['average']) * 0.2
        else:
            score = ppg / benchmarks['average'] * 0.5
        
        return min(1.0, max(0.0, score))
    
    def _get_consistency_score(self, player_id: str) -> Optional[float]:
        """Get normalized consistency score"""
        try:
            analysis = self.trend_analyzer.analyze_player_trends(player_id)
            
            if "error" in analysis:
                return None
            
            consistency = analysis['consistency_metrics']
            cv = consistency['coefficient_of_variation']
            
            # Lower CV is better (more consistent)
            if cv < 0.3:
                score = 0.9
            elif cv < 0.5:
                score = 0.8 - (cv - 0.3) * 0.5
            elif cv < 0.7:
                score = 0.6 - (cv - 0.5) * 0.5
            else:
                score = 0.4 - min(0.3, (cv - 0.7) * 0.3)
            
            return max(0.1, score)
            
        except Exception as e:
            logger.error(f"Failed to get consistency score: {str(e)}")
            return None
    
    def _get_trend_score(self, player_id: str) -> Optional[float]:
        """Get normalized trend score"""
        try:
            analysis = self.trend_analyzer.analyze_player_trends(player_id)
            
            if "error" in analysis:
                return None
            
            trend = analysis['performance_trend']['overall_trend']
            form = analysis['hot_cold_streaks']['current_form']
            
            # Base score
            trend_scores = {
                'improving': 0.8,
                'stable': 0.6,
                'declining': 0.3
            }
            
            form_modifiers = {
                'Hot': 0.2,
                'Average': 0.0,
                'Cold': -0.1
            }
            
            score = trend_scores.get(trend, 0.5)
            score += form_modifiers.get(form, 0.0)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Failed to get trend score: {str(e)}")
            return None
    
    def _get_availability_score(self, player_id: str) -> float:
        """Get availability/durability score"""
        with self.SessionLocal() as db:
            # Get player's game participation rate
            recent_seasons = db.query(
                PlayerStats.season,
                func.count(PlayerStats.week).label('games_played')
            ).filter(
                PlayerStats.player_id == player_id,
                PlayerStats.season >= 2021
            ).group_by(
                PlayerStats.season
            ).all()
            
            if not recent_seasons:
                return 0.5  # Default for new players
            
            # Calculate average games played percentage
            total_possible_games = len(recent_seasons) * 17
            total_games_played = sum(s.games_played for s in recent_seasons)
            
            participation_rate = total_games_played / total_possible_games if total_possible_games > 0 else 0
            
            return participation_rate
    
    def _assign_tier(self, position: str, rank: int) -> int:
        """Assign draft tier based on position and rank"""
        # Position-specific tier breakpoints
        tier_breakpoints = {
            'QB': [3, 6, 10, 14, 18, 24],
            'RB': [6, 12, 18, 24, 36, 48],
            'WR': [6, 12, 18, 24, 36, 48],
            'TE': [3, 6, 10, 14, 18, 24],
            'K': [3, 6, 10, 15, 20, 24]
        }
        
        breakpoints = tier_breakpoints.get(position, [5, 10, 15, 20, 30, 40])
        
        for tier, breakpoint in enumerate(breakpoints, 1):
            if rank <= breakpoint:
                return tier
        
        return len(breakpoints) + 1
    
    def _add_overall_rankings(
        self,
        rankings_df: pd.DataFrame,
        scoring_format: str
    ) -> pd.DataFrame:
        """Add overall rankings using Value Based Drafting (VBD)"""
        # Calculate replacement values by position
        replacement_values = {}
        
        for position, baseline_rank in self.vbd_baselines.items():
            position_players = rankings_df[rankings_df['position'] == position]
            
            if len(position_players) >= baseline_rank:
                replacement_player = position_players.iloc[baseline_rank - 1]
                replacement_values[position] = replacement_player['prediction_score']
            else:
                replacement_values[position] = 0.0
        
        # Calculate VBD score
        rankings_df['vbd_score'] = rankings_df.apply(
            lambda row: row['composite_score'] - replacement_values.get(row['position'], 0),
            axis=1
        )
        
        # Add positional scarcity factor
        position_scarcity = {
            'RB': 1.1,  # RBs are scarcer
            'TE': 1.05,  # Elite TEs are valuable
            'QB': 0.95,  # QBs are deeper
            'WR': 1.0,   # Baseline
            'K': 0.8     # Kickers are replaceable
        }
        
        rankings_df['adjusted_vbd'] = rankings_df.apply(
            lambda row: row['vbd_score'] * position_scarcity.get(row['position'], 1.0),
            axis=1
        )
        
        # Sort by adjusted VBD
        rankings_df = rankings_df.sort_values('adjusted_vbd', ascending=False)
        
        # Add overall rank
        rankings_df['overall_rank'] = range(1, len(rankings_df) + 1)
        
        # Add ADP comparison (would need actual ADP data)
        rankings_df['adp'] = rankings_df['overall_rank'] * 1.1  # Placeholder
        rankings_df['adp_diff'] = rankings_df['overall_rank'] - rankings_df['adp']
        
        return rankings_df
    
    def get_position_tiers(
        self,
        position: str,
        season: int,
        scoring_format: str = 'ppr'
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Get players grouped by tiers for a position"""
        rankings = self.generate_rankings(
            season=season,
            scoring_format=scoring_format,
            positions=[position]
        )
        
        # Group by tier
        tiers = {}
        for _, player in rankings.iterrows():
            tier = player['tier']
            if tier not in tiers:
                tiers[tier] = []
            
            tiers[tier].append({
                'player_id': player['player_id'],
                'name': player['name'],
                'team': player['team'],
                'position_rank': player['position_rank'],
                'composite_score': player['composite_score'],
                'prediction_score': player['prediction_score']
            })
        
        return tiers
    
    def export_rankings(
        self,
        rankings_df: pd.DataFrame,
        filename: str,
        format: str = 'csv'
    ):
        """Export rankings to file"""
        if format == 'csv':
            rankings_df.to_csv(filename, index=False)
        elif format == 'json':
            rankings_df.to_json(filename, orient='records', indent=2)
        elif format == 'excel':
            rankings_df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported rankings to {filename}")


# Example usage
if __name__ == "__main__":
    ranker = PlayerRankingSystem()
    
    # Generate PPR rankings for 2024
    rankings = ranker.generate_rankings(
        season=2024,
        scoring_format='ppr',
        positions=['QB', 'RB', 'WR', 'TE']
    )
    
    # Display top 20
    print("Top 20 Overall Rankings (PPR):")
    print("-" * 100)
    print(f"{'Rank':<5} {'Name':<25} {'Pos':<4} {'Team':<4} {'Score':<8} {'Pred':<8} {'Consist':<8} {'Tier':<5}")
    print("-" * 100)
    
    for _, player in rankings.head(20).iterrows():
        print(f"{player['overall_rank']:<5} {player['name']:<25} {player['position']:<4} {player['team']:<4} "
              f"{player['composite_score']:<8.3f} {player['prediction_score']:<8.3f} "
              f"{player['consistency_score']:<8.3f} {player['tier']:<5}")
    
    # Export to CSV
    ranker.export_rankings(rankings, "fantasy_rankings_2024_ppr.csv")