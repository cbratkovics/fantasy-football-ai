"""
Comprehensive Player Profile System
Includes all physical, historical, and contextual attributes for ML accuracy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class PlayerProfile:
    """Complete player profile with all relevant attributes"""
    
    # Basic Information
    player_id: str
    name: str
    position: str
    team: str
    jersey_number: int
    
    # Physical Attributes
    height_inches: int
    weight_lbs: int
    age: float  # Decimal age for precision
    date_of_birth: datetime
    hand_size: Optional[float] = None  # Important for QBs/WRs
    arm_length: Optional[float] = None  # Reach advantage
    
    # Athletic Metrics (NFL Combine + Pro Day)
    forty_yard_dash: Optional[float] = None
    bench_press_reps: Optional[int] = None
    vertical_jump: Optional[float] = None
    broad_jump: Optional[int] = None
    three_cone_drill: Optional[float] = None
    twenty_yard_shuttle: Optional[float] = None
    sixty_yard_shuttle: Optional[float] = None
    
    # Calculated Athletic Scores
    speed_score: Optional[float] = None  # (Weight * 200) / (40-time ^ 4)
    burst_score: Optional[float] = None  # Vertical + Broad Jump
    agility_score: Optional[float] = None  # 3-cone + shuttle composite
    catch_radius: Optional[float] = None  # Height + arm length + vertical
    bmi: Optional[float] = None
    
    # Experience & Background
    years_in_league: int = 0
    games_played_career: int = 0
    games_started_career: int = 0
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None
    draft_value: Optional[float] = None  # Trade value chart points
    undrafted: bool = False
    
    # College Background
    college: Optional[str] = None
    college_conference: Optional[str] = None
    college_games_played: Optional[int] = None
    college_production_score: Optional[float] = None
    college_dominator_rating: Optional[float] = None  # % of team yards/TDs
    breakout_age: Optional[float] = None  # Age of college breakout season
    early_declare: bool = False  # Left college early
    
    # Career Statistics
    career_stats: Dict[str, float] = field(default_factory=dict)
    season_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    game_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance Metrics
    career_ppg: float = 0.0  # Fantasy points per game
    career_consistency: float = 0.0  # StdDev of fantasy points
    career_floor: float = 0.0  # 25th percentile performance
    career_ceiling: float = 0.0  # 75th percentile performance
    boom_rate: float = 0.0  # % games > 20 fantasy points
    bust_rate: float = 0.0  # % games < 10 fantasy points
    
    # Injury History
    injury_history: List[Dict[str, Any]] = field(default_factory=list)
    games_missed_injuries: int = 0
    injury_risk_score: float = 0.0  # ML-calculated risk
    current_health_status: str = "Healthy"
    recovery_speed: float = 1.0  # Historical recovery rate
    
    # Team Context
    offensive_scheme: str = ""  # West Coast, Air Raid, etc.
    offensive_coordinator: str = ""
    head_coach: str = ""
    quarterback_rating: Optional[float] = None  # Team QB rating
    offensive_line_rank: Optional[int] = None
    team_pace: Optional[float] = None  # Plays per game
    team_pass_rate: Optional[float] = None
    red_zone_opportunities: Optional[float] = None
    
    # Situational Performance
    home_ppg: float = 0.0
    away_ppg: float = 0.0
    dome_ppg: float = 0.0
    outdoor_ppg: float = 0.0
    primetime_ppg: float = 0.0
    division_ppg: float = 0.0
    weather_impact: Dict[str, float] = field(default_factory=dict)  # By condition
    
    # Matchup History
    vs_opponent_history: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vs_defense_ranking: Dict[str, float] = field(default_factory=dict)  # By defensive rank tier
    
    # Usage Patterns
    snap_count_pct: float = 0.0
    route_participation: float = 0.0  # For pass catchers
    target_share: float = 0.0
    air_yards_share: float = 0.0
    red_zone_share: float = 0.0
    opportunity_share: float = 0.0  # Touches + targets
    
    # Advanced Metrics
    yards_before_contact: Optional[float] = None  # RBs
    yards_after_contact: Optional[float] = None  # RBs
    true_catch_rate: Optional[float] = None  # Adjusted for difficulty
    separation_score: Optional[float] = None  # WRs/TEs
    pass_block_grade: Optional[float] = None  # RBs
    run_block_grade: Optional[float] = None  # TEs
    pressure_rate_when_targeted: Optional[float] = None  # QBs
    
    # Fantasy Specific
    adp: Optional[float] = None  # Average Draft Position
    auction_value: Optional[float] = None
    dynasty_value: Optional[float] = None
    keeper_value: Optional[float] = None
    ceiling_projection: Optional[float] = None
    floor_projection: Optional[float] = None
    
    def calculate_athletic_scores(self):
        """Calculate composite athletic scores"""
        # Speed Score (RB-centric but useful for all)
        if self.forty_yard_dash and self.weight_lbs:
            self.speed_score = (self.weight_lbs * 200) / (self.forty_yard_dash ** 4)
        
        # Burst Score
        if self.vertical_jump and self.broad_jump:
            self.burst_score = self.vertical_jump + (self.broad_jump / 12)
        
        # Agility Score
        if self.three_cone_drill and self.twenty_yard_shuttle:
            self.agility_score = 2 / (1/self.three_cone_drill + 1/self.twenty_yard_shuttle)
        
        # Catch Radius (WR/TE)
        if self.position in ['WR', 'TE'] and self.height_inches and self.arm_length and self.vertical_jump:
            self.catch_radius = self.height_inches + self.arm_length + (self.vertical_jump / 2)
        
        # BMI
        if self.height_inches and self.weight_lbs:
            self.bmi = (self.weight_lbs / (self.height_inches ** 2)) * 703
    
    def calculate_consistency_metrics(self):
        """Calculate performance consistency metrics"""
        if self.game_logs:
            fantasy_points = [g.get('fantasy_points', 0) for g in self.game_logs]
            if fantasy_points:
                self.career_ppg = np.mean(fantasy_points)
                self.career_consistency = np.std(fantasy_points)
                self.career_floor = np.percentile(fantasy_points, 25)
                self.career_ceiling = np.percentile(fantasy_points, 75)
                self.boom_rate = sum(1 for fp in fantasy_points if fp > 20) / len(fantasy_points)
                self.bust_rate = sum(1 for fp in fantasy_points if fp < 10) / len(fantasy_points)
    
    def get_age_at_date(self, date: datetime) -> float:
        """Get precise age at a specific date"""
        if self.date_of_birth:
            delta = date - self.date_of_birth
            return delta.days / 365.25
        return self.age
    
    def get_experience_score(self) -> float:
        """Calculate experience score (peaks around year 4-7)"""
        if self.years_in_league <= 0:
            return 0.5
        elif self.years_in_league <= 3:
            return 0.5 + (self.years_in_league * 0.15)
        elif self.years_in_league <= 7:
            return 1.0
        else:
            # Gradual decline after year 7
            return max(0.7, 1.0 - (self.years_in_league - 7) * 0.05)
    
    def get_athletic_percentile(self, metric: str, position: str) -> Optional[float]:
        """Get percentile ranking for athletic metric by position"""
        # This would reference a database of historical combine results
        # Placeholder for demonstration
        percentile_data = {
            'QB': {'forty_yard_dash': 4.8, 'vertical_jump': 30},
            'RB': {'forty_yard_dash': 4.5, 'vertical_jump': 35},
            'WR': {'forty_yard_dash': 4.45, 'vertical_jump': 36},
            'TE': {'forty_yard_dash': 4.7, 'vertical_jump': 33}
        }
        
        if position in percentile_data and metric in percentile_data[position]:
            avg = percentile_data[position][metric]
            value = getattr(self, metric, None)
            if value:
                # Simple percentile calculation (would be more sophisticated in practice)
                if metric == 'forty_yard_dash':
                    # Lower is better for 40-yard dash
                    percentile = 50 + (avg - value) * 50
                else:
                    # Higher is better for most metrics
                    percentile = 50 + (value - avg) * 2
                return max(0, min(100, percentile))
        return None


class PlayerProfileBuilder:
    """Build comprehensive player profiles from multiple data sources"""
    
    def __init__(self):
        self.profiles: Dict[str, PlayerProfile] = {}
    
    def build_profile(self, player_data: Dict[str, Any]) -> PlayerProfile:
        """Build a complete player profile from raw data"""
        profile = PlayerProfile(
            player_id=player_data.get('player_id', ''),
            name=player_data.get('name', ''),
            position=player_data.get('position', ''),
            team=player_data.get('team', ''),
            jersey_number=player_data.get('jersey_number', 0),
            height_inches=player_data.get('height_inches', 0),
            weight_lbs=player_data.get('weight_lbs', 0),
            age=player_data.get('age', 0),
            date_of_birth=player_data.get('date_of_birth', datetime.now())
        )
        
        # Add athletic metrics
        for metric in ['forty_yard_dash', 'bench_press_reps', 'vertical_jump', 
                      'broad_jump', 'three_cone_drill', 'twenty_yard_shuttle']:
            if metric in player_data:
                setattr(profile, metric, player_data[metric])
        
        # Calculate derived scores
        profile.calculate_athletic_scores()
        
        # Add career data
        if 'career_stats' in player_data:
            profile.career_stats = player_data['career_stats']
        
        if 'game_logs' in player_data:
            profile.game_logs = player_data['game_logs']
            profile.calculate_consistency_metrics()
        
        return profile
    
    def enrich_with_situational_data(self, profile: PlayerProfile, 
                                     situational_data: pd.DataFrame) -> PlayerProfile:
        """Add situational performance data to profile"""
        player_games = situational_data[situational_data['player_id'] == profile.player_id]
        
        if not player_games.empty:
            # Home/Away splits
            home_games = player_games[player_games['is_home'] == True]
            away_games = player_games[player_games['is_home'] == False]
            profile.home_ppg = home_games['fantasy_points'].mean() if not home_games.empty else 0
            profile.away_ppg = away_games['fantasy_points'].mean() if not away_games.empty else 0
            
            # Dome/Outdoor splits
            dome_games = player_games[player_games['dome_game'] == True]
            outdoor_games = player_games[player_games['dome_game'] == False]
            profile.dome_ppg = dome_games['fantasy_points'].mean() if not dome_games.empty else 0
            profile.outdoor_ppg = outdoor_games['fantasy_points'].mean() if not outdoor_games.empty else 0
            
            # Weather impact
            for condition in ['Clear', 'Rain', 'Snow', 'Wind']:
                condition_games = player_games[player_games['weather_condition'] == condition]
                if not condition_games.empty:
                    profile.weather_impact[condition] = condition_games['fantasy_points'].mean()
            
            # Opponent history
            for opponent in player_games['opponent'].unique():
                opp_games = player_games[player_games['opponent'] == opponent]
                profile.vs_opponent_history[opponent] = {
                    'games': len(opp_games),
                    'ppg': opp_games['fantasy_points'].mean(),
                    'last_performance': opp_games.iloc[-1]['fantasy_points'] if not opp_games.empty else 0
                }
        
        return profile
    
    def create_ml_features(self, profile: PlayerProfile, game_context: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for ML model from profile and game context"""
        features = []
        
        # Physical features
        features.extend([
            profile.height_inches,
            profile.weight_lbs,
            profile.bmi or 0,
            profile.get_age_at_date(game_context.get('game_date', datetime.now()))
        ])
        
        # Athletic features
        features.extend([
            profile.speed_score or 0,
            profile.burst_score or 0,
            profile.agility_score or 0,
            profile.catch_radius or 0 if profile.position in ['WR', 'TE'] else 0
        ])
        
        # Experience features
        features.extend([
            profile.years_in_league,
            profile.games_played_career,
            profile.get_experience_score(),
            profile.draft_value or 0
        ])
        
        # Performance features
        features.extend([
            profile.career_ppg,
            profile.career_consistency,
            profile.career_floor,
            profile.career_ceiling,
            profile.boom_rate,
            profile.bust_rate
        ])
        
        # Recent form (last 3, 5, 10 games)
        recent_games = profile.game_logs[-10:] if profile.game_logs else []
        if recent_games:
            recent_points = [g.get('fantasy_points', 0) for g in recent_games]
            features.extend([
                np.mean(recent_points[-3:]) if len(recent_points) >= 3 else profile.career_ppg,
                np.mean(recent_points[-5:]) if len(recent_points) >= 5 else profile.career_ppg,
                np.mean(recent_points),
                np.std(recent_points) if len(recent_points) > 1 else profile.career_consistency
            ])
        else:
            features.extend([profile.career_ppg] * 4)
        
        # Matchup features
        opponent = game_context.get('opponent', '')
        if opponent in profile.vs_opponent_history:
            opp_history = profile.vs_opponent_history[opponent]
            features.extend([
                opp_history['ppg'],
                opp_history['games'],
                opp_history['last_performance']
            ])
        else:
            features.extend([profile.career_ppg, 0, profile.career_ppg])
        
        # Situational features
        is_home = game_context.get('is_home', True)
        is_dome = game_context.get('dome_game', False)
        weather = game_context.get('weather_condition', 'Clear')
        
        features.extend([
            profile.home_ppg if is_home else profile.away_ppg,
            profile.dome_ppg if is_dome else profile.outdoor_ppg,
            profile.weather_impact.get(weather, profile.career_ppg),
            1 if game_context.get('is_primetime', False) else 0,
            1 if game_context.get('is_division', False) else 0
        ])
        
        # Team context features
        features.extend([
            profile.offensive_line_rank or 16,
            profile.team_pace or 65,
            profile.team_pass_rate or 0.6,
            profile.quarterback_rating or 90
        ])
        
        # Usage features
        features.extend([
            profile.snap_count_pct,
            profile.target_share,
            profile.air_yards_share,
            profile.red_zone_share,
            profile.opportunity_share
        ])
        
        # Injury features
        features.extend([
            profile.injury_risk_score,
            1 if profile.current_health_status == "Healthy" else 0.5,
            profile.recovery_speed
        ])
        
        return np.array(features)


def create_player_database(data_sources: Dict[str, pd.DataFrame]) -> Dict[str, PlayerProfile]:
    """Create comprehensive player database from multiple sources"""
    builder = PlayerProfileBuilder()
    
    # Combine data sources
    nfl_data = data_sources.get('nfl_stats', pd.DataFrame())
    combine_data = data_sources.get('combine', pd.DataFrame())
    injury_data = data_sources.get('injuries', pd.DataFrame())
    game_logs = data_sources.get('game_logs', pd.DataFrame())
    
    player_profiles = {}
    
    # Build profiles for each unique player
    if not nfl_data.empty:
        for player_id in nfl_data['player_id'].unique():
            player_rows = nfl_data[nfl_data['player_id'] == player_id]
            
            # Get most recent data
            latest = player_rows.iloc[-1]
            
            # Build base profile
            player_data = {
                'player_id': player_id,
                'name': latest.get('name', ''),
                'position': latest.get('position', ''),
                'team': latest.get('team', ''),
                'height_inches': latest.get('height_inches', 0),
                'weight_lbs': latest.get('weight_lbs', 0),
                'age': latest.get('age', 0),
                'years_in_league': latest.get('years_experience', 0)
            }
            
            # Add combine data if available
            if not combine_data.empty:
                combine_row = combine_data[combine_data['player_id'] == player_id]
                if not combine_row.empty:
                    player_data.update(combine_row.iloc[0].to_dict())
            
            # Add game logs
            if not game_logs.empty:
                player_games = game_logs[game_logs['player_id'] == player_id]
                player_data['game_logs'] = player_games.to_dict('records')
            
            # Build profile
            profile = builder.build_profile(player_data)
            
            # Enrich with situational data
            if not game_logs.empty:
                profile = builder.enrich_with_situational_data(profile, game_logs)
            
            player_profiles[player_id] = profile
    
    return player_profiles