"""
Synthetic Data Generator for Enhanced Features
Fallback when API data is unavailable
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate realistic synthetic data for ML training"""
    
    def __init__(self):
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
                      'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                      'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                      'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
        
        # Position-specific stat ranges
        self.stat_ranges = {
            'QB': {
                'pass_attempts': (20, 45),
                'pass_yards': (150, 400),
                'pass_tds': (0, 4),
                'interceptions': (0, 3),
                'rush_attempts': (2, 8),
                'rush_yards': (-5, 50),
                'rush_tds': (0, 1),
                'sacks': (0, 5),
                'fumbles': (0, 2)
            },
            'RB': {
                'rush_attempts': (8, 25),
                'rush_yards': (20, 150),
                'rush_tds': (0, 2),
                'receptions': (1, 8),
                'rec_yards': (5, 80),
                'rec_tds': (0, 1),
                'targets': (2, 10),
                'fumbles': (0, 1)
            },
            'WR': {
                'receptions': (2, 12),
                'rec_yards': (20, 180),
                'rec_tds': (0, 2),
                'targets': (4, 15),
                'rush_attempts': (0, 2),
                'rush_yards': (0, 20),
                'air_yards': (30, 200),
                'yac': (10, 100)
            },
            'TE': {
                'receptions': (1, 8),
                'rec_yards': (10, 120),
                'rec_tds': (0, 2),
                'targets': (2, 10),
                'blocking_snaps': (10, 40),
                'routes_run': (15, 35)
            }
        }
        
    def generate_historical_data(self, years: int = 10, players_per_position: int = 50) -> pd.DataFrame:
        """Generate years of synthetic NFL data"""
        logger.info(f"Generating {years} years of synthetic data...")
        
        all_data = []
        current_year = datetime.now().year
        
        # Generate player profiles
        players = self._generate_player_profiles(players_per_position)
        
        # Generate game data for each year
        for year in range(current_year - years, current_year + 1):
            logger.info(f"Generating data for {year} season...")
            
            # Regular season (18 weeks)
            for week in range(1, 19):
                week_data = self._generate_week_data(players, year, week)
                all_data.extend(week_data)
        
        df = pd.DataFrame(all_data)
        
        # Add calculated features
        df = self._add_enhanced_features(df)
        
        logger.info(f"Generated {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def _generate_player_profiles(self, players_per_position: int) -> list:
        """Generate player profiles with consistent attributes"""
        players = []
        player_id = 1000
        
        for position in self.positions:
            for i in range(players_per_position):
                # Generate consistent player attributes
                player = {
                    'player_id': f'P{player_id}',
                    'name': f'{position}_Player_{i+1}',
                    'position': position,
                    'team': np.random.choice(self.teams),
                    'age': np.random.randint(21, 35),
                    'years_experience': np.random.randint(0, 12),
                    'draft_position': np.random.randint(1, 256) if np.random.random() > 0.2 else 0,
                    
                    # Physical attributes
                    'height_inches': self._generate_height(position),
                    'weight_lbs': self._generate_weight(position),
                    
                    # Combine metrics
                    'forty_yard': self._generate_forty_time(position),
                    'vertical_jump': np.random.uniform(28, 42),
                    'broad_jump': np.random.uniform(100, 140),
                    'bench_press': np.random.randint(15, 35) if position != 'QB' else np.random.randint(10, 25),
                    'three_cone': np.random.uniform(6.5, 7.5),
                    'shuttle': np.random.uniform(3.8, 4.5),
                    
                    # Skill ratings (affects performance)
                    'skill_rating': np.random.beta(2, 5),  # Most players average, few elite
                    'consistency': np.random.beta(5, 2),   # Most consistent, few boom/bust
                    'injury_prone': np.random.beta(2, 8)   # Most healthy, few injury prone
                }
                
                # College stats for young players
                if player['years_experience'] <= 3:
                    player.update({
                        'college_dominator': np.random.uniform(0.15, 0.45),
                        'college_ypc': np.random.uniform(5, 15) if position == 'WR' else np.random.uniform(4, 8),
                        'college_td_rate': np.random.uniform(0.05, 0.15),
                        'breakout_age': np.random.uniform(18, 22)
                    })
                
                players.append(player)
                player_id += 1
        
        return players
    
    def _generate_height(self, position: str) -> int:
        """Generate realistic height by position (inches)"""
        height_ranges = {
            'QB': (72, 78),
            'RB': (68, 74),
            'WR': (69, 76),
            'TE': (74, 78)
        }
        min_h, max_h = height_ranges[position]
        return np.random.randint(min_h, max_h + 1)
    
    def _generate_weight(self, position: str) -> int:
        """Generate realistic weight by position (lbs)"""
        weight_ranges = {
            'QB': (205, 245),
            'RB': (195, 230),
            'WR': (175, 220),
            'TE': (235, 265)
        }
        min_w, max_w = weight_ranges[position]
        return np.random.randint(min_w, max_w + 1)
    
    def _generate_forty_time(self, position: str) -> float:
        """Generate realistic 40-yard dash time by position"""
        forty_ranges = {
            'QB': (4.6, 5.1),
            'RB': (4.3, 4.7),
            'WR': (4.25, 4.6),
            'TE': (4.5, 4.9)
        }
        min_t, max_t = forty_ranges[position]
        return round(np.random.uniform(min_t, max_t), 2)
    
    def _generate_week_data(self, players: list, year: int, week: int) -> list:
        """Generate game data for all players in a week"""
        week_data = []
        game_date = datetime(year, 9, 1) + timedelta(weeks=week-1)
        
        # Create matchups
        team_matchups = self._create_matchups()
        
        for player in players:
            # Determine if player plays this week
            if np.random.random() < player['injury_prone']:
                continue  # Injured this week
            
            # Get opponent
            opponent = team_matchups.get(player['team'], np.random.choice(self.teams))
            
            # Generate game conditions
            game_conditions = self._generate_game_conditions(player['team'], opponent)
            
            # Generate performance based on player skill and conditions
            stats = self._generate_player_stats(player, game_conditions)
            
            # Create game record
            game_record = {
                **player,  # Include all player attributes
                **stats,   # Include game stats
                **game_conditions,  # Include game conditions
                'year': year,
                'week': week,
                'game_date': game_date,
                'opponent': opponent,
                'fantasy_points': self._calculate_fantasy_points(stats, player['position'])
            }
            
            week_data.append(game_record)
        
        return week_data
    
    def _create_matchups(self) -> dict:
        """Create team matchups for a week"""
        teams_copy = self.teams.copy()
        np.random.shuffle(teams_copy)
        matchups = {}
        
        for i in range(0, len(teams_copy), 2):
            if i + 1 < len(teams_copy):
                matchups[teams_copy[i]] = teams_copy[i + 1]
                matchups[teams_copy[i + 1]] = teams_copy[i]
        
        return matchups
    
    def _generate_game_conditions(self, home_team: str, away_team: str) -> dict:
        """Generate game conditions (weather, field, etc.)"""
        # Determine if dome game
        dome_teams = ['ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LAR', 'LV', 'MIN', 'NO']
        is_dome = home_team in dome_teams
        
        if is_dome:
            return {
                'temperature': 72,
                'wind_speed': 0,
                'precipitation': 0,
                'humidity': 50,
                'dome_game': True,
                'weather_condition': 'Clear',
                'offensive_line_rank': np.random.randint(1, 33),
                'opp_def_rank': np.random.randint(1, 33),
                'is_primetime': np.random.random() < 0.15,
                'is_division_game': np.random.random() < 0.375
            }
        else:
            # Outdoor conditions
            temp = np.random.normal(65, 15)
            temp = max(20, min(95, temp))  # Realistic bounds
            
            return {
                'temperature': round(temp),
                'wind_speed': abs(np.random.normal(8, 5)),
                'precipitation': max(0, np.random.normal(0, 0.5)),
                'humidity': np.random.randint(30, 90),
                'dome_game': False,
                'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow', 'Windy']),
                'offensive_line_rank': np.random.randint(1, 33),
                'opp_def_rank': np.random.randint(1, 33),
                'is_primetime': np.random.random() < 0.15,
                'is_division_game': np.random.random() < 0.375
            }
    
    def _generate_player_stats(self, player: dict, conditions: dict) -> dict:
        """Generate realistic stats based on player and conditions"""
        position = player['position']
        skill = player['skill_rating']
        consistency = player['consistency']
        
        # Add variance based on consistency
        performance_factor = np.random.beta(consistency * 10, (1 - consistency) * 10)
        
        # Weather impact
        weather_penalty = 1.0
        if not conditions['dome_game']:
            if conditions['wind_speed'] > 15:
                weather_penalty *= 0.9 if position in ['QB', 'WR'] else 0.95
            if conditions['precipitation'] > 0.5:
                weather_penalty *= 0.85
        
        # Opponent impact
        opp_factor = 1.0 - (conditions['opp_def_rank'] - 16) / 32
        
        # Generate base stats
        stats = {}
        ranges = self.stat_ranges[position]
        
        for stat, (min_val, max_val) in ranges.items():
            base_value = min_val + (max_val - min_val) * skill
            adjusted_value = base_value * performance_factor * weather_penalty * opp_factor
            
            # Add some randomness
            final_value = adjusted_value + np.random.normal(0, (max_val - min_val) * 0.1)
            
            # Ensure within bounds and round appropriately
            if stat in ['pass_yards', 'rush_yards', 'rec_yards', 'air_yards', 'yac']:
                stats[stat] = int(max(0, min(max_val * 1.2, final_value)))
            elif stat in ['pass_tds', 'rush_tds', 'rec_tds', 'interceptions', 'fumbles', 'sacks']:
                stats[stat] = int(max(0, min(max_val, round(final_value))))
            else:
                stats[stat] = int(max(0, min(max_val, round(final_value))))
        
        # Add position-specific derived stats
        if position == 'QB':
            stats['completions'] = int(stats['pass_attempts'] * np.random.uniform(0.55, 0.75))
        elif position in ['RB', 'WR', 'TE']:
            if 'targets' in stats and 'receptions' not in stats:
                stats['receptions'] = int(stats['targets'] * np.random.uniform(0.5, 0.8))
        
        # Fill in missing stats with zeros
        all_stats = ['pass_attempts', 'pass_yards', 'pass_tds', 'interceptions',
                     'rush_attempts', 'rush_yards', 'rush_tds', 'receptions',
                     'rec_yards', 'rec_tds', 'targets', 'fumbles', 'sacks']
        
        for stat in all_stats:
            if stat not in stats:
                stats[stat] = 0
        
        return stats
    
    def _calculate_fantasy_points(self, stats: dict, position: str) -> float:
        """Calculate fantasy points (PPR scoring)"""
        points = 0
        
        # Passing
        points += stats.get('pass_yards', 0) * 0.04
        points += stats.get('pass_tds', 0) * 4
        points -= stats.get('interceptions', 0) * 2
        
        # Rushing
        points += stats.get('rush_yards', 0) * 0.1
        points += stats.get('rush_tds', 0) * 6
        
        # Receiving
        points += stats.get('receptions', 0) * 1  # PPR
        points += stats.get('rec_yards', 0) * 0.1
        points += stats.get('rec_tds', 0) * 6
        
        # Negative plays
        points -= stats.get('fumbles', 0) * 2
        
        return round(points, 2)
    
    def _add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features to match enhanced feature engineering"""
        # Sort by player and date for rolling calculations
        df = df.sort_values(['player_id', 'game_date'])
        
        # Add team-level stats (mock data)
        df['team_total_plays'] = np.random.randint(55, 75, len(df))
        df['team_pass_attempts'] = np.random.randint(25, 40, len(df))
        df['team_rush_attempts'] = df['team_total_plays'] - df['team_pass_attempts']
        df['team_points'] = np.random.randint(10, 40, len(df))
        
        # Add snap counts
        df['snap_count'] = np.random.randint(30, 70, len(df))
        df['snap_count_percentage'] = df['snap_count'] / 70
        
        # Add red zone stats
        df['red_zone_targets'] = np.where(
            df['position'].isin(['WR', 'TE']),
            np.random.poisson(1.5, len(df)),
            0
        )
        df['red_zone_touches'] = np.where(
            df['position'] == 'RB',
            np.random.poisson(2, len(df)),
            df['red_zone_targets']
        )
        df['red_zone_tds'] = np.minimum(
            df['red_zone_touches'],
            df['rush_tds'] + df['rec_tds']
        )
        
        # Add advanced receiving stats
        df['air_yards'] = np.where(
            df['position'].isin(['WR', 'TE']),
            df['targets'] * np.random.uniform(8, 12, len(df)),
            0
        )
        df['yac'] = np.where(
            df['receptions'] > 0,
            df['rec_yards'] * np.random.uniform(0.2, 0.5, len(df)),
            0
        )
        
        # Add blocking/route data
        df['blocking_snaps'] = np.where(
            df['position'] == 'TE',
            np.random.randint(5, 25, len(df)),
            0
        )
        df['routes_run'] = np.where(
            df['position'].isin(['WR', 'TE']),
            df['team_pass_attempts'] * np.random.uniform(0.7, 0.95, len(df)),
            0
        )
        
        # Add defensive stats for opponents
        df['opp_pass_def_rank'] = np.random.randint(1, 33, len(df))
        df['opp_rush_def_rank'] = np.random.randint(1, 33, len(df))
        df['opp_fantasy_points_allowed'] = np.random.uniform(15, 30, len(df))
        
        # Add injury status
        df['injury_status'] = np.random.choice(
            ['Healthy', 'Questionable', 'Doubtful'],
            len(df),
            p=[0.85, 0.12, 0.03]
        )
        df['games_missed'] = 0  # Since we only generate games played
        
        # Add time-based features
        df['week_of_season'] = df['week']
        df['month'] = df['game_date'].dt.month
        df['rest_days'] = 7  # Simplified - all regular rest
        
        return df


def generate_enhanced_training_data(years: int = 10) -> pd.DataFrame:
    """Generate enhanced synthetic training data"""
    generator = SyntheticDataGenerator()
    return generator.generate_historical_data(years=years)