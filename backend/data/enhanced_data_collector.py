"""
Enhanced Data Collection System
Collects 10 years of NFL data + college stats for rookies
Includes combine data, weather, injuries, and advanced metrics
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dataclasses import dataclass
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class PlayerProfile:
    """Complete player profile with all data sources"""
    player_id: str
    name: str
    position: str
    team: str
    # NFL Stats
    nfl_stats: Dict[str, Any]
    career_stats: Dict[str, Any]
    # College Stats (for rookies)
    college_stats: Optional[Dict[str, Any]]
    # Combine Data
    combine_metrics: Optional[Dict[str, Any]]
    # Physical Attributes
    height: Optional[int]  # in inches
    weight: Optional[int]  # in pounds
    age: Optional[int]
    # Injury History
    injury_history: List[Dict[str, Any]]
    # Team Context
    offensive_line_rank: Optional[int]
    offensive_coordinator: Optional[str]
    team_pass_rate: Optional[float]
    

class EnhancedDataCollector:
    """
    Comprehensive data collection system for fantasy football ML
    """
    
    def __init__(self):
        self.base_urls = {
            'nfl_stats': 'https://api.sportsdata.io/v3/nfl/stats/json/',
            'college_stats': 'https://api.collegefootballdata.com/',
            'combine': 'https://api.sportsdata.io/v3/nfl/scores/json/',
            'weather': 'https://api.openweathermap.org/data/2.5/',
            'injuries': 'https://www.pro-football-reference.com/'
        }
        
        # API Keys (should be in environment variables)
        self.api_keys = {
            'sportsdata': os.getenv('SPORTSDATA_API_KEY'),
            'weather': os.getenv('OPENWEATHER_API_KEY'),
            'college': os.getenv('CFBD_API_KEY')
        }
        
        # Stadium coordinates for weather data
        self.stadium_coords = {
            'ARI': (33.5276, -112.2626),
            'ATL': (33.7553, -84.4006),
            'BAL': (39.2780, -76.6227),
            'BUF': (42.7738, -78.7870),
            'CAR': (35.2258, -80.8528),
            'CHI': (41.8623, -87.6167),
            'CIN': (39.0954, -84.5160),
            'CLE': (41.5061, -81.6995),
            'DAL': (32.7473, -97.0945),
            'DEN': (39.7439, -105.0201),
            'DET': (42.3400, -83.0456),
            'GB': (44.5013, -88.0622),
            'HOU': (29.6847, -95.4107),
            'IND': (39.7601, -86.1639),
            'JAX': (30.3239, -81.6373),
            'KC': (39.0489, -94.4839),
            'LAC': (33.8643, -118.2611),
            'LAR': (33.9535, -118.3392),
            'LV': (36.0909, -115.1833),
            'MIA': (25.9580, -80.2389),
            'MIN': (44.9736, -93.2575),
            'NE': (42.0909, -71.2643),
            'NO': (29.9511, -90.0812),
            'NYG': (40.8128, -74.0742),
            'NYJ': (40.8135, -74.0745),
            'PHI': (39.9008, -75.1675),
            'PIT': (40.4468, -80.0158),
            'SEA': (47.5952, -122.3316),
            'SF': (37.7133, -122.3862),
            'TB': (27.9759, -82.5033),
            'TEN': (36.1665, -86.7713),
            'WAS': (38.9076, -76.8645)
        }
        
    async def collect_historical_nfl_data(self, years: int = 10) -> pd.DataFrame:
        """Collect 10 years of NFL data"""
        all_data = []
        current_year = datetime.now().year
        
        async with aiohttp.ClientSession() as session:
            for year in range(current_year - years, current_year + 1):
                logger.info(f"Collecting NFL data for {year} season")
                
                # Collect regular season data
                for week in range(1, 19):  # 18 weeks regular season
                    data = await self._fetch_nfl_week_data(session, year, week)
                    if data:
                        all_data.extend(data)
                
                # Collect playoff data
                for week in range(1, 5):  # Playoffs
                    data = await self._fetch_nfl_week_data(session, year, week, playoffs=True)
                    if data:
                        all_data.extend(data)
        
        return pd.DataFrame(all_data)
    
    async def _fetch_nfl_week_data(self, session: aiohttp.ClientSession, 
                                   year: int, week: int, playoffs: bool = False) -> List[Dict]:
        """Fetch NFL data for a specific week"""
        week_type = 'POST' if playoffs else 'REG'
        url = f"{self.base_urls['nfl_stats']}PlayerGameStatsByWeek/{year}/{week_type}/{week}"
        
        headers = {'Ocp-Apim-Subscription-Key': self.api_keys['sportsdata']}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Enhance with additional metrics
                    enhanced_data = []
                    for player in data:
                        enhanced = await self._enhance_player_data(session, player, year, week)
                        enhanced_data.append(enhanced)
                    return enhanced_data
                else:
                    logger.warning(f"Failed to fetch data for {year} week {week}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching NFL data: {e}")
            return []
    
    async def _enhance_player_data(self, session: aiohttp.ClientSession, 
                                   player_data: Dict, year: int, week: int) -> Dict:
        """Enhance player data with additional features"""
        enhanced = player_data.copy()
        
        # Add weather data
        if player_data.get('HomeOrAway') == 'HOME':
            team = player_data.get('Team')
            if team in self.stadium_coords:
                weather = await self._fetch_weather_data(
                    session, 
                    self.stadium_coords[team],
                    player_data.get('GameDate')
                )
                enhanced['weather'] = weather
        
        # Add opponent defensive strength
        opponent = player_data.get('Opponent')
        enhanced['opponent_def_rank'] = await self._get_defensive_rank(opponent, year, week)
        
        # Add injury status
        enhanced['injury_status'] = await self._get_injury_status(
            player_data.get('PlayerID'),
            year,
            week
        )
        
        return enhanced
    
    async def collect_college_data(self, player_name: str, college_year: int) -> Dict:
        """Collect college statistics for rookies"""
        url = f"{self.base_urls['college_stats']}stats/player/season"
        params = {
            'year': college_year,
            'searchTerm': player_name
        }
        headers = {'Authorization': f'Bearer {self.api_keys["college"]}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_college_stats(data)
                else:
                    logger.warning(f"Failed to fetch college data for {player_name}")
                    return {}
    
    def _process_college_stats(self, raw_stats: List[Dict]) -> Dict:
        """Process raw college statistics"""
        if not raw_stats:
            return {}
        
        # Aggregate stats across seasons
        processed = {
            'total_yards': sum(s.get('passingYards', 0) + s.get('rushingYards', 0) 
                              + s.get('receivingYards', 0) for s in raw_stats),
            'total_tds': sum(s.get('passingTDs', 0) + s.get('rushingTDs', 0) 
                            + s.get('receivingTDs', 0) for s in raw_stats),
            'games_played': sum(s.get('games', 0) for s in raw_stats),
            'completion_pct': np.mean([s.get('completionPercentage', 0) 
                                      for s in raw_stats if s.get('completionPercentage')]),
            'yards_per_attempt': np.mean([s.get('yardsPerAttempt', 0) 
                                         for s in raw_stats if s.get('yardsPerAttempt')])
        }
        
        return processed
    
    async def collect_combine_data(self, year: int) -> pd.DataFrame:
        """Collect NFL Combine data"""
        url = f"https://www.pro-football-reference.com/draft/{year}-combine.htm"
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the combine results table
            table = soup.find('table', {'id': 'combine'})
            if not table:
                return pd.DataFrame()
            
            # Parse table to DataFrame
            df = pd.read_html(str(table))[0]
            
            # Clean and process combine data
            combine_data = self._process_combine_data(df)
            return combine_data
            
        except Exception as e:
            logger.error(f"Error collecting combine data: {e}")
            return pd.DataFrame()
    
    def _process_combine_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean combine data"""
        # Select relevant columns
        relevant_cols = [
            'Player', 'Pos', 'School', 'Ht', 'Wt', '40yd', 'Bench', 
            'Vertical', 'Broad Jump', 'Shuttle', '3Cone'
        ]
        
        df_clean = df[relevant_cols].copy()
        
        # Convert height to inches
        if 'Ht' in df_clean.columns:
            df_clean['height_inches'] = df_clean['Ht'].apply(self._height_to_inches)
        
        # Clean numeric columns
        numeric_cols = ['Wt', '40yd', 'Bench', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _height_to_inches(self, height_str: str) -> Optional[int]:
        """Convert height string (e.g., '6-2') to inches"""
        if pd.isna(height_str) or not isinstance(height_str, str):
            return None
        
        parts = height_str.split('-')
        if len(parts) == 2:
            feet = int(parts[0])
            inches = int(parts[1])
            return feet * 12 + inches
        return None
    
    async def _fetch_weather_data(self, session: aiohttp.ClientSession,
                                  coords: Tuple[float, float], 
                                  game_date: str) -> Dict:
        """Fetch historical weather data for game location"""
        lat, lon = coords
        
        # Convert game date to timestamp
        game_dt = datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%S')
        timestamp = int(game_dt.timestamp())
        
        url = f"{self.base_urls['weather']}onecall/timemachine"
        params = {
            'lat': lat,
            'lon': lon,
            'dt': timestamp,
            'appid': self.api_keys['weather'],
            'units': 'imperial'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract relevant weather features
                    if 'current' in data:
                        weather = data['current']
                        return {
                            'temperature': weather.get('temp'),
                            'wind_speed': weather.get('wind_speed'),
                            'humidity': weather.get('humidity'),
                            'precipitation': weather.get('rain', {}).get('1h', 0),
                            'weather_condition': weather.get('weather', [{}])[0].get('main')
                        }
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
        
        return {}
    
    async def _get_defensive_rank(self, team: str, year: int, week: int) -> int:
        """Get team's defensive ranking at the time"""
        # This would connect to a stats API or database
        # For now, return a placeholder
        # In production, this would calculate DVOA or similar metric
        return np.random.randint(1, 33)
    
    async def _get_injury_status(self, player_id: str, year: int, week: int) -> str:
        """Get player's injury status for the week"""
        # This would connect to injury report API
        # For now, return random status for demonstration
        statuses = ['Healthy', 'Questionable', 'Doubtful', 'Out']
        weights = [0.7, 0.15, 0.1, 0.05]
        return np.random.choice(statuses, p=weights)
    
    async def collect_offensive_line_rankings(self, year: int) -> Dict[str, int]:
        """Collect offensive line rankings by team"""
        # This would scrape PFF or Football Outsiders O-line rankings
        # For now, return random rankings
        teams = list(self.stadium_coords.keys())
        rankings = list(range(1, len(teams) + 1))
        np.random.shuffle(rankings)
        
        return dict(zip(teams, rankings))
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced metrics from raw stats"""
        df = df.copy()
        
        # Efficiency metrics
        df['yards_per_touch'] = (df['PassingYards'] + df['RushingYards'] + 
                                 df['ReceivingYards']) / (df['PassingAttempts'] + 
                                 df['RushingAttempts'] + df['Receptions']).replace(0, 1)
        
        # Red zone efficiency
        df['rz_td_rate'] = df['RushingTouchdowns'] / df['RedZoneTargets'].replace(0, 1)
        
        # Target share
        df['target_share'] = df['Targets'] / df['TeamPassAttempts'].replace(0, 1)
        
        # Air yards share (for receivers)
        df['air_yards_share'] = df['AirYards'] / df['TeamAirYards'].replace(0, 1)
        
        # Opportunity score (touches + targets)
        df['opportunity_score'] = (df['RushingAttempts'] + df['Targets']) * df['SnapCountPercentage']
        
        # Consistency score (rolling std of fantasy points)
        df['consistency_score'] = df.groupby('PlayerID')['FantasyPoints'].transform(
            lambda x: x.rolling(window=4, min_periods=1).std()
        )
        
        return df
    
    async def build_complete_dataset(self) -> pd.DataFrame:
        """Build complete dataset with all features"""
        logger.info("Starting comprehensive data collection...")
        
        # Collect 10 years of NFL data
        nfl_data = await self.collect_historical_nfl_data(years=10)
        logger.info(f"Collected {len(nfl_data)} NFL game records")
        
        # Collect combine data for each year
        combine_dfs = []
        for year in range(2014, 2025):
            combine_df = await self.collect_combine_data(year)
            combine_df['draft_year'] = year
            combine_dfs.append(combine_df)
        
        combine_data = pd.concat(combine_dfs, ignore_index=True)
        logger.info(f"Collected {len(combine_data)} combine records")
        
        # Merge datasets
        merged_data = self._merge_all_data(nfl_data, combine_data)
        
        # Calculate advanced metrics
        final_data = self.calculate_advanced_metrics(merged_data)
        
        logger.info(f"Final dataset contains {len(final_data)} records with {len(final_data.columns)} features")
        
        return final_data
    
    def _merge_all_data(self, nfl_df: pd.DataFrame, combine_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources"""
        # This would be more sophisticated in production
        # For now, simple merge on player name
        merged = pd.merge(
            nfl_df,
            combine_df,
            left_on='Name',
            right_on='Player',
            how='left'
        )
        
        return merged


# Usage example
async def main():
    collector = EnhancedDataCollector()
    dataset = await collector.build_complete_dataset()
    
    # Save to parquet for efficient storage
    dataset.to_parquet('enhanced_nfl_dataset.parquet', index=False)
    logger.info("Dataset saved successfully")

if __name__ == "__main__":
    asyncio.run(main())