"""
Open-Meteo Weather Client for NFL Stadium Weather Data
Free Tier - Commercial Use Allowed
"""

import os
import redis
import json
import logging
import httpx
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StadiumLocation:
    """NFL Stadium location data"""
    name: str
    city: str
    state: str
    latitude: float
    longitude: float
    dome: bool  # Indoor stadium
    
    
# NFL Stadium Coordinates (2024 Season)
NFL_STADIUMS = {
    'ARI': StadiumLocation('State Farm Stadium', 'Glendale', 'AZ', 33.5276, -112.2626, True),
    'ATL': StadiumLocation('Mercedes-Benz Stadium', 'Atlanta', 'GA', 33.7553, -84.4006, True),
    'BAL': StadiumLocation('M&T Bank Stadium', 'Baltimore', 'MD', 39.2780, -76.6227, False),
    'BUF': StadiumLocation('Highmark Stadium', 'Orchard Park', 'NY', 42.7738, -78.7870, False),
    'CAR': StadiumLocation('Bank of America Stadium', 'Charlotte', 'NC', 35.2258, -80.8528, False),
    'CHI': StadiumLocation('Soldier Field', 'Chicago', 'IL', 41.8623, -87.6167, False),
    'CIN': StadiumLocation('Paycor Stadium', 'Cincinnati', 'OH', 39.0954, -84.5160, False),
    'CLE': StadiumLocation('Cleveland Browns Stadium', 'Cleveland', 'OH', 41.5061, -81.6995, False),
    'DAL': StadiumLocation('AT&T Stadium', 'Arlington', 'TX', 32.7473, -97.0945, True),
    'DEN': StadiumLocation('Empower Field', 'Denver', 'CO', 39.7439, -105.0201, False),
    'DET': StadiumLocation('Ford Field', 'Detroit', 'MI', 42.3400, -83.0456, True),
    'GB': StadiumLocation('Lambeau Field', 'Green Bay', 'WI', 44.5013, -88.0622, False),
    'HOU': StadiumLocation('NRG Stadium', 'Houston', 'TX', 29.6847, -95.4107, True),
    'IND': StadiumLocation('Lucas Oil Stadium', 'Indianapolis', 'IN', 39.7601, -86.1639, True),
    'JAX': StadiumLocation('TIAA Bank Field', 'Jacksonville', 'FL', 30.3239, -81.6373, False),
    'KC': StadiumLocation('Arrowhead Stadium', 'Kansas City', 'MO', 39.0489, -94.4839, False),
    'LA': StadiumLocation('SoFi Stadium', 'Inglewood', 'CA', 33.9535, -118.3392, True),  # Rams
    'LAC': StadiumLocation('SoFi Stadium', 'Inglewood', 'CA', 33.9535, -118.3392, True),  # Chargers
    'LV': StadiumLocation('Allegiant Stadium', 'Las Vegas', 'NV', 36.0909, -115.1833, True),
    'MIA': StadiumLocation('Hard Rock Stadium', 'Miami Gardens', 'FL', 25.9580, -80.2389, False),
    'MIN': StadiumLocation('U.S. Bank Stadium', 'Minneapolis', 'MN', 44.9736, -93.2575, True),
    'NE': StadiumLocation('Gillette Stadium', 'Foxborough', 'MA', 42.0909, -71.2643, False),
    'NO': StadiumLocation('Caesars Superdome', 'New Orleans', 'LA', 29.9511, -90.0812, True),
    'NYG': StadiumLocation('MetLife Stadium', 'East Rutherford', 'NJ', 40.8135, -74.0745, False),
    'NYJ': StadiumLocation('MetLife Stadium', 'East Rutherford', 'NJ', 40.8135, -74.0745, False),
    'PHI': StadiumLocation('Lincoln Financial Field', 'Philadelphia', 'PA', 39.9012, -75.1675, False),
    'PIT': StadiumLocation('Acrisure Stadium', 'Pittsburgh', 'PA', 40.4468, -80.0158, False),
    'SEA': StadiumLocation('Lumen Field', 'Seattle', 'WA', 47.5952, -122.3316, False),
    'SF': StadiumLocation("Levi's Stadium", 'Santa Clara', 'CA', 37.4033, -121.9694, False),
    'TB': StadiumLocation('Raymond James Stadium', 'Tampa', 'FL', 27.9759, -82.5033, False),
    'TEN': StadiumLocation('Nissan Stadium', 'Nashville', 'TN', 36.1665, -86.7713, False),
    'WAS': StadiumLocation('FedEx Field', 'Landover', 'MD', 38.9076, -76.8645, False),
}


class WeatherClient:
    """
    Open-Meteo Weather API Client
    Free tier allows commercial use with no API key required
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self):
        """Initialize weather client with Redis caching"""
        # Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Cache TTL
        self.WEATHER_CACHE_TTL = int(os.getenv('WEATHER_CACHE_TTL', 86400))  # 24 hours
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("Weather client initialized (Open-Meteo - no API key required)")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        
    def _get_cache_key(self, team: str, date: str) -> str:
        """Generate cache key for weather data"""
        return f"weather:{team}:{date}"
        
    async def get_game_weather(self, home_team: str, game_date: datetime, 
                              game_time_hour: int = 13) -> Dict[str, Any]:
        """
        Get weather data for a game
        
        Args:
            home_team: Home team abbreviation (e.g., 'GB', 'BUF')
            game_date: Date of the game
            game_time_hour: Hour of game start (24-hour format, default 1 PM)
            
        Returns:
            Weather data dict or None if indoor stadium
        """
        # Check if stadium is indoor
        stadium = NFL_STADIUMS.get(home_team)
        if not stadium:
            logger.warning(f"Unknown team: {home_team}")
            return {}
            
        if stadium.dome:
            return {
                'temperature': 72.0,  # Standard indoor temp
                'wind_speed': 0.0,
                'precipitation': 0.0,
                'humidity': 50.0,
                'weather_impact': 'none',
                'indoor': True
            }
            
        # Check cache
        date_str = game_date.strftime('%Y-%m-%d')
        cache_key = self._get_cache_key(home_team, date_str)
        
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
            
        # Fetch weather data
        weather_data = await self._fetch_weather(
            stadium.latitude, 
            stadium.longitude, 
            game_date,
            game_time_hour
        )
        
        # Cache the result
        self.redis_client.setex(cache_key, self.WEATHER_CACHE_TTL, json.dumps(weather_data))
        
        return weather_data
        
    async def _fetch_weather(self, latitude: float, longitude: float, 
                           date: datetime, hour: int) -> Dict[str, Any]:
        """
        Fetch weather data from Open-Meteo
        
        Args:
            latitude: Stadium latitude
            longitude: Stadium longitude
            date: Game date
            hour: Game hour
            
        Returns:
            Weather data dict
        """
        # Determine if we need forecast or historical data
        today = datetime.now().date()
        is_future = date.date() >= today
        
        if is_future:
            # Use forecast API
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'hourly': 'temperature_2m,windspeed_10m,precipitation,relativehumidity_2m',
                'temperature_unit': 'fahrenheit',
                'windspeed_unit': 'mph',
                'precipitation_unit': 'inch',
                'timezone': 'America/New_York',
                'start_date': date.strftime('%Y-%m-%d'),
                'end_date': date.strftime('%Y-%m-%d')
            }
            url = self.BASE_URL
        else:
            # Use archive API for historical data
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'hourly': 'temperature_2m,windspeed_10m,precipitation,relativehumidity_2m',
                'temperature_unit': 'fahrenheit',
                'windspeed_unit': 'mph',
                'precipitation_unit': 'inch',
                'timezone': 'America/New_York',
                'start_date': date.strftime('%Y-%m-%d'),
                'end_date': date.strftime('%Y-%m-%d')
            }
            url = self.ARCHIVE_URL
            
        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract hourly data for game time
            hourly_data = data.get('hourly', {})
            
            # Find the index for the game hour
            hour_index = hour
            
            weather = {
                'temperature': hourly_data.get('temperature_2m', [72])[hour_index],
                'wind_speed': hourly_data.get('windspeed_10m', [0])[hour_index],
                'precipitation': hourly_data.get('precipitation', [0])[hour_index],
                'humidity': hourly_data.get('relativehumidity_2m', [50])[hour_index],
                'indoor': False
            }
            
            # Calculate weather impact
            weather['weather_impact'] = self._calculate_weather_impact(weather)
            
            return weather
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            # Return neutral weather on error
            return {
                'temperature': 65.0,
                'wind_speed': 5.0,
                'precipitation': 0.0,
                'humidity': 50.0,
                'weather_impact': 'minimal',
                'indoor': False
            }
            
    def _calculate_weather_impact(self, weather: Dict[str, float]) -> str:
        """
        Calculate weather impact on game
        
        Args:
            weather: Weather data dict
            
        Returns:
            Impact level: 'none', 'minimal', 'moderate', 'significant', 'severe'
        """
        if weather.get('indoor', False):
            return 'none'
            
        temp = weather.get('temperature', 65)
        wind = weather.get('wind_speed', 0)
        precip = weather.get('precipitation', 0)
        
        # Temperature impact
        temp_impact = 0
        if temp < 20 or temp > 90:
            temp_impact = 3
        elif temp < 32 or temp > 85:
            temp_impact = 2
        elif temp < 40 or temp > 80:
            temp_impact = 1
            
        # Wind impact
        wind_impact = 0
        if wind > 25:
            wind_impact = 3
        elif wind > 15:
            wind_impact = 2
        elif wind > 10:
            wind_impact = 1
            
        # Precipitation impact
        precip_impact = 0
        if precip > 0.5:
            precip_impact = 3
        elif precip > 0.25:
            precip_impact = 2
        elif precip > 0.1:
            precip_impact = 1
            
        # Total impact
        total_impact = temp_impact + wind_impact + precip_impact
        
        if total_impact >= 6:
            return 'severe'
        elif total_impact >= 4:
            return 'significant'
        elif total_impact >= 2:
            return 'moderate'
        elif total_impact >= 1:
            return 'minimal'
        else:
            return 'none'
            
    async def get_historical_weather_batch(self, games: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get weather data for multiple historical games
        
        Args:
            games: List of dicts with 'home_team', 'game_date', 'game_time_hour'
            
        Returns:
            DataFrame with weather data
        """
        weather_data = []
        
        for game in games:
            weather = await self.get_game_weather(
                game['home_team'],
                game['game_date'],
                game.get('game_time_hour', 13)
            )
            
            weather_data.append({
                'home_team': game['home_team'],
                'game_date': game['game_date'],
                **weather
            })
            
        return pd.DataFrame(weather_data)
        
    def get_stadium_info(self) -> pd.DataFrame:
        """
        Get DataFrame of all NFL stadium information
        
        Returns:
            DataFrame with stadium data
        """
        stadium_data = []
        
        for team, stadium in NFL_STADIUMS.items():
            stadium_data.append({
                'team': team,
                'stadium_name': stadium.name,
                'city': stadium.city,
                'state': stadium.state,
                'latitude': stadium.latitude,
                'longitude': stadium.longitude,
                'dome': stadium.dome
            })
            
        return pd.DataFrame(stadium_data)


# Example usage
async def test_weather_client():
    """Test the weather client"""
    async with WeatherClient() as client:
        # Test single game weather
        weather = await client.get_game_weather(
            'GB',  # Green Bay (outdoor stadium)
            datetime(2024, 1, 14),  # January playoff game
            13  # 1 PM
        )
        print("Green Bay January weather:", weather)
        
        # Test indoor stadium
        weather = await client.get_game_weather(
            'MIN',  # Minnesota (indoor stadium)
            datetime(2024, 1, 14),
            13
        )
        print("Minnesota indoor weather:", weather)
        
        # Get stadium info
        stadiums = client.get_stadium_info()
        print(f"\nTotal stadiums: {len(stadiums)}")
        print(f"Indoor stadiums: {stadiums['dome'].sum()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_weather_client())