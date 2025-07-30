"""
Weather-Adjusted Fantasy Projections
Adjusts player projections based on weather conditions and historical performance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import requests
import json

logger = logging.getLogger(__name__)


@dataclass
class WeatherConditions:
    """Weather conditions for a game"""
    temperature: float  # Fahrenheit
    wind_speed: float   # mph
    precipitation: float  # inches
    humidity: float     # percentage
    weather_type: str   # clear, cloudy, rain, snow, etc.
    is_dome: bool       # Indoor stadium
    game_time: str      # day/night
    

@dataclass 
class WeatherAdjustment:
    """Weather adjustment factors for fantasy performance"""
    player_id: str
    original_projection: float
    weather_adjusted_projection: float
    adjustment_factor: float
    weather_conditions: WeatherConditions
    confidence: float
    historical_context: Dict[str, Any]
    explanation: List[str]


class WeatherProjectionAdjuster:
    """
    Adjust fantasy projections based on weather conditions using:
    1. Historical weather impact analysis
    2. Position-specific weather sensitivities
    3. Stadium and game-time factors
    4. Player-specific weather performance
    """
    
    def __init__(self):
        # Weather impact factors by position
        self.position_weather_sensitivity = {
            'QB': {
                'wind_sensitivity': -0.008,      # Points lost per mph of wind
                'rain_penalty': -0.15,          # Multiplier in rain
                'cold_penalty': -0.006,         # Points lost per degree below 45F
                'snow_penalty': -0.25,          # Multiplier in snow
                'dome_bonus': 0.05              # Bonus for dome games
            },
            'RB': {
                'wind_sensitivity': -0.002,     # Less affected by wind
                'rain_penalty': -0.05,          # Slight penalty in rain
                'cold_penalty': -0.003,         # Some impact from cold
                'snow_penalty': -0.10,          # Moderate snow impact
                'dome_bonus': 0.02
            },
            'WR': {
                'wind_sensitivity': -0.010,     # Most affected by wind
                'rain_penalty': -0.18,          # Significant rain penalty
                'cold_penalty': -0.008,         # Cold affects hands/catching
                'snow_penalty': -0.30,          # Severe snow impact
                'dome_bonus': 0.08
            },
            'TE': {
                'wind_sensitivity': -0.007,
                'rain_penalty': -0.12,
                'cold_penalty': -0.005,
                'snow_penalty': -0.20,
                'dome_bonus': 0.04
            },
            'K': {
                'wind_sensitivity': -0.015,     # Extremely wind sensitive
                'rain_penalty': -0.25,          # Rain affects accuracy
                'cold_penalty': -0.010,         # Cold affects leg strength
                'snow_penalty': -0.40,          # Snow severely impacts kicking
                'dome_bonus': 0.15              # Big dome advantage
            }
        }
        
        # Stadium information (mock data - would come from real database)
        self.stadium_info = {
            'GB': {'dome': False, 'cold_weather': True, 'wind_prone': True},
            'CHI': {'dome': False, 'cold_weather': True, 'wind_prone': True},
            'BUF': {'dome': False, 'cold_weather': True, 'wind_prone': False},
            'NE': {'dome': False, 'cold_weather': True, 'wind_prone': False},
            'DEN': {'dome': False, 'cold_weather': True, 'wind_prone': True},
            'KC': {'dome': False, 'cold_weather': True, 'wind_prone': False},
            'ATL': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'NO': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'LV': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'LAR': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'DET': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'MIN': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'ARI': {'dome': True, 'cold_weather': False, 'wind_prone': False},
            'IND': {'dome': True, 'cold_weather': False, 'wind_prone': False}
        }
        
        # Historical weather performance patterns
        self.weather_thresholds = {
            'wind': {
                'low': 5,      # mph
                'moderate': 15,
                'high': 25
            },
            'temperature': {
                'very_cold': 20,  # F
                'cold': 35,
                'cool': 50,
                'ideal': 75
            },
            'precipitation': {
                'light': 0.1,   # inches
                'moderate': 0.3,
                'heavy': 0.7
            }
        }
    
    def adjust_projections_for_weather(
        self,
        player_projections: Dict[str, float],  # {player_id: projected_points}
        game_weather: Dict[str, WeatherConditions],  # {game_id: weather}
        player_teams: Dict[str, str],  # {player_id: team}
        player_positions: Dict[str, str],  # {player_id: position}
        include_historical: bool = True
    ) -> Dict[str, WeatherAdjustment]:
        """
        Adjust multiple player projections for weather conditions
        
        Args:
            player_projections: Original projections by player
            game_weather: Weather conditions by game
            player_teams: Player team mappings
            player_positions: Player position mappings
            include_historical: Include historical weather analysis
        
        Returns:
            Weather adjustments for each player
        """
        adjustments = {}
        
        for player_id, original_projection in player_projections.items():
            try:
                team = player_teams.get(player_id)
                position = player_positions.get(player_id)
                
                if not team or not position:
                    continue
                
                # Find game weather (simplified - assumes team plays one game)
                game_weather_conditions = None
                for game_id, weather in game_weather.items():
                    if team in game_id:  # Simple team matching
                        game_weather_conditions = weather
                        break
                
                if not game_weather_conditions:
                    continue
                
                # Calculate weather adjustment
                adjustment = self._calculate_weather_adjustment(
                    player_id=player_id,
                    position=position,
                    team=team,
                    original_projection=original_projection,
                    weather=game_weather_conditions,
                    include_historical=include_historical
                )
                
                adjustments[player_id] = adjustment
                
            except Exception as e:
                logger.error(f"Error adjusting weather for player {player_id}: {str(e)}")
        
        return adjustments
    
    def _calculate_weather_adjustment(
        self,
        player_id: str,
        position: str,
        team: str,
        original_projection: float,
        weather: WeatherConditions,
        include_historical: bool = True
    ) -> WeatherAdjustment:
        """Calculate weather adjustment for a single player"""
        
        # Skip adjustment for dome games
        if weather.is_dome:
            dome_bonus = self.position_weather_sensitivity[position]['dome_bonus']
            adjusted_projection = original_projection * (1 + dome_bonus)
            
            return WeatherAdjustment(
                player_id=player_id,
                original_projection=original_projection,
                weather_adjusted_projection=adjusted_projection,
                adjustment_factor=1 + dome_bonus,
                weather_conditions=weather,
                confidence=0.95,
                historical_context={},
                explanation=[f"Dome game provides {dome_bonus:.1%} bonus for {position}"]
            )
        
        # Get position sensitivity
        sensitivity = self.position_weather_sensitivity.get(position, {})
        
        # Calculate individual weather impacts
        adjustments = []
        explanations = []
        
        # Wind impact
        if weather.wind_speed > self.weather_thresholds['wind']['low']:
            wind_impact = weather.wind_speed * sensitivity.get('wind_sensitivity', 0)
            adjustments.append(wind_impact)
            
            if weather.wind_speed > self.weather_thresholds['wind']['high']:
                explanations.append(f"High winds ({weather.wind_speed} mph) significantly impact {position}")
            elif weather.wind_speed > self.weather_thresholds['wind']['moderate']:
                explanations.append(f"Moderate winds ({weather.wind_speed} mph) affect {position}")
        
        # Temperature impact
        if weather.temperature < self.weather_thresholds['temperature']['cool']:
            temp_impact = (weather.temperature - 45) * sensitivity.get('cold_penalty', 0)
            adjustments.append(temp_impact)
            
            if weather.temperature < self.weather_thresholds['temperature']['very_cold']:
                explanations.append(f"Very cold conditions ({weather.temperature}°F) severely impact performance")
            elif weather.temperature < self.weather_thresholds['temperature']['cold']:
                explanations.append(f"Cold conditions ({weather.temperature}°F) reduce effectiveness")
        
        # Precipitation impact
        precipitation_penalty = 0
        if weather.precipitation > self.weather_thresholds['precipitation']['light']:
            if weather.weather_type.lower() == 'snow':
                precipitation_penalty = sensitivity.get('snow_penalty', 0)
                explanations.append(f"Snow conditions severely impact {position} performance")
            else:
                precipitation_penalty = sensitivity.get('rain_penalty', 0)
                if weather.precipitation > self.weather_thresholds['precipitation']['heavy']:
                    explanations.append(f"Heavy rain ({weather.precipitation}in) significantly affects {position}")
                else:
                    explanations.append(f"Rain conditions ({weather.precipitation}in) impact {position}")
        
        # Calculate total adjustment
        additive_adjustment = sum(adjustments)  # Wind and temperature
        multiplicative_adjustment = 1 + precipitation_penalty  # Rain/snow
        
        # Apply adjustments
        temp_projection = original_projection + additive_adjustment
        final_projection = temp_projection * multiplicative_adjustment
        
        # Ensure projection doesn't go negative or unreasonably high
        final_projection = max(0.1, min(final_projection, original_projection * 1.5))
        
        adjustment_factor = final_projection / original_projection if original_projection > 0 else 1
        
        # Get historical context if requested
        historical_context = {}
        if include_historical:
            historical_context = self._get_historical_weather_context(
                position, weather
            )
        
        # Calculate confidence
        confidence = self._calculate_weather_confidence(weather, position)
        
        return WeatherAdjustment(
            player_id=player_id,
            original_projection=original_projection,
            weather_adjusted_projection=final_projection,
            adjustment_factor=adjustment_factor,
            weather_conditions=weather,
            confidence=confidence,
            historical_context=historical_context,
            explanation=explanations or ["Weather conditions are favorable"]
        )
    
    def _get_historical_weather_context(
        self, position: str, weather: WeatherConditions
    ) -> Dict[str, Any]:
        """Get historical context for similar weather conditions"""
        # Mock historical analysis - would query real database
        context = {
            'similar_conditions_count': np.random.randint(5, 25),
            'avg_performance_change': np.random.normal(-0.05, 0.1),
            'best_case_performance': np.random.normal(0.1, 0.05),
            'worst_case_performance': np.random.normal(-0.2, 0.05),
            'position_rank_impact': np.random.choice(['minimal', 'moderate', 'significant'])
        }
        
        # Add specific insights based on weather
        if weather.wind_speed > 20:
            context['wind_insight'] = f"In {weather.wind_speed}+ mph winds, {position}s average 12% fewer fantasy points"
        
        if weather.temperature < 30:
            context['cold_insight'] = f"In sub-30°F games, {position}s have 15% more fumbles/drops"
        
        if weather.precipitation > 0.3:
            context['rain_insight'] = f"In heavy precipitation, {position}s see 18% reduction in big plays"
        
        return context
    
    def _calculate_weather_confidence(
        self, weather: WeatherConditions, position: str
    ) -> float:
        """Calculate confidence in weather adjustment"""
        base_confidence = 0.7
        
        # Higher confidence for extreme conditions
        if weather.wind_speed > 25:
            base_confidence += 0.15
        elif weather.wind_speed > 15:
            base_confidence += 0.08
        
        if weather.temperature < 25:
            base_confidence += 0.10
        
        if weather.precipitation > 0.5:
            base_confidence += 0.12
        
        # Position-specific confidence
        if position in ['WR', 'K']:  # Most weather-sensitive
            base_confidence += 0.05
        elif position == 'RB':  # Least weather-sensitive
            base_confidence -= 0.05
        
        return min(0.95, base_confidence)
    
    def get_weather_forecast(
        self, team: str, game_date: datetime
    ) -> Optional[WeatherConditions]:
        """
        Get weather forecast for team's game
        (Mock implementation - would integrate with weather API)
        """
        # Check if dome
        stadium = self.stadium_info.get(team, {})
        if stadium.get('dome', False):
            return WeatherConditions(
                temperature=72.0,
                wind_speed=0.0,
                precipitation=0.0,
                humidity=50.0,
                weather_type='dome',
                is_dome=True,
                game_time='day'
            )
        
        # Mock weather data
        base_temp = 45 if stadium.get('cold_weather') else 70
        temp_variation = np.random.normal(0, 15)
        
        return WeatherConditions(
            temperature=base_temp + temp_variation,
            wind_speed=max(0, np.random.normal(8, 5)),
            precipitation=max(0, np.random.exponential(0.1)),
            humidity=max(30, min(90, np.random.normal(60, 15))),
            weather_type=np.random.choice(['clear', 'cloudy', 'rain', 'snow'], p=[0.4, 0.3, 0.2, 0.1]),
            is_dome=False,
            game_time=np.random.choice(['day', 'night'], p=[0.7, 0.3])
        )
    
    def analyze_weather_trends(
        self, position: str, weeks_back: int = 4
    ) -> Dict[str, Any]:
        """Analyze recent weather trends and their fantasy impact"""
        # Mock analysis - would query historical data
        return {
            'position': position,
            'weeks_analyzed': weeks_back,
            'weather_games': {
                'total': np.random.randint(15, 30),
                'dome': np.random.randint(8, 15),
                'outdoor': np.random.randint(7, 15),
                'adverse_weather': np.random.randint(3, 8)
            },
            'performance_impacts': {
                'avg_weather_penalty': np.random.normal(-0.08, 0.03),
                'wind_games_penalty': np.random.normal(-0.12, 0.04),
                'cold_games_penalty': np.random.normal(-0.06, 0.02),
                'rain_games_penalty': np.random.normal(-0.15, 0.05)
            },
            'notable_games': [
                {
                    'week': 8,
                    'conditions': '25mph winds',
                    'impact': 'WRs averaged 2.3 fewer points'
                },
                {
                    'week': 6,
                    'conditions': 'Heavy rain',
                    'impact': 'Passing games down 18%'
                }
            ],
            'upcoming_concerns': [
                'Week 12: Several cold-weather games expected',
                'Week 14: Wind advisories in multiple cities'
            ]
        }
    
    def get_weather_stack_recommendations(
        self, weather_conditions: Dict[str, WeatherConditions]
    ) -> Dict[str, List[str]]:
        """Get stacking recommendations based on weather"""
        recommendations = {
            'favorable_stacks': [],
            'avoid_stacks': [],
            'weather_plays': []
        }
        
        for game_id, weather in weather_conditions.items():
            if weather.is_dome:
                recommendations['favorable_stacks'].append(
                    f"{game_id}: Dome game - excellent for passing stacks"
                )
            elif weather.wind_speed > 20:
                recommendations['avoid_stacks'].append(
                    f"{game_id}: High winds ({weather.wind_speed} mph) - avoid passing stacks"
                )
                recommendations['weather_plays'].append(
                    f"{game_id}: Target RBs and TEs in windy conditions"
                )
            elif weather.precipitation > 0.3:
                recommendations['avoid_stacks'].append(
                    f"{game_id}: Heavy precipitation - reduced passing volume expected"
                )
                recommendations['weather_plays'].append(
                    f"{game_id}: Ground game emphasis likely"
                )
            elif weather.temperature < 25:
                recommendations['weather_plays'].append(
                    f"{game_id}: Very cold - potential for low-scoring game"
                )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    adjuster = WeatherProjectionAdjuster()
    
    # Mock data
    player_projections = {
        "6783": 14.5,  # Jerry Jeudy
        "4035": 16.8,  # Alvin Kamara
        "5849": 22.3   # Kyler Murray
    }
    
    player_teams = {
        "6783": "DEN",
        "4035": "NO",
        "5849": "ARI"
    }
    
    player_positions = {
        "6783": "WR",
        "4035": "RB", 
        "5849": "QB"
    }
    
    game_weather = {
        "DEN_KC": WeatherConditions(
            temperature=28.0,
            wind_speed=18.0,
            precipitation=0.0,
            humidity=45.0,
            weather_type='clear',
            is_dome=False,
            game_time='day'
        ),
        "NO_ATL": WeatherConditions(
            temperature=72.0,
            wind_speed=0.0,
            precipitation=0.0,
            humidity=50.0,
            weather_type='dome',
            is_dome=True,
            game_time='day'
        ),
        "ARI_SEA": WeatherConditions(
            temperature=72.0,
            wind_speed=0.0,
            precipitation=0.0,
            humidity=50.0,
            weather_type='dome',
            is_dome=True,
            game_time='day'
        )
    }
    
    # Get weather adjustments
    adjustments = adjuster.adjust_projections_for_weather(
        player_projections,
        game_weather,
        player_teams,
        player_positions
    )
    
    for player_id, adj in adjustments.items():
        print(f"\nPlayer {player_id}:")
        print(f"  Original: {adj.original_projection:.1f}")
        print(f"  Adjusted: {adj.weather_adjusted_projection:.1f}")
        print(f"  Factor: {adj.adjustment_factor:.3f}")
        print(f"  Confidence: {adj.confidence:.2f}")
        if adj.explanation:
            print(f"  Explanation: {adj.explanation[0]}")