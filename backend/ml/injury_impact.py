"""
Injury Impact Calculator
Predicts fantasy impact of player injuries based on historical data and injury types
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from backend.models.database import SessionLocal, PlayerStats

logger = logging.getLogger(__name__)


@dataclass
class InjuryImpact:
    """Container for injury impact analysis"""
    player_id: str
    injury_type: str
    severity: str
    expected_games_missed: float
    return_performance_factor: float  # 0.0-1.0, 1.0 = full performance
    confidence: float
    historical_comparisons: List[Dict[str, Any]]
    recommendations: List[str]


class InjuryImpactCalculator:
    """
    Calculate fantasy impact of injuries using:
    1. Historical injury data and recovery patterns
    2. Position-specific impact analysis
    3. Age and usage-based recovery factors
    4. Return timeline predictions
    """
    
    def __init__(self):
        # Injury severity mappings
        self.injury_severity = {
            # Lower body injuries
            'hamstring': {'mild': 1.2, 'moderate': 2.8, 'severe': 6.5},
            'ankle': {'mild': 0.8, 'moderate': 2.1, 'severe': 8.2},
            'knee': {'mild': 1.5, 'moderate': 4.3, 'severe': 16.8},
            'quad': {'mild': 1.0, 'moderate': 2.5, 'severe': 5.2},
            'calf': {'mild': 0.7, 'moderate': 1.8, 'severe': 4.1},
            'foot': {'mild': 1.1, 'moderate': 3.2, 'severe': 8.9},
            'hip': {'mild': 1.8, 'moderate': 4.7, 'severe': 12.3},
            'achilles': {'mild': 2.5, 'moderate': 8.2, 'severe': 24.0},
            
            # Upper body injuries
            'shoulder': {'mild': 0.9, 'moderate': 2.4, 'severe': 8.7},
            'ribs': {'mild': 1.3, 'moderate': 3.1, 'severe': 6.2},
            'wrist': {'mild': 0.6, 'moderate': 1.9, 'severe': 5.4},
            'elbow': {'mild': 1.1, 'moderate': 2.8, 'severe': 7.6},
            'hand': {'mild': 0.8, 'moderate': 2.2, 'severe': 6.1},
            'thumb': {'mild': 0.5, 'moderate': 1.4, 'severe': 4.2},
            'finger': {'mild': 0.3, 'moderate': 0.8, 'severe': 2.1},
            
            # Head/neck injuries
            'concussion': {'mild': 1.2, 'moderate': 2.8, 'severe': 8.5},
            'neck': {'mild': 1.6, 'moderate': 4.2, 'severe': 12.1},
            
            # Core injuries
            'back': {'mild': 1.4, 'moderate': 3.6, 'severe': 9.8},
            'groin': {'mild': 1.0, 'moderate': 2.7, 'severe': 6.4}
        }
        
        # Position-specific injury impact multipliers
        self.position_impact = {
            'QB': {
                'throwing_arm': 1.8,
                'shoulder': 1.6,
                'ribs': 1.4,
                'ankle': 1.2,
                'knee': 1.3,
                'concussion': 1.5
            },
            'RB': {
                'ankle': 1.5,
                'knee': 1.7,
                'hamstring': 1.6,
                'shoulder': 0.9,
                'ribs': 1.1,
                'concussion': 1.3
            },
            'WR': {
                'hamstring': 1.6,
                'ankle': 1.4,
                'hand': 1.3,
                'shoulder': 1.2,
                'concussion': 1.4,
                'knee': 1.5
            },
            'TE': {
                'ankle': 1.3,
                'knee': 1.4,
                'shoulder': 1.3,
                'ribs': 1.2,
                'hand': 1.2,
                'concussion': 1.3
            }
        }
        
        # Recovery factors based on age
        self.age_recovery = {
            'young': 0.85,    # <25 years
            'prime': 1.0,     # 25-29 years
            'veteran': 1.25,  # 30-32 years
            'old': 1.5        # >32 years
        }
        
        # Historical injury patterns (mock data - would come from real database)
        self.historical_patterns = self._load_historical_patterns()
    
    def _load_historical_patterns(self) -> Dict[str, Dict]:
        """Load historical injury patterns (mock implementation)"""
        return {
            'hamstring': {
                'avg_games_missed': 2.3,
                'return_performance': 0.82,
                'reinjury_rate': 0.28,
                'similar_cases': [
                    {'player': 'Julio Jones', 'games_missed': 3, 'return_perf': 0.78},
                    {'player': 'Keenan Allen', 'games_missed': 2, 'return_perf': 0.85},
                    {'player': 'Adam Thielen', 'games_missed': 4, 'return_perf': 0.73}
                ]
            },
            'knee': {
                'avg_games_missed': 4.7,
                'return_performance': 0.74,
                'reinjury_rate': 0.31,
                'similar_cases': [
                    {'player': 'Saquon Barkley', 'games_missed': 16, 'return_perf': 0.71},
                    {'player': 'Christian McCaffrey', 'games_missed': 6, 'return_perf': 0.89},
                    {'player': 'Dalvin Cook', 'games_missed': 3, 'return_perf': 0.82}
                ]
            },
            'ankle': {
                'avg_games_missed': 2.1,
                'return_performance': 0.86,
                'reinjury_rate': 0.22,
                'similar_cases': [
                    {'player': 'Tyreek Hill', 'games_missed': 1, 'return_perf': 0.92},
                    {'player': 'Ezekiel Elliott', 'games_missed': 2, 'return_perf': 0.84},
                    {'player': 'DeAndre Hopkins', 'games_missed': 3, 'return_perf': 0.79}
                ]
            },
            'concussion': {
                'avg_games_missed': 1.8,
                'return_performance': 0.91,
                'reinjury_rate': 0.15,
                'similar_cases': [
                    {'player': 'Tua Tagovailoa', 'games_missed': 4, 'return_perf': 0.88},
                    {'player': 'Chris Godwin', 'games_missed': 1, 'return_perf': 0.94},
                    {'player': 'JuJu Smith-Schuster', 'games_missed': 2, 'return_perf': 0.89}
                ]
            }
        }
    
    def calculate_injury_impact(
        self,
        player_id: str,
        injury_type: str,
        severity: str = "moderate",
        player_age: Optional[int] = None,
        position: Optional[str] = None,
        usage_rate: Optional[float] = None,
        season: int = 2024
    ) -> InjuryImpact:
        """
        Calculate comprehensive injury impact analysis
        
        Args:
            player_id: Player identifier
            injury_type: Type of injury (e.g., 'hamstring', 'knee')
            severity: Injury severity ('mild', 'moderate', 'severe')
            player_age: Player age (if None, will look up)
            position: Player position (if None, will look up)
            usage_rate: Player usage rate (touches/targets per game)
            season: Current season
        
        Returns:
            InjuryImpact with comprehensive analysis
        """
        # Get player information if not provided
        if not player_age or not position:
            with SessionLocal() as db:
                from backend.models.database import Player
                player = db.query(Player).filter(Player.player_id == player_id).first()
                if player:
                    player_age = player_age or player.age or 27
                    position = position or player.position
                else:
                    player_age = 27  # Default age
                    position = "WR"  # Default position
        
        # Normalize injury type
        injury_type_normalized = injury_type.lower()
        
        # Get base injury impact
        base_games_missed = self._get_base_games_missed(injury_type_normalized, severity)
        base_performance_factor = self._get_base_performance_factor(injury_type_normalized, severity)
        
        # Apply position-specific adjustments
        position_multiplier = self._get_position_multiplier(injury_type_normalized, position)
        
        # Apply age-based recovery factor
        age_multiplier = self._get_age_multiplier(player_age)
        
        # Apply usage-based adjustment
        usage_multiplier = self._get_usage_multiplier(usage_rate)
        
        # Calculate final estimates
        expected_games_missed = base_games_missed * position_multiplier * age_multiplier * usage_multiplier
        return_performance_factor = base_performance_factor / (age_multiplier * 0.5 + 0.5)
        
        # Get historical comparisons
        historical_comparisons = self._get_historical_comparisons(
            injury_type_normalized, position, severity
        )
        
        # Calculate confidence based on available data
        confidence = self._calculate_confidence(
            injury_type_normalized, position, severity, historical_comparisons
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            injury_type_normalized, severity, expected_games_missed,
            return_performance_factor, position
        )
        
        return InjuryImpact(
            player_id=player_id,
            injury_type=injury_type,
            severity=severity,
            expected_games_missed=max(0, expected_games_missed),
            return_performance_factor=min(1.0, max(0.3, return_performance_factor)),
            confidence=confidence,
            historical_comparisons=historical_comparisons,
            recommendations=recommendations
        )
    
    def _get_base_games_missed(self, injury_type: str, severity: str) -> float:
        """Get base games missed for injury type and severity"""
        injury_data = self.injury_severity.get(injury_type, {})
        return injury_data.get(severity, 2.0)  # Default to 2 games
    
    def _get_base_performance_factor(self, injury_type: str, severity: str) -> float:
        """Get base performance factor upon return"""
        # Performance factors based on injury severity
        severity_factors = {
            'mild': 0.92,
            'moderate': 0.85,
            'severe': 0.72
        }
        
        # Injury-specific adjustments
        injury_adjustments = {
            'concussion': 0.95,  # Usually return to full performance
            'hamstring': 0.80,   # Often linger
            'knee': 0.75,        # Significant impact
            'achilles': 0.65,    # Major impact
            'ankle': 0.88,       # Moderate impact
            'shoulder': 0.85,    # Varies by position
        }
        
        base_factor = severity_factors.get(severity, 0.85)
        injury_factor = injury_adjustments.get(injury_type, 1.0)
        
        return base_factor * injury_factor
    
    def _get_position_multiplier(self, injury_type: str, position: str) -> float:
        """Get position-specific impact multiplier"""
        position_impacts = self.position_impact.get(position, {})
        return position_impacts.get(injury_type, 1.0)
    
    def _get_age_multiplier(self, age: int) -> float:
        """Get age-based recovery multiplier"""
        if age < 25:
            return self.age_recovery['young']
        elif age <= 29:
            return self.age_recovery['prime']
        elif age <= 32:
            return self.age_recovery['veteran']
        else:
            return self.age_recovery['old']
    
    def _get_usage_multiplier(self, usage_rate: Optional[float]) -> float:
        """Get usage-based adjustment (higher usage = longer recovery)"""
        if usage_rate is None:
            return 1.0
        
        # High usage players take longer to recover
        if usage_rate > 20:  # High usage
            return 1.15
        elif usage_rate > 15:  # Moderate usage
            return 1.05
        else:  # Low usage
            return 0.95
    
    def _get_historical_comparisons(
        self, injury_type: str, position: str, severity: str
    ) -> List[Dict[str, Any]]:
        """Get historical comparisons for similar injuries"""
        pattern = self.historical_patterns.get(injury_type, {})
        similar_cases = pattern.get('similar_cases', [])
        
        # Add some context to the comparisons
        for case in similar_cases:
            case['injury_type'] = injury_type
            case['severity_estimate'] = severity
            case['relevance_score'] = self._calculate_relevance_score(case, position)
        
        # Sort by relevance
        similar_cases.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return similar_cases[:5]  # Top 5 most relevant cases
    
    def _calculate_relevance_score(self, case: Dict[str, Any], position: str) -> float:
        """Calculate relevance score for historical case"""
        # Mock relevance calculation
        base_score = 0.7
        
        # Would factor in:
        # - Position similarity
        # - Age similarity
        # - Usage similarity
        # - Injury circumstances
        
        return base_score + np.random.normal(0, 0.1)
    
    def _calculate_confidence(
        self, injury_type: str, position: str, severity: str,
        historical_comparisons: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in injury impact prediction"""
        base_confidence = 0.6
        
        # Increase confidence if we have historical data
        if injury_type in self.historical_patterns:
            base_confidence += 0.2
        
        # Increase confidence with more historical comparisons
        if len(historical_comparisons) >= 3:
            base_confidence += 0.1
        
        # Adjust based on injury type predictability
        predictable_injuries = ['ankle', 'hamstring', 'concussion']
        if injury_type in predictable_injuries:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _generate_recommendations(
        self, injury_type: str, severity: str, games_missed: float,
        performance_factor: float, position: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Games missed recommendations
        if games_missed > 4:
            recommendations.append("Consider dropping or trading - extended absence expected")
        elif games_missed > 2:
            recommendations.append("Move to IR if available, seek replacement")
        elif games_missed > 0.5:
            recommendations.append("Monitor closely - may miss 1-2 games")
        else:
            recommendations.append("Should play through injury - minimal impact expected")
        
        # Performance recommendations
        if performance_factor < 0.7:
            recommendations.append("Expect significantly reduced performance upon return")
        elif performance_factor < 0.85:
            recommendations.append("Performance likely to be impacted for several weeks")
        else:
            recommendations.append("Should return close to full performance")
        
        # Injury-specific recommendations
        if injury_type == 'hamstring':
            recommendations.append("High re-injury risk - monitor snap counts carefully")
        elif injury_type == 'concussion':
            recommendations.append("Return timeline unpredictable - prepare backup options")
        elif injury_type == 'knee':
            recommendations.append("Recovery highly variable - wait for clear updates")
        
        # Position-specific recommendations
        if position == 'RB' and injury_type in ['ankle', 'knee', 'hamstring']:
            recommendations.append("Critical injury for RB - consider immediate replacement")
        elif position == 'WR' and injury_type == 'hamstring':
            recommendations.append("Route-running ability may be compromised")
        
        return recommendations
    
    def get_injury_timeline(
        self, player_id: str, injury_type: str, severity: str = "moderate"
    ) -> Dict[str, Any]:
        """Get detailed injury timeline with milestones"""
        impact = self.calculate_injury_impact(player_id, injury_type, severity)
        
        timeline = {
            'immediate': {
                'week': 0,
                'status': 'Injured',
                'availability': 0.0,
                'description': f'{severity.title()} {injury_type} injury sustained'
            },
            'short_term': {
                'week': 1,
                'status': 'Questionable' if impact.expected_games_missed < 1 else 'Out',
                'availability': 0.0 if impact.expected_games_missed >= 1 else 0.3,
                'description': 'Initial recovery phase, limited practice participation'
            },
            'medium_term': {
                'week': int(impact.expected_games_missed) + 1,
                'status': 'Probable',
                'availability': impact.return_performance_factor * 0.8,
                'description': 'Return to play, may be on snap count'
            },
            'long_term': {
                'week': int(impact.expected_games_missed) + 3,
                'status': 'Healthy',
                'availability': impact.return_performance_factor,
                'description': 'Full recovery expected'
            }
        }
        
        return {
            'player_id': player_id,
            'injury_details': {
                'type': injury_type,
                'severity': severity,
                'expected_games_missed': impact.expected_games_missed
            },
            'timeline': timeline,
            'risk_factors': {
                'reinjury_risk': self.historical_patterns.get(injury_type, {}).get('reinjury_rate', 0.2),
                'performance_decline': 1 - impact.return_performance_factor,
                'uncertainty_level': 1 - impact.confidence
            }
        }
    
    def compare_similar_injuries(
        self, injury_type: str, position: str, severity: str = "moderate"
    ) -> Dict[str, Any]:
        """Compare outcomes of similar injuries across players"""
        pattern = self.historical_patterns.get(injury_type, {})
        
        if not pattern:
            return {"error": "No historical data available for this injury type"}
        
        similar_cases = pattern.get('similar_cases', [])
        
        if not similar_cases:
            return {"error": "No similar cases found"}
        
        analysis = {
            'injury_type': injury_type,
            'position': position,
            'severity': severity,
            'historical_average': {
                'games_missed': pattern.get('avg_games_missed', 0),
                'return_performance': pattern.get('return_performance', 0.85),
                'reinjury_rate': pattern.get('reinjury_rate', 0.2)
            },
            'case_studies': similar_cases,
            'outcome_distribution': {
                'quick_recovery': len([c for c in similar_cases if c['games_missed'] <= 1]),
                'moderate_recovery': len([c for c in similar_cases if 1 < c['games_missed'] <= 4]),
                'slow_recovery': len([c for c in similar_cases if c['games_missed'] > 4])
            },
            'performance_outcomes': {
                'full_recovery': len([c for c in similar_cases if c['return_perf'] >= 0.9]),
                'partial_recovery': len([c for c in similar_cases if 0.75 <= c['return_perf'] < 0.9]),
                'poor_recovery': len([c for c in similar_cases if c['return_perf'] < 0.75])
            }
        }
        
        return analysis


# Example usage
if __name__ == "__main__":
    calculator = InjuryImpactCalculator()
    
    # Test injury impact calculation
    impact = calculator.calculate_injury_impact(
        player_id="6783",
        injury_type="hamstring",
        severity="moderate",
        player_age=26,
        position="WR"
    )
    
    print(f"Injury Impact Analysis:")
    print(f"Expected games missed: {impact.expected_games_missed:.1f}")
    print(f"Return performance factor: {impact.return_performance_factor:.2f}")
    print(f"Confidence: {impact.confidence:.2f}")
    print(f"\nRecommendations:")
    for rec in impact.recommendations:
        print(f"  - {rec}")
    
    # Test injury timeline
    timeline = calculator.get_injury_timeline("6783", "hamstring", "moderate")
    print(f"\nInjury Timeline:")
    for phase, details in timeline['timeline'].items():
        print(f"  {phase}: Week {details['week']} - {details['status']} ({details['availability']:.0%} availability)")