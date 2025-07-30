"""
Transparency Engine for Fantasy Football AI
Provides plain English explanations for ML predictions
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PredictionExplanation:
    """Structured explanation for a prediction"""
    summary: str
    confidence_level: str
    key_factors: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendation: str


class TransparencyEngine:
    """
    Converts ML model outputs into human-readable explanations
    Focuses on WHY predictions were made, not just WHAT was predicted
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.80,
            'medium': 0.60,
            'low': 0.0
        }
        
        self.impact_phrases = {
            'positive': [
                "boosting projection",
                "contributing positively",
                "increasing confidence",
                "supporting higher output"
            ],
            'negative': [
                "limiting projection",
                "causing concern",
                "reducing confidence",
                "suggesting caution"
            ],
            'neutral': [
                "maintaining baseline",
                "showing stability",
                "indicating consistency"
            ]
        }
    
    def explain_prediction(
        self,
        player_name: str,
        position: str,
        predicted_points: float,
        confidence_score: float,
        trend_analysis: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        matchup_data: Optional[Dict[str, Any]] = None
    ) -> PredictionExplanation:
        """Generate comprehensive explanation for a prediction"""
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        # Extract key factors
        key_factors = self._identify_key_factors(
            trend_analysis, feature_importance, matchup_data, position
        )
        
        # Assess risk
        risk_assessment = self._assess_risk(
            trend_analysis, confidence_score, position
        )
        
        # Generate summary
        summary = self._generate_summary(
            player_name, predicted_points, confidence_level, key_factors
        )
        
        # Create recommendation
        recommendation = self._generate_recommendation(
            predicted_points, confidence_level, risk_assessment, position
        )
        
        return PredictionExplanation(
            summary=summary,
            confidence_level=confidence_level,
            key_factors=key_factors,
            risk_assessment=risk_assessment,
            recommendation=recommendation
        )
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric confidence to human-readable level"""
        if score >= self.confidence_thresholds['high']:
            return "High"
        elif score >= self.confidence_thresholds['medium']:
            return "Medium"
        else:
            return "Low"
    
    def _identify_key_factors(
        self,
        trend_analysis: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]],
        matchup_data: Optional[Dict[str, Any]],
        position: str
    ) -> List[Dict[str, Any]]:
        """Extract and explain the most important factors"""
        factors = []
        
        # Recent performance trend
        if trend_analysis and 'performance_trend' in trend_analysis:
            trend = trend_analysis['performance_trend']
            trend_type = trend.get('overall_trend', 'stable')
            
            if trend_type == 'improving':
                explanation = f"Recent performance trending upward with {trend.get('last_3_games_avg', 0):.1f} point average over last 3 games"
                impact = "positive"
            elif trend_type == 'declining':
                explanation = f"Performance declining recently, averaging {trend.get('last_3_games_avg', 0):.1f} points in last 3 games"
                impact = "negative"
            else:
                explanation = f"Consistent performance with {trend.get('last_3_games_avg', 0):.1f} point average recently"
                impact = "neutral"
            
            factors.append({
                'factor': 'Recent Performance',
                'explanation': explanation,
                'impact': impact,
                'weight': 'high'
            })
        
        # Current form (hot/cold streaks)
        if trend_analysis and 'hot_cold_streaks' in trend_analysis:
            streaks = trend_analysis['hot_cold_streaks']
            form = streaks.get('current_form', 'Neutral')
            
            if form == 'Hot':
                explanation = f"Currently on a hot streak, exceeding expectations in {streaks.get('games_above_avg', 0)} of last 5 games"
                impact = "positive"
            elif form == 'Cold':
                explanation = f"In a cold streak, underperforming in recent games"
                impact = "negative"
            else:
                explanation = "Performing at expected levels with no significant streaks"
                impact = "neutral"
            
            factors.append({
                'factor': 'Current Form',
                'explanation': explanation,
                'impact': impact,
                'weight': 'high'
            })
        
        # Consistency
        if trend_analysis and 'consistency_metrics' in trend_analysis:
            consistency = trend_analysis['consistency_metrics']
            rating = consistency.get('consistency_rating', 'Unknown')
            cv = consistency.get('coefficient_of_variation', 0)
            
            if rating == 'Very Consistent':
                explanation = f"Highly reliable performer with only {cv:.1%} variation in scoring"
                impact = "positive"
            elif rating == 'Consistent':
                explanation = f"Generally consistent with {cv:.1%} scoring variation"
                impact = "positive"
            elif rating == 'Volatile':
                explanation = f"High variance player with {cv:.1%} scoring variation - boom or bust potential"
                impact = "negative"
            else:
                explanation = "Moderate consistency in recent performances"
                impact = "neutral"
            
            factors.append({
                'factor': 'Consistency',
                'explanation': explanation,
                'impact': impact,
                'weight': 'medium'
            })
        
        # Matchup analysis (if available)
        if matchup_data:
            opponent_rank = matchup_data.get('opponent_rank_vs_position', 16)
            
            if opponent_rank <= 5:
                explanation = f"Facing {opponent_rank}th ranked defense against {position}s - tough matchup"
                impact = "negative"
            elif opponent_rank >= 28:
                explanation = f"Facing {opponent_rank}th ranked defense against {position}s - favorable matchup"
                impact = "positive"
            else:
                explanation = f"Facing {opponent_rank}th ranked defense - neutral matchup"
                impact = "neutral"
            
            factors.append({
                'factor': 'Matchup',
                'explanation': explanation,
                'impact': impact,
                'weight': 'medium'
            })
        
        # Position-specific factors
        if position == 'RB' and trend_analysis:
            if 'workload_metrics' in trend_analysis:
                workload = trend_analysis['workload_metrics']
                touches = workload.get('avg_touches', 0)
                
                if touches > 20:
                    explanation = f"Heavy workload with {touches:.1f} touches per game"
                    impact = "positive"
                elif touches < 10:
                    explanation = f"Limited workload with only {touches:.1f} touches per game"
                    impact = "negative"
                else:
                    explanation = f"Moderate workload with {touches:.1f} touches per game"
                    impact = "neutral"
                
                factors.append({
                    'factor': 'Workload',
                    'explanation': explanation,
                    'impact': impact,
                    'weight': 'high'
                })
        
        return factors[:3]  # Return top 3 factors
    
    def _assess_risk(
        self,
        trend_analysis: Dict[str, Any],
        confidence_score: float,
        position: str
    ) -> Dict[str, Any]:
        """Assess risk factors for the prediction"""
        risk_level = "Medium"  # Default
        risk_factors = []
        
        # Volatility risk
        if trend_analysis and 'consistency_metrics' in trend_analysis:
            consistency = trend_analysis['consistency_metrics']
            if consistency.get('consistency_rating') == 'Volatile':
                risk_factors.append("High scoring variance")
                risk_level = "High"
        
        # Confidence-based risk
        if confidence_score < 0.6:
            risk_factors.append("Lower prediction confidence")
            risk_level = "High"
        elif confidence_score > 0.8:
            risk_level = "Low"
        
        # Bust probability
        if trend_analysis and 'bust_probability' in trend_analysis:
            bust_prob = trend_analysis['bust_probability']
            if bust_prob > 0.3:
                risk_factors.append(f"{bust_prob:.0%} chance of significant underperformance")
                risk_level = "High"
        
        return {
            'level': risk_level,
            'factors': risk_factors,
            'bust_probability': trend_analysis.get('bust_probability', 0.15) if trend_analysis else 0.15
        }
    
    def _generate_summary(
        self,
        player_name: str,
        predicted_points: float,
        confidence_level: str,
        key_factors: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language summary"""
        
        # Find the most impactful positive/negative factor
        positive_factors = [f for f in key_factors if f['impact'] == 'positive']
        negative_factors = [f for f in key_factors if f['impact'] == 'negative']
        
        summary_parts = [f"Projecting {predicted_points:.1f} points for {player_name}"]
        
        if confidence_level == "High":
            summary_parts.append("with strong confidence")
        elif confidence_level == "Low":
            summary_parts.append("with some uncertainty")
        
        if positive_factors:
            summary_parts.append(f"based on {positive_factors[0]['factor'].lower()}")
        
        if negative_factors:
            summary_parts.append(f"despite {negative_factors[0]['factor'].lower()}")
        
        return " ".join(summary_parts) + "."
    
    def _generate_recommendation(
        self,
        predicted_points: float,
        confidence_level: str,
        risk_assessment: Dict[str, Any],
        position: str
    ) -> str:
        """Generate actionable recommendation"""
        
        # Position-based thresholds for "good" performance
        good_thresholds = {
            'QB': 18,
            'RB': 12,
            'WR': 12,
            'TE': 8,
            'K': 8,
            'DEF': 8
        }
        
        threshold = good_thresholds.get(position, 10)
        risk_level = risk_assessment['level']
        
        if predicted_points >= threshold * 1.2:
            if risk_level == "Low":
                return "Strong start recommendation - high upside with minimal risk"
            elif risk_level == "High":
                return "Start with awareness of volatility - high ceiling but risky"
            else:
                return "Confident start - projected for above-average performance"
        
        elif predicted_points >= threshold:
            if risk_level == "Low":
                return "Solid start option - reliable floor with decent upside"
            elif risk_level == "High":
                return "Risky start - consider alternatives if available"
            else:
                return "Viable start - expected to meet projections"
        
        else:
            if confidence_level == "High":
                return "Consider alternatives - projected below typical starter threshold"
            else:
                return "Bench if better options available - uncertain outlook"
    
    def format_for_display(self, explanation: PredictionExplanation) -> Dict[str, Any]:
        """Format explanation for API response"""
        return {
            'summary': explanation.summary,
            'confidence_level': explanation.confidence_level,
            'key_factors': [
                {
                    'factor': f['factor'],
                    'explanation': f['explanation'],
                    'impact': f['impact'],
                    'impact_icon': '↑' if f['impact'] == 'positive' else '↓' if f['impact'] == 'negative' else '→'
                }
                for f in explanation.key_factors
            ],
            'risk_assessment': {
                'level': explanation.risk_assessment['level'],
                'factors': explanation.risk_assessment['factors'],
                'bust_probability': f"{explanation.risk_assessment['bust_probability']:.0%}"
            },
            'recommendation': explanation.recommendation
        }