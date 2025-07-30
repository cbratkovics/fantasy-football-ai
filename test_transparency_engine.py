#!/usr/bin/env python3
"""
Test the Transparency Engine without database dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.explainer import TransparencyEngine


def test_transparency_engine():
    """Test various scenarios with the transparency engine"""
    print("Fantasy Football AI - Transparency Engine Test")
    print("=" * 50)
    
    engine = TransparencyEngine()
    
    # Test Case 1: High-performing QB with positive trends
    print("\nTest Case 1: Elite QB Performance")
    print("-" * 30)
    
    trend_analysis = {
        'performance_trend': {
            'overall_trend': 'improving',
            'last_3_games_avg': 28.5
        },
        'consistency_metrics': {
            'consistency_rating': 'Very Consistent',
            'coefficient_of_variation': 0.15
        },
        'hot_cold_streaks': {
            'current_form': 'Hot',
            'last_5_games_avg': 27.0,
            'games_above_avg': 4
        },
        'bust_probability': 0.08
    }
    
    matchup_data = {
        'opponent_rank_vs_position': 30,  # Favorable matchup
        'is_home_game': True
    }
    
    explanation = engine.explain_prediction(
        player_name="Patrick Mahomes",
        position="QB",
        predicted_points=29.5,
        confidence_score=0.89,
        trend_analysis=trend_analysis,
        matchup_data=matchup_data
    )
    
    display_explanation(engine, explanation)
    
    # Test Case 2: Struggling RB with concerns
    print("\n\nTest Case 2: Underperforming RB")
    print("-" * 30)
    
    trend_analysis = {
        'performance_trend': {
            'overall_trend': 'declining',
            'last_3_games_avg': 8.2
        },
        'consistency_metrics': {
            'consistency_rating': 'Volatile',
            'coefficient_of_variation': 0.45
        },
        'hot_cold_streaks': {
            'current_form': 'Cold',
            'last_5_games_avg': 9.1,
            'games_above_avg': 1
        },
        'workload_metrics': {
            'avg_touches': 12.5
        },
        'bust_probability': 0.35
    }
    
    matchup_data = {
        'opponent_rank_vs_position': 5,  # Tough matchup
        'is_home_game': False
    }
    
    explanation = engine.explain_prediction(
        player_name="Clyde Edwards-Helaire",
        position="RB",
        predicted_points=7.8,
        confidence_score=0.52,
        trend_analysis=trend_analysis,
        matchup_data=matchup_data
    )
    
    display_explanation(engine, explanation)
    
    # Test Case 3: Consistent WR with medium confidence
    print("\n\nTest Case 3: Reliable WR2")
    print("-" * 30)
    
    trend_analysis = {
        'performance_trend': {
            'overall_trend': 'stable',
            'last_3_games_avg': 14.5
        },
        'consistency_metrics': {
            'consistency_rating': 'Consistent',
            'coefficient_of_variation': 0.28
        },
        'hot_cold_streaks': {
            'current_form': 'Neutral',
            'last_5_games_avg': 14.2,
            'games_above_avg': 3
        },
        'bust_probability': 0.18
    }
    
    matchup_data = {
        'opponent_rank_vs_position': 16,  # Average matchup
        'is_home_game': True
    }
    
    explanation = engine.explain_prediction(
        player_name="Terry McLaurin",
        position="WR",
        predicted_points=14.8,
        confidence_score=0.73,
        trend_analysis=trend_analysis,
        matchup_data=matchup_data
    )
    
    display_explanation(engine, explanation)
    
    print("\n\n✅ All transparency engine tests completed successfully!")


def display_explanation(engine, explanation):
    """Display formatted explanation"""
    formatted = engine.format_for_display(explanation)
    
    print(f"\n📊 {formatted['summary']}")
    
    print("\n🔍 Key Factors:")
    for factor in formatted['key_factors']:
        print(f"  {factor['impact_icon']} {factor['factor']}: {factor['explanation']}")
    
    print(f"\n⚠️  Risk Assessment:")
    print(f"  Level: {formatted['risk_assessment']['level']}")
    if formatted['risk_assessment']['factors']:
        print(f"  Concerns: {', '.join(formatted['risk_assessment']['factors'])}")
    print(f"  Bust Probability: {formatted['risk_assessment']['bust_probability']}")
    
    print(f"\n💡 {formatted['recommendation']}")


if __name__ == "__main__":
    test_transparency_engine()