#!/usr/bin/env python3
"""
Test script for the enhanced prediction API
Verifies the transparency engine and prediction endpoints work correctly
"""

import requests
import json
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.predictor import EnhancedPredictor
from backend.services.explainer import TransparencyEngine


def test_transparency_engine():
    """Test the transparency engine directly"""
    print("Testing Transparency Engine...")
    
    engine = TransparencyEngine()
    
    # Test data
    trend_analysis = {
        'performance_trend': {
            'overall_trend': 'improving',
            'last_3_games_avg': 22.5
        },
        'consistency_metrics': {
            'consistency_rating': 'Consistent',
            'coefficient_of_variation': 0.25
        },
        'hot_cold_streaks': {
            'current_form': 'Hot',
            'last_5_games_avg': 24.0,
            'games_above_avg': 4
        },
        'bust_probability': 0.15
    }
    
    # Generate explanation
    explanation = engine.explain_prediction(
        player_name="Patrick Mahomes",
        position="QB",
        predicted_points=26.5,
        confidence_score=0.85,
        trend_analysis=trend_analysis
    )
    
    # Display results
    formatted = engine.format_for_display(explanation)
    print("\nExplanation Summary:", formatted['summary'])
    print("\nKey Factors:")
    for factor in formatted['key_factors']:
        print(f"  {factor['impact_icon']} {factor['factor']}: {factor['explanation']}")
    print(f"\nRisk Level: {formatted['risk_assessment']['level']}")
    print(f"Recommendation: {formatted['recommendation']}")
    
    return True


def test_local_predictor():
    """Test the predictor service locally without database"""
    print("\n\nTesting Enhanced Predictor...")
    
    # Create mock database session
    class MockDB:
        def query(self, model):
            return self
        
        def filter(self, *args):
            return self
        
        def first(self):
            # Return mock player
            class MockPlayer:
                player_id = "6783"
                first_name = "Josh"
                last_name = "Allen"
                position = "QB"
                team = "BUF"
                status = "Active"
                age = 27
                years_exp = 6
            
            return MockPlayer()
        
        def order_by(self, *args):
            return self
        
        def limit(self, n):
            return self
        
        def all(self):
            # Return mock stats
            class MockStat:
                def __init__(self, week, points):
                    self.player_id = "6783"
                    self.season = 2024
                    self.week = week
                    self.fantasy_points_ppr = points
                    self.stats = {}
            
            return [
                MockStat(1, 28.5),
                MockStat(2, 22.3),
                MockStat(3, 31.2),
                MockStat(4, 19.8),
                MockStat(5, 26.7)
            ]
    
    # Test prediction with explanation
    predictor = EnhancedPredictor()
    
    # Mock the ML prediction
    predictor.prediction_engine.predict_player_week = lambda **kwargs: {
        'player_id': '6783',
        'player_name': 'Josh Allen',
        'position': 'QB',
        'team': 'BUF',
        'season': 2024,
        'week': 10,
        'predictions': {
            'ppr': {
                'point_estimate': 26.5,
                'lower_bound': 21.2,
                'upper_bound': 31.8,
                'uncertainty': 0.15
            },
            'standard': {
                'point_estimate': 26.5,
                'lower_bound': 21.2,
                'upper_bound': 31.8
            },
            'half_ppr': {
                'point_estimate': 26.5,
                'lower_bound': 21.2,
                'upper_bound': 31.8
            }
        },
        'confidence': {
            'score': 0.82,
            'level': 'High',
            'factors': {
                'sample_size': 50,
                'consistency': 'Consistent',
                'recent_form': 'Hot'
            }
        },
        'key_factors': []
    }
    
    # Generate prediction
    import asyncio
    
    async def run_test():
        result = await predictor.predict_with_explanation(
            player_id="6783",
            season=2024,
            week=10,
            db=MockDB()
        )
        
        print("\nPrediction Results:")
        print(f"Player: {result['player']['name']} ({result['player']['position']})")
        print(f"Predicted Points (PPR): {result['prediction']['scoring_formats']['ppr']['point_estimate']}")
        print(f"Confidence: {result['confidence']['level']} ({result['confidence']['score']:.0%})")
        print(f"\nExplanation: {result['explanation']['summary']}")
        print("\nKey Factors:")
        for factor in result['explanation']['key_factors']:
            print(f"  {factor['impact_icon']} {factor['factor']}: {factor['explanation']}")
        print(f"\nRecommendation: {result['explanation']['recommendation']}")
    
    asyncio.run(run_test())
    return True


def test_api_endpoint():
    """Test the actual API endpoint (requires server running)"""
    print("\n\nTesting API Endpoint...")
    print("Note: This requires the server to be running with: uvicorn backend.main:app --reload")
    
    # Test endpoints
    base_url = "http://localhost:8000"
    
    # 1. Health check
    try:
        response = requests.get(f"{base_url}/")
        print(f"Health Check: {response.json()}")
    except:
        print("Server not running. Start with: uvicorn backend.main:app --reload")
        return False
    
    # 2. Test pricing endpoint (no auth required)
    response = requests.get(f"{base_url}/api/payments/pricing")
    if response.status_code == 200:
        print("\nPricing Info:")
        pricing = response.json()
        print(f"  Season Pass: ${pricing['season_pass']['price']}")
        print(f"  Free Trial: {pricing['season_pass']['trial_days']} days")
    
    return True


if __name__ == "__main__":
    print("Fantasy Football AI - Enhanced Prediction API Test")
    print("=" * 50)
    
    # Test components
    success = True
    
    try:
        success &= test_transparency_engine()
    except Exception as e:
        print(f"Transparency Engine Test Failed: {e}")
        success = False
    
    try:
        success &= test_local_predictor()
    except Exception as e:
        print(f"Predictor Test Failed: {e}")
        success = False
    
    try:
        success &= test_api_endpoint()
    except Exception as e:
        print(f"API Test Failed: {e}")
        success = False
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")