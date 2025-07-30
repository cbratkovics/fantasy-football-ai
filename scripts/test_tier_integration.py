#!/usr/bin/env python3
"""
Test GMM tier integration with predictions
"""
import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.predictions_simple import SimplePredictionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tier_integration():
    """Test that predictions include tier information"""
    logger.info("Testing GMM Tier Integration with Predictions")
    logger.info("="*60)
    
    predictor = SimplePredictionEngine()
    
    # Test with some players
    test_players = ['6783', '4035', '5849']
    
    for player_id in test_players:
        try:
            prediction = predictor.predict_player_week(
                player_id=player_id,
                season=2024,
                week=10
            )
            
            if "error" not in prediction:
                logger.info(f"\nPlayer: {prediction['player_name']}")
                logger.info(f"Position: {prediction['position']} - Team: {prediction['team']}")
                logger.info(f"Week 10 Prediction: {prediction['predictions']['ppr']['point_estimate']} PPR points")
                
                if "draft_tier" in prediction:
                    tier = prediction['draft_tier']
                    logger.info(f"Draft Tier: {tier['tier']} - {tier['label']}")
                    logger.info(f"Tier Confidence: {tier['confidence']:.1%}")
                    
                    if tier['alternative_tiers']:
                        logger.info("Alternative Tiers:")
                        for alt_tier, prob in tier['alternative_tiers'].items():
                            logger.info(f"  - Tier {alt_tier}: {prob:.1%}")
                else:
                    logger.info("No tier information available")
            else:
                logger.error(f"Prediction failed: {prediction['error']}")
                
        except Exception as e:
            logger.error(f"Failed to predict for player {player_id}: {str(e)}")
    
    logger.info("\n" + "="*60)
    logger.info("Tier Integration Test Complete!")


if __name__ == "__main__":
    test_tier_integration()