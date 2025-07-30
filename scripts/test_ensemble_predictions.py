#!/usr/bin/env python3
"""
Test ensemble prediction system
"""
import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.ensemble_predictions import EnsemblePredictionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ensemble_predictions():
    """Test ensemble prediction system"""
    logger.info("Testing Ensemble Prediction System")
    logger.info("="*60)
    
    engine = EnsemblePredictionEngine()
    
    # Test players
    test_cases = [
        {'player_id': '6783', 'name': 'Jerry Jeudy'},
        {'player_id': '4035', 'name': 'Alvin Kamara'},
        {'player_id': '5849', 'name': 'Kyler Murray'}
    ]
    
    for test in test_cases:
        logger.info(f"\nTesting prediction for {test['name']}...")
        
        try:
            result = engine.predict_player_week(
                player_id=test['player_id'],
                season=2024,
                week=10
            )
            
            if "error" not in result:
                logger.info(f"Player: {result['player_name']}")
                logger.info(f"Position: {result['position']} - Team: {result['team']}")
                
                # Ensemble prediction
                ensemble = result['predictions']['ensemble']
                logger.info(f"\nEnsemble Prediction:")
                logger.info(f"  PPR: {ensemble['ppr']['point_estimate']} ({ensemble['ppr']['lower_bound']}-{ensemble['ppr']['upper_bound']})")
                logger.info(f"  Standard: {ensemble['standard']['point_estimate']}")
                logger.info(f"  Half-PPR: {ensemble['half_ppr']['point_estimate']}")
                logger.info(f"  Method: {ensemble['method']}")
                logger.info(f"  Models used: {', '.join(ensemble['models_used'])}")
                
                # Individual model predictions
                logger.info(f"\nIndividual Models:")
                for model, pred in result['predictions']['models'].items():
                    logger.info(f"  {model}: {pred:.2f}")
                
                # Confidence
                conf = result['confidence']
                logger.info(f"\nConfidence: {conf['level']} ({conf['score']:.1%})")
                logger.info(f"  Factors: {conf['factors']}")
                
                # Trend adjustment
                trend = result['trend_adjustment']
                if trend['percentage'] != 0:
                    logger.info(f"\nTrend Adjustment: {trend['percentage']:+.1f}%")
                    for reason in trend['reasons']:
                        logger.info(f"  - {reason}")
                
                # Draft tier
                if 'draft_tier' in result:
                    tier = result['draft_tier']
                    logger.info(f"\nDraft Tier: {tier['tier']} - {tier['label']}")
                
                # Explanations
                if 'explanations' in result:
                    logger.info(f"\nExplanations:")
                    for exp in result['explanations']:
                        logger.info(f"  - {exp}")
                        
            else:
                logger.error(f"Prediction failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error predicting for {test['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("Ensemble Prediction Test Complete!")


if __name__ == "__main__":
    test_ensemble_predictions()