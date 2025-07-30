#!/usr/bin/env python3
"""
Test Efficiency Ratio Calculator
"""
import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.efficiency_ratio import EfficiencyRatioCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_efficiency_ratio():
    """Test efficiency ratio calculations"""
    logger.info("Testing Efficiency Ratio Calculator")
    logger.info("="*60)
    
    calculator = EfficiencyRatioCalculator()
    
    # Test players from different positions
    test_cases = [
        {'player_id': '6783', 'name': 'Jerry Jeudy', 'position': 'WR'},
        {'player_id': '4035', 'name': 'Alvin Kamara', 'position': 'RB'},
        {'player_id': '5849', 'name': 'Kyler Murray', 'position': 'QB'},
        {'player_id': '6804', 'name': 'Dallas Goedert', 'position': 'TE'}
    ]
    
    for test in test_cases:
        logger.info(f"\nTesting {test['name']} ({test['position']})...")
        
        try:
            # Calculate efficiency
            result = calculator.calculate_player_efficiency(
                player_id=test['player_id'],
                season=2023,
                include_components=True
            )
            
            if "error" not in result:
                logger.info(f"Player: {result['player_name']}")
                logger.info(f"Position: {result['position']}")
                logger.info(f"Games Analyzed: {result['games_analyzed']}")
                logger.info(f"\nEfficiency Metrics:")
                logger.info(f"  Overall Ratio: {result['efficiency_ratio']}")
                logger.info(f"  Grade: {result['efficiency_grade']}")
                logger.info(f"  Percentile: {result['percentile_rank']}%")
                logger.info(f"  vs Average: {result['comparison_to_average']}")
                
                if 'components' in result:
                    logger.info(f"\nDetailed Components:")
                    for comp, value in result['components'].items():
                        if value is not None:
                            logger.info(f"  {comp}: {value}")
                    
                    logger.info(f"\nDetailed Metrics:")
                    for metric, value in result['components']['detailed_metrics'].items():
                        logger.info(f"  {metric}: {value}")
                
                if 'insights' in result:
                    logger.info(f"\nInsights:")
                    for insight in result['insights']:
                        logger.info(f"  - {insight}")
                
                # Test weekly trend
                logger.info(f"\nCalculating weekly efficiency trend...")
                trend = calculator.calculate_weekly_efficiency_trend(
                    player_id=test['player_id'],
                    season=2023,
                    last_n_weeks=5
                )
                
                if "error" not in trend:
                    logger.info(f"  Average Efficiency: {trend['average_efficiency']}")
                    logger.info(f"  Trend: {trend['trend']}")
                    logger.info(f"  Recent Direction: {trend['recent_direction']}")
                    logger.info(f"  Weekly Breakdown:")
                    for week_data in trend['weekly_efficiencies']:
                        logger.info(f"    Week {week_data['week']}: {week_data['efficiency']} ({week_data['grade']})")
                        
            else:
                logger.error(f"Error calculating efficiency: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error testing {test['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test position rankings
    logger.info("\n" + "="*60)
    logger.info("Testing Position Rankings")
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        logger.info(f"\nTop 10 {position}s by Efficiency:")
        try:
            rankings = calculator.get_position_efficiency_rankings(
                position=position,
                season=2023,
                min_games=5
            )
            
            for player in rankings[:10]:
                logger.info(f"  {player['rank']}. {player['player_name']} ({player['team']}): "
                          f"{player['efficiency_ratio']} ({player['grade']}) - {player['games']} games")
                
        except Exception as e:
            logger.error(f"Error getting {position} rankings: {str(e)}")
    
    logger.info("\n" + "="*60)
    logger.info("Efficiency Ratio Test Complete!")


if __name__ == "__main__":
    test_efficiency_ratio()