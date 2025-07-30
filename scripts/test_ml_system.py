#!/usr/bin/env python3
"""
Test ML System - Verify all ML components are working correctly
Tests model training, predictions, trends, and rankings
"""

import sys
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.train import ModelTrainer
from backend.ml.trend_analysis import PlayerTrendAnalyzer
from backend.ml.predictions import PredictionEngine
from backend.ml.ranking_algorithm import PlayerRankingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLSystemTester:
    """Test all ML system components"""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.trend_analyzer = PlayerTrendAnalyzer()
        self.prediction_engine = PredictionEngine()
        self.ranking_system = PlayerRankingSystem()
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
        if details:
            logger.info(f"  Details: {details}")
        
        self.test_results["tests"].append({
            "name": test_name,
            "status": status,
            "details": details
        })
        
        if passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
    
    def test_data_availability(self):
        """Test that historical data is available"""
        logger.info("Testing data availability...")
        
        try:
            # Test loading data for different positions
            positions_to_test = ['QB', 'RB', 'WR']
            
            for position in positions_to_test:
                df = self.trainer.load_historical_data([position])
                
                if len(df) > 100:
                    self.log_test(
                        f"Data availability for {position}",
                        True,
                        f"Found {len(df)} player-season records"
                    )
                else:
                    self.log_test(
                        f"Data availability for {position}",
                        False,
                        f"Only found {len(df)} records"
                    )
            
        except Exception as e:
            self.log_test("Data availability", False, str(e))
    
    def test_trend_analysis(self):
        """Test trend analysis functionality"""
        logger.info("\nTesting trend analysis...")
        
        # Test with a known player
        test_players = ['6783', '4035', '5849']  # Example player IDs
        
        for player_id in test_players:
            try:
                analysis = self.trend_analyzer.analyze_player_trends(player_id)
                
                if "error" not in analysis:
                    self.log_test(
                        f"Trend analysis for player {player_id}",
                        True,
                        f"Analyzed {analysis['games_analyzed']} games, "
                        f"trend: {analysis['performance_trend']['overall_trend']}"
                    )
                    
                    # Print sample results
                    logger.info(f"  Player: {analysis['player_name']}")
                    logger.info(f"  Average PPR: {analysis['consistency_metrics']['average_points']}")
                    logger.info(f"  Consistency: {analysis['consistency_metrics']['consistency_rating']}")
                    logger.info(f"  Current Form: {analysis['hot_cold_streaks']['current_form']}")
                else:
                    self.log_test(
                        f"Trend analysis for player {player_id}",
                        False,
                        analysis['error']
                    )
                    
            except Exception as e:
                self.log_test(f"Trend analysis for player {player_id}", False, str(e))
    
    def test_model_training(self):
        """Test model training (lightweight version)"""
        logger.info("\nTesting model training...")
        
        try:
            # Train GMM for draft tiers
            logger.info("Training GMM clustering model...")
            gmm_result = self.trainer.train_gmm_model()
            
            if 'n_clusters' in gmm_result:
                self.log_test(
                    "GMM clustering training",
                    True,
                    f"Created {gmm_result['n_clusters']} clusters"
                )
            else:
                self.log_test("GMM clustering training", False, "No clusters created")
            
            # Train a position model (just QB for speed)
            logger.info("Training neural network for QB position...")
            nn_result = self.trainer.train_position_model('QB')
            
            if 'mae' in nn_result:
                self.log_test(
                    "Neural network training (QB)",
                    True,
                    f"MAE: {nn_result['mae']:.2f}, R²: {nn_result['r2']:.3f}"
                )
            else:
                self.log_test("Neural network training (QB)", False, str(nn_result))
                
        except Exception as e:
            self.log_test("Model training", False, str(e))
    
    def test_predictions(self):
        """Test prediction generation"""
        logger.info("\nTesting predictions...")
        
        # Test players
        test_cases = [
            {'player_id': '6783', 'season': 2024, 'week': 10},
            {'player_id': '4035', 'season': 2024, 'week': 10},
        ]
        
        for test in test_cases:
            try:
                prediction = self.prediction_engine.predict_player_week(
                    player_id=test['player_id'],
                    season=test['season'],
                    week=test['week']
                )
                
                if "error" not in prediction:
                    ppr_pred = prediction['predictions']['ppr']['point_estimate']
                    confidence = prediction['confidence']['level']
                    
                    self.log_test(
                        f"Prediction for {prediction['player_name']}",
                        True,
                        f"Week {test['week']}: {ppr_pred:.1f} PPR points, "
                        f"Confidence: {confidence}"
                    )
                    
                    # Show key factors
                    logger.info("  Key factors:")
                    for factor in prediction['key_factors'][:3]:
                        logger.info(f"    - {factor['factor']}: {factor['value']} ({factor['impact']})")
                else:
                    self.log_test(
                        f"Prediction for player {test['player_id']}",
                        False,
                        prediction['error']
                    )
                    
            except Exception as e:
                self.log_test(f"Prediction for player {test['player_id']}", False, str(e))
    
    def test_rankings(self):
        """Test ranking generation"""
        logger.info("\nTesting rankings...")
        
        try:
            # Generate rankings for key positions
            rankings = self.ranking_system.generate_rankings(
                season=2024,
                scoring_format='ppr',
                positions=['QB', 'RB', 'WR'],
                min_games_played=3  # Lower threshold for testing
            )
            
            if len(rankings) > 0:
                self.log_test(
                    "Ranking generation",
                    True,
                    f"Generated rankings for {len(rankings)} players"
                )
                
                # Show top 10
                logger.info("\n  Top 10 Overall Rankings:")
                logger.info("  " + "-" * 80)
                for _, player in rankings.head(10).iterrows():
                    logger.info(
                        f"  {player['overall_rank']:3d}. {player['name']:<20} "
                        f"{player['position']:>3} {player['team']:>3} "
                        f"Score: {player['composite_score']:.3f}"
                    )
                
                # Test position tiers
                qb_tiers = self.ranking_system.get_position_tiers('QB', 2024)
                logger.info(f"\n  QB Tiers: {len(qb_tiers)} tiers created")
                logger.info(f"  Tier 1 QBs: {len(qb_tiers.get(1, []))} players")
                
            else:
                self.log_test("Ranking generation", False, "No rankings generated")
                
        except Exception as e:
            self.log_test("Ranking generation", False, str(e))
    
    def test_player_comparison(self):
        """Test player comparison functionality"""
        logger.info("\nTesting player comparison...")
        
        try:
            # Compare some top players
            player_ids = ['6783', '4035', '5849']  # Example IDs
            
            comparison = self.trend_analyzer.compare_players(player_ids)
            
            if len(comparison) > 0:
                self.log_test(
                    "Player comparison",
                    True,
                    f"Compared {len(comparison)} players"
                )
                
                logger.info("\n  Player Comparison:")
                logger.info(comparison.to_string())
            else:
                self.log_test("Player comparison", False, "No comparison data")
                
        except Exception as e:
            self.log_test("Player comparison", False, str(e))
    
    def run_all_tests(self):
        """Run all ML system tests"""
        logger.info("=" * 80)
        logger.info("Fantasy Football AI - ML System Test Suite")
        logger.info("=" * 80)
        
        # Run tests in sequence
        self.test_data_availability()
        self.test_trend_analysis()
        
        # Skip heavy tests if models don't exist
        models_dir = Path("./models")
        if models_dir.exists() and any(models_dir.glob("*.pkl")):
            self.test_predictions()
            self.test_rankings()
        else:
            logger.warning("Skipping prediction/ranking tests - no trained models found")
            logger.info("Run model training first with: python -m backend.ml.train")
        
        self.test_player_comparison()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Test Summary:")
        logger.info(f"  Passed: {self.test_results['passed']}")
        logger.info(f"  Failed: {self.test_results['failed']}")
        logger.info(f"  Total:  {self.test_results['passed'] + self.test_results['failed']}")
        
        if self.test_results['failed'] > 0:
            logger.info("\nFailed tests:")
            for test in self.test_results['tests']:
                if test['status'] == 'FAILED':
                    logger.info(f"  - {test['name']}: {test['details']}")
        
        return self.test_results['failed'] == 0


if __name__ == "__main__":
    tester = MLSystemTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)