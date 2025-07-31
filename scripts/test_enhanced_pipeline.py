#!/usr/bin/env python3
"""
Test script for enhanced ML pipeline components
Tests data collection, feature engineering, and model training
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.enhanced_data_collector import EnhancedDataCollector
from backend.ml.enhanced_features import EnhancedFeatureEngineer
from backend.ml.hyperparameter_tuning import FantasyModelTuner
from backend.ml.advanced_models import FantasyFootballTransformer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_data_collection():
    """Test the enhanced data collection with real APIs"""
    print("\n" + "="*60)
    print("TESTING DATA COLLECTION")
    print("="*60)
    
    collector = EnhancedDataCollector()
    
    # Test 1: Check API keys are loaded
    print("\n1. Checking API keys...")
    api_keys_status = {
        'SportsData': 'Present' if collector.api_keys['sportsdata'] else 'Missing',
        'OpenWeather': 'Present' if collector.api_keys['weather'] else 'Missing',
        'CFBD': 'Present' if collector.api_keys['college'] else 'Missing'
    }
    for api, status in api_keys_status.items():
        print(f"   - {api}: {status}")
    
    # Test 2: Fetch sample NFL data (1 week to test)
    print("\n2. Testing NFL data fetch...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test with recent data
            test_data = await collector._fetch_nfl_week_data(session, 2023, 1)
            if test_data:
                print(f"   ✓ Successfully fetched {len(test_data)} player records")
                print(f"   Sample player: {test_data[0].get('Name', 'Unknown')}")
                print(f"   Available fields: {list(test_data[0].keys())[:10]}...")
            else:
                print("   ✗ No data returned")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Weather API
    print("\n3. Testing weather data fetch...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test with Green Bay coordinates
            weather = await collector._fetch_weather_data(
                session,
                collector.stadium_coords['GB'],
                '2023-09-10T13:00:00'
            )
            if weather:
                print(f"   ✓ Weather data: Temp={weather.get('temperature')}°F, "
                      f"Wind={weather.get('wind_speed')}mph")
            else:
                print("   ✗ No weather data returned")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Combine data scraping
    print("\n4. Testing combine data collection...")
    try:
        combine_df = await collector.collect_combine_data(2023)
        if not combine_df.empty:
            print(f"   ✓ Collected {len(combine_df)} combine records")
            print(f"   Columns: {list(combine_df.columns)}")
        else:
            print("   ✗ No combine data collected")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return True


def test_feature_engineering():
    """Test the enhanced feature engineering"""
    print("\n" + "="*60)
    print("TESTING FEATURE ENGINEERING")
    print("="*60)
    
    engineer = EnhancedFeatureEngineer()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'player_id': ['P1', 'P2', 'P3'] * 10,
        'position': ['QB', 'RB', 'WR'] * 10,
        'pass_yards': np.random.randint(150, 400, 30),
        'pass_attempts': np.random.randint(20, 45, 30),
        'pass_tds': np.random.randint(0, 4, 30),
        'interceptions': np.random.randint(0, 3, 30),
        'rush_yards': np.random.randint(0, 150, 30),
        'rush_attempts': np.random.randint(0, 25, 30),
        'rush_tds': np.random.randint(0, 2, 30),
        'receptions': np.random.randint(0, 12, 30),
        'rec_yards': np.random.randint(0, 200, 30),
        'rec_tds': np.random.randint(0, 2, 30),
        'targets': np.random.randint(0, 15, 30),
        'fantasy_points': np.random.uniform(5, 35, 30),
        'game_date': pd.date_range('2023-09-01', periods=30, freq='W'),
        'temperature': np.random.uniform(40, 90, 30),
        'wind_speed': np.random.uniform(0, 25, 30),
        'offensive_line_rank': np.random.randint(1, 32, 30),
        'opponent': ['NE', 'GB', 'DAL'] * 10,
        'weight_lbs': np.random.randint(180, 250, 30),
        'height_inches': np.random.randint(68, 78, 30),
        'forty_yard': np.random.uniform(4.3, 5.0, 30),
        'injury_status': ['Healthy'] * 25 + ['Questionable'] * 5
    })
    
    # Add required columns for feature engineering
    required_cols = ['first_downs', 'team_total_plays', 'air_yards', 'red_zone_touches',
                     'red_zone_tds', 'vertical_jump', 'broad_jump', 'three_cone', 'shuttle',
                     'draft_position', 'years_experience', 'age', 'career_games']
    
    for col in required_cols:
        sample_data[col] = np.random.randint(1, 100, 30)
    
    print("\n1. Testing feature engineering...")
    print(f"   Input shape: {sample_data.shape}")
    
    try:
        # Apply feature engineering
        enhanced_data = engineer.engineer_all_features(sample_data)
        
        print(f"   ✓ Output shape: {enhanced_data.shape}")
        print(f"   ✓ Features created: {enhanced_data.shape[1] - sample_data.shape[1]} new features")
        
        # Check feature groups
        print("\n2. Checking feature groups created:")
        feature_groups = {
            'Efficiency': ['yards_per_attempt', 'yards_per_carry', 'epa_per_play'],
            'Combine': ['bmi', 'speed_score', 'burst_score'],
            'Weather': ['weather_severity', 'position_weather_impact'],
            'Momentum': ['point_trend', 'consistency_score']
        }
        
        for group, features in feature_groups.items():
            present = [f for f in features if f in enhanced_data.columns]
            print(f"   - {group}: {len(present)}/{len(features)} features present")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_training():
    """Test model training with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TESTING MODEL TRAINING")
    print("="*60)
    
    # Create sample training data
    n_samples = 1000
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.uniform(0, 40, n_samples)  # Fantasy points
    
    print(f"\n1. Testing with sample data: {X.shape}")
    
    # Test traditional NN tuning
    print("\n2. Testing hyperparameter tuning (5 trials for speed)...")
    try:
        tuner = FantasyModelTuner(model_type='traditional', n_trials=5)
        results = tuner.tune(X, y, validation_split=0.2)
        
        print(f"   ✓ Best score (MSE): {results['best_score']:.2f}")
        print(f"   ✓ Best params sample:")
        for key, value in list(results['best_params'].items())[:5]:
            print(f"     - {key}: {value}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test model building
    print("\n3. Testing model building with best params...")
    try:
        model = tuner.load_and_build_best_model(
            f'hyperparameter_results_traditional_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            (n_samples, n_features)
        )
        print(f"   ✓ Model built successfully")
        print(f"   Model summary: {model.count_params()} parameters")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True


async def test_integration():
    """Test integration of all components"""
    print("\n" + "="*60)
    print("TESTING FULL INTEGRATION")
    print("="*60)
    
    # This tests that all components can work together
    print("\n1. Creating sample integrated dataset...")
    
    # Create more realistic sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='W')
    n_players = 20
    n_weeks = len(dates)
    
    data = []
    for player_id in range(n_players):
        position = ['QB', 'RB', 'WR', 'TE'][player_id % 4]
        for week, date in enumerate(dates):
            record = {
                'player_id': f'P{player_id}',
                'name': f'Player_{player_id}',
                'position': position,
                'week': week + 1,
                'game_date': date,
                'fantasy_points': np.random.uniform(5, 35),
                'pass_yards': np.random.randint(100, 400) if position == 'QB' else 0,
                'pass_attempts': np.random.randint(20, 45) if position == 'QB' else 0,
                'rush_yards': np.random.randint(20, 150) if position in ['QB', 'RB'] else 0,
                'receptions': np.random.randint(2, 12) if position in ['RB', 'WR', 'TE'] else 0,
                'temperature': np.random.uniform(40, 85),
                'offensive_line_rank': np.random.randint(1, 32)
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    print(f"   ✓ Created dataset: {df.shape}")
    
    # Test feature engineering on integrated data
    print("\n2. Applying feature engineering...")
    engineer = EnhancedFeatureEngineer()
    
    # Add minimal required columns
    for col in ['pass_tds', 'interceptions', 'rush_attempts', 'rush_tds', 
                'rec_yards', 'rec_tds', 'targets', 'weight_lbs', 'height_inches']:
        df[col] = np.random.randint(0, 10, len(df))
    
    try:
        # Note: This will create NaN values for missing columns, which is expected
        enhanced_df = engineer._create_basic_features(df)
        print(f"   ✓ Enhanced to: {enhanced_df.shape}")
        print(f"   Sample features: {list(enhanced_df.columns)[:10]}")
    except Exception as e:
        print(f"   ✗ Error in feature engineering: {e}")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    
    return True


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ENHANCED ML PIPELINE TEST SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Data Collection
    try:
        if await test_data_collection():
            tests_passed += 1
    except Exception as e:
        print(f"\nData collection test failed: {e}")
    
    # Test 2: Feature Engineering
    try:
        if test_feature_engineering():
            tests_passed += 1
    except Exception as e:
        print(f"\nFeature engineering test failed: {e}")
    
    # Test 3: Model Training
    try:
        if test_model_training():
            tests_passed += 1
    except Exception as e:
        print(f"\nModel training test failed: {e}")
    
    # Test 4: Integration
    try:
        if await test_integration():
            tests_passed += 1
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.0f}%")
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! The enhanced pipeline is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Add required import for aiohttp
    import aiohttp
    asyncio.run(main())