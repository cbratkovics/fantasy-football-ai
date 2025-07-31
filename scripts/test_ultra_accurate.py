#!/usr/bin/env python3
"""
Test Ultra-Accurate Models - Quick validation
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.synthetic_data_generator import SyntheticDataGenerator
from backend.models.player_profile import PlayerProfileBuilder
from backend.ml.ultra_accurate_model import UltraAccurateFantasyModel
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test():
    """Quick test of ultra-accurate model capabilities"""
    print("\n" + "="*70)
    print("TESTING ULTRA-ACCURATE MODEL SYSTEM")
    print("="*70)
    
    # Generate test data
    print("\n1. Generating test data...")
    generator = SyntheticDataGenerator()
    data = generator.generate_historical_data(years=2, players_per_position=25)
    
    # Focus on QB position for quick test
    qb_data = data[data['position'] == 'QB'].copy()
    print(f"   QB samples: {len(qb_data)}")
    
    # Add advanced features
    print("\n2. Creating advanced features...")
    
    # Add lag features
    qb_data = qb_data.sort_values(['player_id', 'game_date'])
    for lag in [1, 2, 3]:
        qb_data[f'fantasy_points_lag_{lag}'] = qb_data.groupby('player_id')['fantasy_points'].shift(lag)
    
    # Add rolling averages
    qb_data['rolling_avg_3'] = qb_data.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Add derived features
    qb_data['yards_per_attempt'] = qb_data['pass_yards'] / qb_data['pass_attempts'].replace(0, 1)
    qb_data['td_rate'] = qb_data['pass_tds'] / qb_data['pass_attempts'].replace(0, 1)
    qb_data['experience_factor'] = qb_data['years_experience'].apply(
        lambda x: 1.0 if 3 <= x <= 7 else 0.8
    )
    
    # Select features
    feature_cols = [
        'age', 'height_inches', 'weight_lbs', 'years_experience',
        'pass_attempts', 'pass_yards', 'pass_tds', 'interceptions',
        'yards_per_attempt', 'td_rate', 'experience_factor',
        'offensive_line_rank', 'opp_def_rank', 'temperature', 'wind_speed',
        'fantasy_points_lag_1', 'fantasy_points_lag_2', 'fantasy_points_lag_3',
        'rolling_avg_3'
    ]
    
    # Remove rows with NaN in features
    qb_data = qb_data.dropna(subset=feature_cols + ['fantasy_points'])
    
    X = qb_data[feature_cols]
    y = qb_data['fantasy_points'].values
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples after cleaning: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create and train model
    print("\n3. Training ultra-accurate ensemble...")
    model = UltraAccurateFantasyModel('QB')
    
    # Add some synthetic advanced features
    X_train_enhanced = model.create_advanced_features(X_train, None, qb_data)
    X_val_enhanced = model.create_advanced_features(X_val, None, qb_data)
    X_test_enhanced = model.create_advanced_features(X_test, None, qb_data)
    
    # Train (reduced epochs for speed)
    model.models['xgboost'].set_params(n_estimators=100)
    model.models['lightgbm'].set_params(n_estimators=100)
    model.models['random_forest'].set_params(n_estimators=50)
    
    results = model.train_ensemble(
        X_train_enhanced, y_train,
        X_val_enhanced, y_val
    )
    
    # Test
    print("\n4. Testing model accuracy...")
    test_pred = model.predict(X_test_enhanced)
    
    # Calculate metrics
    mae = np.mean(np.abs(test_pred - y_test))
    rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    accuracy_3 = np.mean(np.abs(test_pred - y_test) <= 3)
    accuracy_5 = np.mean(np.abs(test_pred - y_test) <= 5)
    
    # Detailed accuracy breakdown
    errors = np.abs(test_pred - y_test)
    accuracy_1 = np.mean(errors <= 1)
    accuracy_2 = np.mean(errors <= 2)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Test MAE: {mae:.2f} fantasy points")
    print(f"Test RMSE: {rmse:.2f} fantasy points")
    print(f"\nAccuracy within margins:")
    print(f"  ±1 point:  {accuracy_1:.1%}")
    print(f"  ±2 points: {accuracy_2:.1%}")
    print(f"  ±3 points: {accuracy_3:.1%} (TARGET: 92%+)")
    print(f"  ±5 points: {accuracy_5:.1%}")
    
    # Show individual model performance
    print("\nIndividual Model Performance:")
    for name, perf in results.items():
        if isinstance(perf, dict) and 'accuracy' in perf:
            print(f"  {name}: {perf['accuracy']:.1%}")
    
    # Feature importance
    if hasattr(model, 'feature_importance') and model.feature_importance:
        print("\nTop 10 Most Important Features:")
        for i, (feat, importance) in enumerate(list(model.feature_importance.items())[:10]):
            print(f"  {i+1}. {feat}: {importance:.3f}")
    
    # Analyze predictions
    print("\nPrediction Analysis:")
    print(f"  Mean actual: {np.mean(y_test):.1f}")
    print(f"  Mean predicted: {np.mean(test_pred):.1f}")
    print(f"  Std actual: {np.std(y_test):.1f}")
    print(f"  Std predicted: {np.std(test_pred):.1f}")
    
    # Success criteria
    print("\n" + "="*50)
    if accuracy_3 >= 0.92:
        print("✅ SUCCESS: Achieved 92%+ accuracy target!")
    else:
        print(f"📊 Current accuracy: {accuracy_3:.1%}")
        print("   With more data and features, 92%+ is achievable")
    
    return accuracy_3


if __name__ == "__main__":
    accuracy = quick_test()
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")