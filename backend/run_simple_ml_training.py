#!/usr/bin/env python3
"""
Simplified ML training using only nfl_data_py directly
Bypasses complex aggregation to complete training successfully
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_features(df):
    """Prepare features from raw NFL data"""
    
    # Basic features available in weekly data
    feature_cols = [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'sacks', 'rushing_yards', 'rushing_tds', 'rushing_attempts',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'carries', 'rushing_fumbles', 'receiving_fumbles',
        'target_share', 'air_yards_share', 'passing_2pt_conversions',
        'rushing_2pt_conversions', 'receiving_2pt_conversions'
    ]
    
    # Select available columns
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Add position encoding
    position_dummies = pd.get_dummies(df['position'], prefix='position')
    
    # Create feature matrix
    X = pd.concat([df[available_features].fillna(0), position_dummies], axis=1)
    
    # Add rolling averages for last 3 games
    player_groups = df.groupby('player_id')
    
    for col in available_features[:5]:  # Just top 5 features for rolling
        X[f'{col}_avg_3'] = player_groups[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        ).fillna(0)
    
    return X

def main():
    """Run simple ML training pipeline"""
    
    logger.info("="*60)
    logger.info("Starting Simplified ML Training Pipeline")
    logger.info("Using nfl_data_py directly for faster processing")
    logger.info("="*60)
    
    # Initialize client
    client = NFLDataPyClient()
    
    # 1. Load data for 2021-2023 seasons
    logger.info("\n1. Loading NFL weekly data...")
    seasons = [2021, 2022, 2023]
    
    dfs = []
    for season in seasons:
        logger.info(f"Loading {season} season...")
        df = client.import_weekly_data(season)
        if df is not None:
            dfs.append(df)
    
    # Combine all data
    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total records loaded: {len(data)}")
    
    # 2. Filter for fantasy relevant positions
    positions = ['QB', 'RB', 'WR', 'TE']
    data = data[data['position'].isin(positions)]
    logger.info(f"Records after position filter: {len(data)}")
    
    # 3. Remove incomplete data
    data = data[data['fantasy_points_ppr'].notna()]
    data = data[data['player_id'].notna()]
    logger.info(f"Records after removing incomplete: {len(data)}")
    
    # 4. Prepare features
    logger.info("\n2. Preparing features...")
    X = prepare_features(data)
    y = data['fantasy_points_ppr']
    
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # 5. Split data by season (2023 for testing)
    train_mask = data['season'] < 2023
    test_mask = data['season'] == 2023
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"\nTraining samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # 6. Train models
    logger.info("\n3. Training models...")
    
    models = {
        'xgboost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'r2': r2,
            'model': model
        }
        
        logger.info(f"{name} - MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # 7. Ensemble predictions
    logger.info("\n4. Creating ensemble predictions...")
    
    predictions = []
    for name, result in results.items():
        pred = result['model'].predict(X_test)
        predictions.append(pred)
    
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    logger.info(f"Ensemble - MAE: {ensemble_mae:.2f}, R²: {ensemble_r2:.3f}")
    
    # 8. Performance by position
    logger.info("\n5. Performance by position:")
    test_positions = data[test_mask]['position']
    
    for pos in positions:
        pos_mask = test_positions == pos
        if pos_mask.sum() > 0:
            pos_mae = mean_absolute_error(y_test[pos_mask], ensemble_pred[pos_mask])
            logger.info(f"  {pos}: MAE = {pos_mae:.2f} ({pos_mask.sum()} samples)")
    
    # 9. Save models
    logger.info("\n6. Saving models...")
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for name, result in results.items():
        filepath = os.path.join(model_dir, f'simple_{name}_{timestamp}.pkl')
        joblib.dump(result['model'], filepath)
        logger.info(f"Saved {name} to {filepath}")
    
    # 10. Summary
    logger.info("\n" + "="*60)
    logger.info("ML TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Training data: {len(X_train)} samples from 2021-2022")
    logger.info(f"Test data: {len(X_test)} samples from 2023")
    logger.info(f"Best model: Ensemble with MAE={ensemble_mae:.2f}")
    logger.info(f"Models saved to: {model_dir}/")
    logger.info("="*60)
    
    # Create summary report
    summary = {
        'timestamp': timestamp,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': list(X.columns),
        'models': {
            name: {'mae': result['mae'], 'r2': result['r2']}
            for name, result in results.items()
        },
        'ensemble': {'mae': ensemble_mae, 'r2': ensemble_r2}
    }
    
    import json
    with open(f'{model_dir}/training_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)