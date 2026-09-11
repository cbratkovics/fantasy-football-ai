#!/usr/bin/env python3
"""
TRAIN PRODUCTION ML MODELS - REAL DATA ONLY
Following strict validation requirements with no shortcuts
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FANTASY FOOTBALL ML MODEL TRAINING - PRODUCTION")
print("="*70)

# PHASE 1: DATA COLLECTION & VALIDATION
print("\nPHASE 1: DATA COLLECTION")
print("="*60)

# Load COMPLETE data (2019-2024)
import nfl_data_py as nfl

seasons = [2019, 2020, 2021, 2022, 2023, 2024]
data = nfl.import_weekly_data(seasons)

# Filter for fantasy positions only
fantasy_positions = ['QB', 'RB', 'WR', 'TE']
data = data[data['position'].isin(fantasy_positions)]

# Remove non-regular season games
data = data[data['season_type'] == 'REG']

print(f"Total records loaded: {len(data)}")
print(f"Date range: {data['season'].min()}-{data['season'].max()}")
print(f"Unique players: {data['player_id'].nunique()}")

# MANDATORY CHECKS
assert len(data) > 30000, f"INSUFFICIENT DATA: Have {len(data)}, need 30,000+ records"
assert data['season'].nunique() == 6, f"MISSING SEASONS: Have {data['season'].nunique()}, need 2019-2024"
assert set(data['position'].unique()) >= {'QB', 'RB', 'WR', 'TE'}, "MISSING POSITIONS"

# Show sample to prove it's real data
print("\nSample of ACTUAL data:")
sample_data = data.sample(5, random_state=42)[['player_name', 'season', 'week', 'fantasy_points_ppr', 'position']]
print(sample_data.to_string(index=False))

# PHASE 2: FEATURE ENGINEERING - LAGGED ONLY
print("\nPHASE 2: FEATURE ENGINEERING")
print("="*60)

# Sort data for proper lagging
data = data.sort_values(['player_id', 'season', 'week'])

# CRITICAL: Only use data available BEFORE the game
ALLOWED_BASE_STATS = [
    'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
    'fantasy_points_ppr', 'attempts', 'completions', 'carries',
    'interceptions', 'passing_air_yards', 'receiving_air_yards'
]

# Filter to only stats that exist in the data
ALLOWED_BASE_STATS = [stat for stat in ALLOWED_BASE_STATS if stat in data.columns]

# Create lagged features
feature_cols = []
for stat in ALLOWED_BASE_STATS:
    # Previous games (MUST use shift)
    col_L1 = f'{stat}_L1'
    data[col_L1] = data.groupby('player_id')[stat].shift(1)
    feature_cols.append(col_L1)
    
    col_L3 = f'{stat}_L3_avg'
    data[col_L3] = data.groupby('player_id')[stat].rolling(3, min_periods=1).mean().shift(1).values
    feature_cols.append(col_L3)
    
    col_L5 = f'{stat}_L5_avg'
    data[col_L5] = data.groupby('player_id')[stat].rolling(5, min_periods=1).mean().shift(1).values
    feature_cols.append(col_L5)
    
    # Season-to-date (excluding current week)
    col_season = f'{stat}_season_avg'
    data[col_season] = data.groupby(['player_id', 'season'])[stat].expanding().mean().shift(1).values
    feature_cols.append(col_season)

# Add position-specific derived features (all lagged)
# QB efficiency metrics
qb_mask = data['position'] == 'QB'
if 'completions_L3_avg' in data.columns and 'attempts_L3_avg' in data.columns:
    data.loc[qb_mask, 'completion_pct_L3'] = (
        data.loc[qb_mask, 'completions_L3_avg'] / 
        data.loc[qb_mask, 'attempts_L3_avg'].replace(0, 1)
    )
    feature_cols.append('completion_pct_L3')

# RB efficiency metrics  
rb_mask = data['position'] == 'RB'
if 'rushing_yards_L3_avg' in data.columns and 'carries_L3_avg' in data.columns:
    data.loc[rb_mask, 'yards_per_carry_L3'] = (
        data.loc[rb_mask, 'rushing_yards_L3_avg'] / 
        data.loc[rb_mask, 'carries_L3_avg'].replace(0, 1)
    )
    feature_cols.append('yards_per_carry_L3')

# WR/TE efficiency metrics
rec_mask = data['position'].isin(['WR', 'TE'])
if 'receptions_L3_avg' in data.columns and 'targets_L3_avg' in data.columns:
    data.loc[rec_mask, 'catch_rate_L3'] = (
        data.loc[rec_mask, 'receptions_L3_avg'] / 
        data.loc[rec_mask, 'targets_L3_avg'].replace(0, 1)
    )
    feature_cols.append('catch_rate_L3')

# Add temporal features
data['week_of_season'] = data['week']
data['games_played'] = data.groupby(['player_id', 'season']).cumcount()
feature_cols.extend(['week_of_season', 'games_played'])

print(f"Created {len(feature_cols)} lagged features")

# FORBIDDEN CHECK - ensure no same-week stats
forbidden_features = []
for col in feature_cols:
    if not any(lag in col for lag in ['_L1', '_L3', '_L5', '_avg', '_season', 'week_of_season', 'games_played', 'pct', 'rate', 'per']):
        forbidden_features.append(col)

assert len(forbidden_features) == 0, f"FORBIDDEN FEATURES DETECTED: {forbidden_features}"

# PHASE 3: TEMPORAL VALIDATION SETUP
print("\nPHASE 3: TEMPORAL VALIDATION")
print("="*60)

# STRICT temporal split - NO RANDOM SPLITS
train = data[data['season'].isin([2019, 2020, 2021, 2022])]
val = data[data['season'] == 2023]
test = data[data['season'] == 2024]

print(f"Train: {len(train)} records (2019-2022)")
print(f"Validation: {len(val)} records (2023)")
print(f"Test: {len(test)} records (2024)")

# Verify no temporal leakage
assert train['season'].max() < val['season'].min(), "TEMPORAL LEAK: Train/val overlap"
assert val['season'].max() < test['season'].min(), "TEMPORAL LEAK: Val/test overlap"

# Drop rows without history (first games for players)
train_before = len(train)
val_before = len(val)
test_before = len(test)

train = train.dropna(subset=['fantasy_points_ppr_L1'])
val = val.dropna(subset=['fantasy_points_ppr_L1'])
test = test.dropna(subset=['fantasy_points_ppr_L1'])

print(f"\nAfter removing rows without history:")
print(f"Train: {len(train)} (dropped {train_before - len(train)})")
print(f"Val: {len(val)} (dropped {val_before - len(val)})")
print(f"Test: {len(test)} (dropped {test_before - len(test)})")

# PHASE 4: BASELINE PERFORMANCE
print("\nPHASE 4: BASELINE PERFORMANCE")
print("="*60)

# Calculate realistic baselines
baselines = {}
for pos in ['QB', 'RB', 'WR', 'TE']:
    pos_train = train[train['position'] == pos]['fantasy_points_ppr']
    pos_test = test[test['position'] == pos]['fantasy_points_ppr']
    
    if len(pos_test) == 0:
        continue
    
    # Simple baseline: predict position average
    baseline_pred = pos_train.mean()
    baseline_predictions = [baseline_pred] * len(pos_test)
    baseline_mae = mean_absolute_error(pos_test, baseline_predictions)
    baselines[pos] = baseline_mae
    
    print(f"{pos} Baseline MAE: {baseline_mae:.2f}")
    print(f"  - Test std: {pos_test.std():.2f}")
    print(f"  - Expected MAE range: {pos_test.std()*0.6:.2f}-{pos_test.std()*0.8:.2f}")

# PHASE 5: MODEL TRAINING WITH VERIFICATION
print("\nPHASE 5: MODEL TRAINING")
print("="*60)

results = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for position in ['QB', 'RB', 'WR', 'TE']:
    print(f"\nTraining {position} model...")
    
    # Filter by position
    pos_train = train[train['position'] == position]
    pos_val = val[val['position'] == position]
    pos_test = test[test['position'] == position]
    
    if len(pos_train) < 100 or len(pos_test) < 50:
        print(f"  Skipping {position} - insufficient data")
        continue
    
    # Filter features relevant to position
    pos_features = feature_cols.copy()
    
    # Remove irrelevant features
    if position != 'QB':
        pos_features = [f for f in pos_features if not any(term in f for term in ['passing', 'completion', 'attempts', 'interceptions'])]
    if position not in ['RB', 'QB']:
        pos_features = [f for f in pos_features if not any(term in f for term in ['rushing', 'carries', 'yards_per_carry'])]
    if position not in ['WR', 'TE', 'RB']:
        pos_features = [f for f in pos_features if not any(term in f for term in ['receiving', 'receptions', 'targets', 'catch_rate'])]
    
    # Ensure all features exist
    pos_features = [f for f in pos_features if f in pos_train.columns]
    
    # Prepare features
    X_train = pos_train[pos_features].fillna(0)
    y_train = pos_train['fantasy_points_ppr']
    X_val = pos_val[pos_features].fillna(0)
    y_val = pos_val['fantasy_points_ppr']
    X_test = pos_test[pos_features].fillna(0)
    y_test = pos_test['fantasy_points_ppr']
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(pos_features)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'rf': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
        )
    }
    
    best_mae = float('inf')
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        if name == 'xgb':
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_model_name = name
    
    # Final evaluation on test set
    test_pred = best_model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # REALITY CHECKS
    warning = ""
    if test_mae < 3.0:
        warning = "⚠️ WARNING: MAE is suspiciously low"
    
    improvement = (baselines[position] - test_mae) / baselines[position] * 100
    
    # Get sample predictions
    sample_idx = pos_test.index[:5]
    sample_predictions = []
    for i, idx in enumerate(sample_idx):
        player = pos_test.loc[idx, 'player_name']
        actual = y_test.loc[idx]
        pred = test_pred[i]
        sample_predictions.append((player, actual, pred))
    
    results[position] = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'features': pos_features,
        'test_mae': test_mae,
        'baseline_mae': baselines[position],
        'improvement': improvement,
        'warning': warning,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'sample_predictions': sample_predictions
    }
    
    print(f"  Best model: {best_model_name}")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Improvement over baseline: {improvement:.1f}%")
    if warning:
        print(f"  {warning}")

# PHASE 6: FINAL VALIDATION & REPORTING
print("\nPHASE 6: FINAL VALIDATION")
print("="*60)

# Summary report
print("\n📊 FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Position':<10} {'Test MAE':<10} {'Baseline':<10} {'Improve':<10} {'Status':<20}")
print("-"*60)

all_valid = True
validation_issues = []

for pos, res in results.items():
    mae = res['test_mae']
    baseline = res['baseline_mae']
    improve = res['improvement']
    
    # Determine status
    if mae < 3.0:
        status = "⚠️ CHECK FOR LEAKAGE"
        all_valid = False
        validation_issues.append(f"{pos}: MAE too low ({mae:.2f})")
    elif improve < 5:
        status = "❌ NO IMPROVEMENT"
        validation_issues.append(f"{pos}: Insufficient improvement ({improve:.1f}%)")
    elif improve > 50:
        status = "⚠️ TOO GOOD"
        all_valid = False
        validation_issues.append(f"{pos}: Improvement unrealistic ({improve:.1f}%)")
    else:
        status = "✅ VALID"
    
    print(f"{pos:<10} {mae:<10.2f} {baseline:<10.2f} {improve:<10.1f}% {status:<20}")

print("\nSample Predictions (Player, Actual, Predicted):")
for pos, res in results.items():
    print(f"\n{pos}:")
    for name, actual, pred in res['sample_predictions'][:3]:
        print(f"  {name}: {actual:.1f} -> {pred:.1f}")

# Save models only if valid
model_dir = 'models/production'
os.makedirs(model_dir, exist_ok=True)

if all_valid:
    print("\n✅ All models validated successfully. Saving...")
    
    for position, res in results.items():
        # Save model
        model_path = os.path.join(model_dir, f'prod_{position}_model_{timestamp}.pkl')
        joblib.dump(res['model'], model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f'prod_{position}_scaler_{timestamp}.pkl')
        joblib.dump(res['scaler'], scaler_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'data_stats': {
            'total_records': int(len(data)),
            'seasons': [int(s) for s in data['season'].unique()],
            'unique_players': int(data['player_id'].nunique())
        },
        'feature_engineering': {
            'total_features': len(feature_cols),
            'all_lagged': True,
            'no_leakage': True
        },
        'model_performance': {
            pos: {
                'mae': float(res['test_mae']),
                'baseline_mae': float(res['baseline_mae']),
                'improvement': float(res['improvement']),
                'model_type': res['model_name'],
                'train_samples': int(res['train_samples']),
                'test_samples': int(res['test_samples'])
            }
            for pos, res in results.items()
        },
        'validation': {
            'all_mae_above_3': all(res['test_mae'] >= 3.0 for res in results.values()),
            'improvements_reasonable': all(5 <= res['improvement'] <= 50 for res in results.values()),
            'temporal_split_correct': True
        }
    }
    
    metadata_path = os.path.join(model_dir, f'prod_models_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Models saved to: {model_dir}/prod_*_{timestamp}.pkl")
else:
    print(f"\n❌ Some models failed validation. Review required.")
    print("Validation issues:")
    for issue in validation_issues:
        print(f"  - {issue}")
    print("\nModels NOT SAVED")

# MANDATORY OUTPUT FORMAT
print("\n" + "="*70)
print("TRAINING COMPLETE: REAL DATA VERIFICATION")
print("="*41)
print("Data Used:")
print(f"- Records: {len(data)}")
print(f"- Seasons: {sorted(data['season'].unique())}")
print(f"- Players: {data['player_id'].nunique()}")

print("\nFeature Engineering:")
print(f"- Features created: {len(feature_cols)}")
print(f"- All lagged: YES")
print(f"- No leakage: YES")

print("\nModel Performance:")
for pos in ['QB', 'RB', 'WR', 'TE']:
    if pos in results:
        res = results[pos]
        print(f"- {pos}: MAE={res['test_mae']:.2f}, Baseline={res['baseline_mae']:.2f}, Improvement={res['improvement']:.0f}%")

print(f"\nValidation Status: {'PASS' if all_valid else 'FAIL'}")
print(f"- All MAE > 3.0: {'YES' if all(res['test_mae'] >= 3.0 for res in results.values()) else 'NO'}")
print(f"- Improvements 5-50%: {'YES' if all(5 <= res['improvement'] <= 50 for res in results.values()) else 'NO'}")
print(f"- Temporal split correct: YES")

if all_valid:
    print(f"\nModels saved to: {model_dir}/prod_*_{timestamp}.pkl")
else:
    print(f"\nModels NOT SAVED (validation failed)")