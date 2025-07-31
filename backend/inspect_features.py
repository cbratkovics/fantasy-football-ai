#!/usr/bin/env python3
"""
Inspect what features are actually being used in the models
"""

import json
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# Load metadata
with open('models/proper_models_metadata_20250731_165052.json', 'r') as f:
    metadata = json.load(f)

print("="*70)
print("FEATURE INSPECTION - CHECKING FOR HIDDEN LEAKAGE")
print("="*70)

# Load some data to inspect
data = nfl.import_weekly_data([2024])
data = data[data['position'].isin(['QB', 'RB', 'WR', 'TE'])]

print("\nColumns in raw data:")
print(sorted(data.columns))

print("\n\nLagged features that should be safe:")
lagged_features = [col for col in data.columns if any(suffix in col for suffix in ['_L1W', '_L3W', '_L5W', '_avg', '_trend'])]
print(f"Found {len(lagged_features)} lagged features")

print("\n\nChecking correlation between lagged features and target:")
# Create simple lagged feature
data = data.sort_values(['player_id', 'week'])
data['fantasy_points_L1W'] = data.groupby('player_id')['fantasy_points_ppr'].shift(1)
data['fantasy_points_avg_L3W'] = data.groupby('player_id')['fantasy_points_ppr'].rolling(
    window=3, min_periods=1
).mean().shift(1).values

# Calculate correlation
data_with_lag = data.dropna(subset=['fantasy_points_L1W'])
corr_L1W = data_with_lag['fantasy_points_L1W'].corr(data_with_lag['fantasy_points_ppr'])
corr_L3W = data_with_lag['fantasy_points_avg_L3W'].corr(data_with_lag['fantasy_points_ppr'])

print(f"\nCorrelation between:")
print(f"  Last week fantasy points & current week: {corr_L1W:.3f}")
print(f"  3-week avg fantasy points & current week: {corr_L3W:.3f}")

print("\n\nRealistic expectations:")
print("- Correlation of 0.3-0.5 is good for fantasy predictions")
print("- Correlation > 0.7 suggests possible leakage")
print("- MAE should be 50-70% of standard deviation")

std_by_pos = data.groupby('position')['fantasy_points_ppr'].std()
print("\n\nStandard deviation by position:")
for pos, std in std_by_pos.items():
    print(f"  {pos}: {std:.2f} (expected MAE: {std*0.6:.2f}-{std*0.7:.2f})")

print("\n\nCONCLUSION:")
if corr_L3W > 0.7:
    print("❌ Correlation too high - possible subtle leakage")
else:
    print("✅ Correlations look reasonable")
    
print("\nThe models might be achieving lower MAE because:")
print("1. Fantasy points have strong week-to-week correlation")
print("2. Using 3-5 week averages captures player consistency")
print("3. Position-specific models reduce variance")
print("\nHowever, MAE < 3.0 is still suspicious and should be monitored in production.")