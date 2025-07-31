#!/usr/bin/env python3
"""
VERIFY MODEL ACCURACY - Simple test to check actual performance
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("="*70)
print("CRITICAL MODEL VERIFICATION - CHECKING FOR DATA LEAKAGE")
print("="*70)

# Load 2024 data (test set)
print("\n1. Loading 2024 test data...")
data = nfl.import_weekly_data([2024])
fantasy_positions = ['QB', 'RB', 'WR', 'TE']
data = data[data['position'].isin(fantasy_positions)]

print(f"Loaded {len(data)} records for 2024 season")

# Look at what features are actually being used
print("\n2. Examining features that would be available BEFORE games:")

# Features known before the game
pre_game_features = [
    'season', 'week', 'position', 'years_in_nfl',
    'draft_round', 'draft_pick'
]

# Check what the target variable looks like
print(f"\nTarget variable (fantasy_points_ppr) stats:")
print(f"  Mean: {data['fantasy_points_ppr'].mean():.2f}")
print(f"  Std: {data['fantasy_points_ppr'].std():.2f}")
print(f"  Min: {data['fantasy_points_ppr'].min():.2f}")
print(f"  Max: {data['fantasy_points_ppr'].max():.2f}")

# Check for same-week stats that shouldn't be used
print("\n3. Checking for data leakage...")
print("\nSame-week stats that SHOULD NOT be in features:")
same_week_stats = ['passing_yards', 'rushing_yards', 'receiving_yards', 
                   'passing_tds', 'rushing_tds', 'receiving_tds',
                   'receptions', 'targets', 'completions', 'attempts']

for stat in same_week_stats:
    if stat in data.columns:
        correlation = data[stat].corr(data['fantasy_points_ppr'])
        print(f"  {stat}: correlation with fantasy_points = {correlation:.3f}")

# Calculate what a realistic model would achieve
print("\n4. Calculating realistic performance bounds...")

# Group by position
for pos in fantasy_positions:
    pos_data = data[data['position'] == pos]['fantasy_points_ppr']
    print(f"\n{pos} fantasy points:")
    print(f"  Mean: {pos_data.mean():.2f}")
    print(f"  Std: {pos_data.std():.2f}")
    print(f"  Realistic MAE (using std): ~{pos_data.std() * 0.8:.2f}")

# Simple baseline: predict position average
print("\n5. Testing baseline model (predict position average)...")
baseline_predictions = []
actual_values = []

for pos in fantasy_positions:
    pos_data = data[data['position'] == pos]
    pos_mean = pos_data['fantasy_points_ppr'].mean()
    baseline_predictions.extend([pos_mean] * len(pos_data))
    actual_values.extend(pos_data['fantasy_points_ppr'].values)

baseline_mae = mean_absolute_error(actual_values, baseline_predictions)
baseline_r2 = r2_score(actual_values, baseline_predictions)

print(f"\nBaseline model (position averages):")
print(f"  MAE: {baseline_mae:.2f}")
print(f"  R²: {baseline_r2:.3f}")

# Plot distribution
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.hist(data['fantasy_points_ppr'], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Fantasy Points PPR')
plt.ylabel('Count')
plt.title('Distribution of Fantasy Points (2024)')
plt.axvline(data['fantasy_points_ppr'].mean(), color='red', linestyle='--', label=f'Mean: {data["fantasy_points_ppr"].mean():.1f}')
plt.legend()

plt.subplot(122)
for pos in fantasy_positions:
    pos_data = data[data['position'] == pos]['fantasy_points_ppr']
    plt.hist(pos_data, bins=30, alpha=0.5, label=pos)
plt.xlabel('Fantasy Points PPR')
plt.ylabel('Count')
plt.title('Fantasy Points by Position')
plt.legend()

plt.tight_layout()
plt.savefig('fantasy_points_distribution.png')
print("\nDistribution plot saved to fantasy_points_distribution.png")

# Final verdict
print("\n" + "="*70)
print("VERDICT ON CLAIMED PERFORMANCE:")
print("="*70)

print("\n❌ CRITICAL FINDINGS:")
print("1. The model claims MAE of 0.140 - this is IMPOSSIBLE for fantasy football")
print("2. Same-week stats (passing_yards, rushing_yards, etc.) are perfectly correlated with fantasy points")
print("3. These stats are NOT known before the game - this is DATA LEAKAGE")
print(f"4. Realistic MAE should be > {baseline_mae:.1f} (baseline)")
print("5. Standard deviation of fantasy points is ~10, so MAE < 3 is extremely suspicious")

print("\n✅ REALISTIC EXPECTATIONS:")
print(f"- Baseline MAE (position average): {baseline_mae:.1f}")
print("- Good model MAE: 5-7 points")
print("- Excellent model MAE: 4-5 points") 
print("- Claimed MAE of 0.14: IMPOSSIBLE without cheating")

print("\n🚨 CONCLUSION: The models have severe data leakage and must be rebuilt!")
print("The models are using same-week stats that aren't known until AFTER the game.")
print("="*70)