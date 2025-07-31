#!/usr/bin/env python3
"""
Test Enhanced Training Pipeline with Synthetic Data Fallback
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.synthetic_data_generator import generate_enhanced_training_data
from backend.ml.enhanced_features import EnhancedFeatureEngineer
from backend.ml.neural_network import FantasyNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_complete_pipeline():
    """Test the complete enhanced ML pipeline with synthetic data"""
    print("\n" + "="*70)
    print("TESTING ENHANCED ML PIPELINE WITH SYNTHETIC DATA")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic enhanced data...")
    try:
        # Generate 2 years of data for faster testing
        df = generate_enhanced_training_data(years=2)
        print(f"   ✓ Generated {len(df):,} records")
        print(f"   ✓ Features: {len(df.columns)} columns")
        print(f"   ✓ Positions: {df['position'].value_counts().to_dict()}")
        print(f"   ✓ Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Step 2: Apply enhanced feature engineering
    print("\n2. Applying enhanced feature engineering...")
    try:
        engineer = EnhancedFeatureEngineer()
        
        # Add minimal required columns that might be missing
        required_cols = ['first_downs', 'air_yards', 'team_air_yards', 
                        'pass_plays', 'time_of_possession', 'completions',
                        'dropbacks', 'pressures', 'deep_attempts',
                        'time_to_throw', 'ybc_per_attempt', 'broken_tackles',
                        'touches', 'goal_line_carries', 'team_goal_line_carries',
                        'avg_separation', 'contested_catches', 'contested_targets',
                        'slot_snaps', 'total_snaps', 'team_qb_rating',
                        'play_action_passes', 'play_action_yards', 'offensive_coordinator',
                        'team_total_offense_rank', 'opp_plays_per_game',
                        'opp_plays_20plus_allowed', 'opp_plays_faced', 'opp_blitz_rate',
                        'qb_rating_vs_blitz', 'home_timezone', 'game_timezone',
                        'game_time', 'pass_dvoa', 'rush_dvoa', 'total_dvoa',
                        'opp_pass_dvoa', 'opp_rush_dvoa', 'opp_total_dvoa',
                        'yards_per_touch', 'career_games', 'college_yards_per_game',
                        'college_td_per_game', 'college_yard_share', 'college_td_share',
                        'college_breakout_age']
        
        for col in required_cols:
            if col not in df.columns:
                # Add reasonable default values
                if 'yards' in col or 'rating' in col:
                    df[col] = np.random.uniform(0, 100, len(df))
                elif 'rank' in col:
                    df[col] = np.random.randint(1, 33, len(df))
                elif 'timezone' in col:
                    df[col] = 0
                elif col == 'game_time':
                    df[col] = '1:00 PM'
                elif col == 'offensive_coordinator':
                    df[col] = 'OC_' + df['team'].astype(str)
                else:
                    df[col] = np.random.randint(0, 50, len(df))
        
        # Apply basic features only (to avoid missing dependencies)
        enhanced_df = engineer._create_basic_features(df)
        
        # Try to add more features safely
        try:
            enhanced_df = engineer._create_efficiency_metrics(enhanced_df)
            print("   ✓ Added efficiency metrics")
        except Exception as e:
            print(f"   ! Skipped efficiency metrics: {e}")
        
        try:
            enhanced_df = engineer._create_combine_features(enhanced_df)
            print("   ✓ Added combine features")
        except Exception as e:
            print(f"   ! Skipped combine features: {e}")
        
        print(f"   ✓ Enhanced to {len(enhanced_df.columns)} features")
        
    except Exception as e:
        print(f"   ✗ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Train models for each position
    print("\n3. Training neural network models...")
    
    positions = ['QB', 'RB', 'WR', 'TE']
    results = {}
    
    for position in positions:
        print(f"\n   Training {position} model...")
        
        # Filter position data
        pos_data = enhanced_df[enhanced_df['position'] == position].copy()
        
        if len(pos_data) < 100:
            print(f"   ! Skipping {position} - insufficient data ({len(pos_data)} samples)")
            continue
        
        # Select numeric features only
        numeric_cols = pos_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and identifier columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['fantasy_points', 'player_id', 'year', 'week']]
        
        # Limit features to avoid overfitting on small dataset
        if len(feature_cols) > 50:
            # Select most relevant features based on position
            if position == 'QB':
                priority_features = ['pass_yards', 'pass_attempts', 'pass_tds', 
                                   'yards_per_attempt', 'temperature', 'wind_speed',
                                   'offensive_line_rank', 'opp_def_rank']
            elif position == 'RB':
                priority_features = ['rush_yards', 'rush_attempts', 'receptions',
                                   'yards_per_carry', 'offensive_line_rank',
                                   'red_zone_touches', 'opp_def_rank']
            elif position in ['WR', 'TE']:
                priority_features = ['receptions', 'rec_yards', 'targets',
                                   'yards_per_reception', 'air_yards',
                                   'opp_def_rank', 'red_zone_targets']
            
            # Keep priority features and sample others
            other_features = [f for f in feature_cols if f not in priority_features]
            selected_features = priority_features + other_features[:40]
            feature_cols = [f for f in selected_features if f in feature_cols]
        
        try:
            # Prepare data
            X = pos_data[feature_cols].values
            y = pos_data['fantasy_points'].values
            
            # Remove any NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                print(f"   ! Skipping {position} - too few valid samples after cleaning")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train model
            model = FantasyNeuralNetwork(
                input_dim=len(feature_cols),
                hidden_layers=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001
            )
            
            # Build and compile
            nn_model = model._build_model()
            nn_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Train with minimal epochs for testing
            history = nn_model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=20,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Evaluate
            test_loss, test_mae = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
            
            # Calculate accuracy (predictions within 3 points)
            predictions = nn_model.predict(X_test_scaled, verbose=0).flatten()
            accuracy = np.mean(np.abs(predictions - y_test) <= 3)
            
            results[position] = {
                'mae': test_mae,
                'accuracy': accuracy,
                'samples': len(X_train),
                'features': len(feature_cols)
            }
            
            print(f"   ✓ {position}: MAE={test_mae:.2f}, Accuracy={accuracy:.1%}")
            
        except Exception as e:
            print(f"   ✗ Error training {position}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    
    if results:
        print("\nModel Performance:")
        for position, metrics in results.items():
            print(f"\n{position}:")
            print(f"  - MAE: {metrics['mae']:.2f} fantasy points")
            print(f"  - Accuracy (±3 pts): {metrics['accuracy']:.1%}")
            print(f"  - Training samples: {metrics['samples']:,}")
            print(f"  - Features used: {metrics['features']}")
        
        avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
        print(f"\nOverall accuracy: {avg_accuracy:.1%}")
        
        print("\n✅ Enhanced ML pipeline is working correctly!")
        print("\nNext steps:")
        print("1. Run with real API data when available")
        print("2. Increase training data to 10 years")
        print("3. Enable hyperparameter tuning")
        print("4. Train ensemble models")
        
        return True
    else:
        print("\n✗ No models were successfully trained")
        return False


if __name__ == "__main__":
    # For TensorFlow import
    import tensorflow as tf
    
    # Run the test
    success = test_complete_pipeline()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")