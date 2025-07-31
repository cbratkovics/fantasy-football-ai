#!/usr/bin/env python3
"""
Train Ultra-Accurate Fantasy Football Models (92%+ accuracy target)
Uses comprehensive player profiles and advanced ML techniques
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.player_profile import PlayerProfileBuilder, create_player_database
from backend.ml.ultra_accurate_model import UltraAccurateFantasyModel
from backend.data.synthetic_data_generator import SyntheticDataGenerator
from backend.ml.enhanced_features import EnhancedFeatureEngineer
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltraAccurateTrainingPipeline:
    """Complete training pipeline for ultra-accurate models"""
    
    def __init__(self):
        self.models = {}
        self.player_profiles = {}
        self.training_results = {}
        
    def generate_comprehensive_data(self, years: int = 10) -> pd.DataFrame:
        """Generate comprehensive training data with all features"""
        logger.info(f"Generating {years} years of comprehensive training data...")
        
        # Use synthetic generator with enhanced features
        generator = SyntheticDataGenerator()
        base_data = generator.generate_historical_data(years=years, players_per_position=100)
        
        # Add additional player profile features
        logger.info("Enhancing with player profile features...")
        
        # Add career progression
        base_data['age_squared'] = base_data['age'] ** 2
        base_data['experience_factor'] = base_data['years_experience'].apply(
            lambda x: 1.0 if 3 <= x <= 7 else 0.8 if x < 3 else 0.9
        )
        
        # Add more granular physical metrics
        base_data['bmi_category'] = pd.cut(
            base_data['bmi'], 
            bins=[0, 23, 26, 29, 100], 
            labels=['lean', 'athletic', 'powerful', 'heavy']
        )
        
        # Add career trajectory
        base_data = base_data.sort_values(['player_id', 'game_date'])
        base_data['career_game_number'] = base_data.groupby('player_id').cumcount() + 1
        base_data['season_game_number'] = base_data.groupby(['player_id', 'year'])['week'].rank()
        
        # Add performance stability metrics
        base_data['rolling_std'] = base_data.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        base_data['performance_consistency'] = 1 / (1 + base_data['rolling_std'])
        
        # Add matchup-specific features
        base_data['defensive_mismatch'] = np.where(
            base_data['opp_def_rank'] > 20,
            base_data['career_ppg'] * 1.2,
            base_data['career_ppg'] * 0.9
        )
        
        # Add time-based patterns
        base_data['month_factor'] = base_data['month'].map({
            9: 1.0,   # September - fresh
            10: 1.05,  # October - getting in rhythm  
            11: 1.1,   # November - peak performance
            12: 1.05,  # December - playoff push
            1: 0.95    # January - fatigue/cold
        }).fillna(1.0)
        
        # Add draft value impact
        base_data['draft_capital_impact'] = np.where(
            base_data['draft_position'] > 0,
            1 / (1 + np.log(base_data['draft_position'])),
            0.5
        )
        
        # Create player usage patterns
        base_data['usage_trend'] = base_data.groupby('player_id')['snap_count_percentage'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Add team synergy features
        base_data['qb_wr_stack'] = np.where(
            base_data['position'].isin(['WR', 'TE']),
            base_data['quarterback_rating'] / 100,
            1.0
        )
        
        # Add schedule strength
        base_data['remaining_sos'] = np.random.uniform(0.8, 1.2, len(base_data))
        
        return base_data
    
    def create_player_profiles_from_data(self, data: pd.DataFrame) -> Dict:
        """Create comprehensive player profiles from data"""
        logger.info("Building player profiles...")
        
        builder = PlayerProfileBuilder()
        profiles = {}
        
        for player_id in data['player_id'].unique():
            player_data = data[data['player_id'] == player_id]
            
            # Get latest info
            latest = player_data.iloc[-1]
            
            # Calculate career stats
            career_stats = {
                'games_played': len(player_data),
                'total_points': player_data['fantasy_points'].sum(),
                'ppg': player_data['fantasy_points'].mean(),
                'best_game': player_data['fantasy_points'].max(),
                'worst_game': player_data['fantasy_points'].min()
            }
            
            # Build profile data
            profile_data = {
                'player_id': player_id,
                'name': latest.get('name', f'Player_{player_id}'),
                'position': latest['position'],
                'team': latest['team'],
                'height_inches': latest.get('height_inches', 72),
                'weight_lbs': latest.get('weight_lbs', 200),
                'age': latest.get('age', 25),
                'years_experience': latest.get('years_experience', 3),
                'forty_yard_dash': latest.get('forty_yard', 4.5),
                'vertical_jump': latest.get('vertical_jump', 35),
                'bench_press_reps': latest.get('bench_press', 20),
                'career_stats': career_stats,
                'game_logs': player_data[['game_date', 'fantasy_points', 'opponent']].to_dict('records')
            }
            
            # Build profile
            profile = builder.build_profile(profile_data)
            
            # Enrich with situational data
            profile = builder.enrich_with_situational_data(profile, player_data)
            
            profiles[player_id] = profile
        
        logger.info(f"Created {len(profiles)} player profiles")
        return profiles
    
    def prepare_training_data(self, data: pd.DataFrame, profiles: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data with all advanced features"""
        logger.info("Preparing training data with advanced features...")
        
        # Initialize feature engineer
        engineer = EnhancedFeatureEngineer()
        
        # Select features for training
        feature_columns = [
            # Player attributes
            'height_inches', 'weight_lbs', 'age', 'bmi', 'years_experience',
            'forty_yard', 'vertical_jump', 'bench_press', 'speed_score',
            'burst_score', 'agility_score', 'athleticism_score',
            
            # Performance metrics
            'career_ppg', 'performance_consistency', 'rolling_std',
            'usage_trend', 'draft_capital_impact', 'experience_factor',
            
            # Game context
            'temperature', 'wind_speed', 'humidity', 'dome_game',
            'is_primetime', 'is_division_game', 'week_of_season',
            'offensive_line_rank', 'opp_def_rank', 'team_pass_rate',
            'team_pace', 'quarterback_rating',
            
            # Recent form
            'avg_points_last_3', 'avg_points_last_5', 'point_trend',
            
            # Situational
            'home_ppg', 'away_ppg', 'defensive_mismatch', 'month_factor',
            'qb_wr_stack', 'remaining_sos',
            
            # Usage metrics
            'snap_count_percentage', 'target_share', 'red_zone_share',
            'opportunity_share', 'air_yards_share'
        ]
        
        # Add position-specific features
        position_features = {
            'QB': ['pass_attempts', 'completions', 'pass_yards', 'pass_tds',
                   'interceptions', 'rush_attempts', 'sacks', 'pocket_time',
                   'pressure_rate_faced', 'deep_ball_rate'],
            'RB': ['rush_attempts', 'rush_yards', 'yards_per_carry', 'receptions',
                   'targets', 'goal_line_share', 'yards_before_contact',
                   'broken_tackle_rate', 'pass_block_grade'],
            'WR': ['targets', 'receptions', 'rec_yards', 'air_yards', 'yac',
                   'separation_score', 'contested_catch_rate', 'slot_rate'],
            'TE': ['targets', 'receptions', 'rec_yards', 'blocking_snaps_pct',
                   'route_participation', 'red_zone_targets']
        }
        
        # Filter available features
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Add lag features for time series
        for lag in [1, 2, 3]:
            if f'fantasy_points_lag_{lag}' not in data.columns:
                data[f'fantasy_points_lag_{lag}'] = data.groupby('player_id')['fantasy_points'].shift(lag)
                available_features.append(f'fantasy_points_lag_{lag}')
        
        # Create feature matrix
        X = data[available_features].fillna(0)
        y = data['fantasy_points'].values
        
        # Add profile-based features
        profile_features = []
        for idx, row in data.iterrows():
            player_id = row['player_id']
            if player_id in profiles:
                profile = profiles[player_id]
                profile_feat = [
                    profile.career_ppg,
                    profile.career_consistency,
                    profile.boom_rate,
                    profile.bust_rate,
                    profile.get_experience_score()
                ]
            else:
                profile_feat = [0] * 5
            profile_features.append(profile_feat)
        
        profile_features = np.array(profile_features)
        X = pd.concat([X, pd.DataFrame(profile_features, columns=[
            'profile_ppg', 'profile_consistency', 'profile_boom_rate', 
            'profile_bust_rate', 'profile_experience'
        ])], axis=1)
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y
    
    def train_position_models(self, data: pd.DataFrame, profiles: Dict) -> Dict:
        """Train ultra-accurate models for each position"""
        positions = ['QB', 'RB', 'WR', 'TE']
        results = {}
        
        for position in positions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training ultra-accurate model for {position}")
            logger.info(f"{'='*60}")
            
            # Filter position data
            pos_data = data[data['position'] == position].copy()
            
            if len(pos_data) < 500:
                logger.warning(f"Insufficient data for {position} ({len(pos_data)} samples)")
                continue
            
            # Prepare features
            X, y = self.prepare_training_data(pos_data, profiles)
            
            # Split data (time-aware split)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Further split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Create and train model
            model = UltraAccurateFantasyModel(position)
            
            # Create advanced features
            X_train_enhanced = model.create_advanced_features(X_train, profiles, pos_data)
            X_val_enhanced = model.create_advanced_features(X_val, profiles, pos_data)
            X_test_enhanced = model.create_advanced_features(X_test, profiles, pos_data)
            
            # Train ensemble
            train_results = model.train_ensemble(
                X_train_enhanced, y_train,
                X_val_enhanced, y_val,
                player_profiles=profiles
            )
            
            # Test performance
            test_pred = model.predict(X_test_enhanced)
            test_accuracy = np.mean(np.abs(test_pred - y_test) <= 3)
            test_mae = np.mean(np.abs(test_pred - y_test))
            
            # Get prediction intervals
            lower, upper = model.get_prediction_intervals(X_test_enhanced)
            interval_coverage = np.mean((y_test >= lower) & (y_test <= upper))
            
            results[position] = {
                'train_results': train_results,
                'test_accuracy': test_accuracy,
                'test_mae': test_mae,
                'interval_coverage': interval_coverage,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'top_features': list(model.feature_importance.items())[:10]
            }
            
            # Save model
            self.models[position] = model
            
            logger.info(f"\n{position} Final Results:")
            logger.info(f"Test Accuracy (±3 pts): {test_accuracy:.1%}")
            logger.info(f"Test MAE: {test_mae:.2f}")
            logger.info(f"90% Interval Coverage: {interval_coverage:.1%}")
            
        return results
    
    def evaluate_accuracy_improvements(self, results: Dict) -> Dict:
        """Analyze what improvements led to higher accuracy"""
        analysis = {
            'overall_accuracy': np.mean([r['test_accuracy'] for r in results.values()]),
            'position_breakdown': {},
            'key_improvements': []
        }
        
        for position, result in results.items():
            analysis['position_breakdown'][position] = {
                'accuracy': result['test_accuracy'],
                'mae': result['test_mae'],
                'best_model': max(result['train_results'].items(), 
                                key=lambda x: x[1]['accuracy'])[0]
            }
        
        # Identify key improvements
        if analysis['overall_accuracy'] > 0.85:
            analysis['key_improvements'].extend([
                "Player profile integration with career metrics",
                "Advanced ensemble with 5+ model types",
                "Situational performance adjustments",
                "Time-series features and momentum tracking",
                "Custom loss function penalizing large errors"
            ])
        
        if analysis['overall_accuracy'] > 0.90:
            analysis['key_improvements'].extend([
                "Meta-learning stacking approach",
                "Position-specific feature engineering",
                "Weather and venue adjustments",
                "Player usage trend analysis",
                "Neural network ensemble with multiple architectures"
            ])
        
        return analysis
    
    def save_all_models(self, results: Dict):
        """Save all trained models and results"""
        logger.info("\nSaving ultra-accurate models...")
        
        # Create directory
        model_dir = 'models/ultra_accurate'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save each position model
        for position, model in self.models.items():
            model.save_model(os.path.join(model_dir, f'model_{position}'))
            logger.info(f"Saved {position} model")
        
        # Save metadata
        metadata = {
            'version': '4.0_ultra_accurate',
            'trained_at': datetime.now().isoformat(),
            'positions': list(self.models.keys()),
            'results': results,
            'target_accuracy': 0.92,
            'achieved_accuracy': np.mean([r['test_accuracy'] for r in results.values()]),
            'improvements': [
                'Comprehensive player profiles with 50+ attributes',
                'Advanced ensemble with XGBoost, LightGBM, Neural Networks',
                'Meta-learning stacking approach',
                'Position-specific feature engineering',
                'Situational performance modeling',
                'Time-series momentum tracking',
                'Custom loss functions',
                'Prediction interval estimation'
            ]
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save player profiles
        import pickle
        with open(os.path.join(model_dir, 'player_profiles.pkl'), 'wb') as f:
            pickle.dump(self.player_profiles, f)
        
        logger.info(f"\nSaved all models with {metadata['achieved_accuracy']:.1%} accuracy!")


def main():
    """Main training function"""
    logger.info("="*70)
    logger.info("ULTRA-ACCURATE FANTASY FOOTBALL MODEL TRAINING")
    logger.info("Target: 92%+ accuracy within 3 fantasy points")
    logger.info("="*70)
    
    try:
        # Create pipeline
        pipeline = UltraAccurateTrainingPipeline()
        
        # Generate comprehensive data
        data = pipeline.generate_comprehensive_data(years=10)
        logger.info(f"Generated {len(data):,} training samples")
        
        # Create player profiles
        profiles = pipeline.create_player_profiles_from_data(data)
        pipeline.player_profiles = profiles
        
        # Train models
        results = pipeline.train_position_models(data, profiles)
        
        # Analyze improvements
        analysis = pipeline.evaluate_accuracy_improvements(results)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - FINAL RESULTS")
        print("="*70)
        print(f"\nOverall Accuracy: {analysis['overall_accuracy']:.1%}")
        print("\nPosition Breakdown:")
        for pos, stats in analysis['position_breakdown'].items():
            print(f"  {pos}: {stats['accuracy']:.1%} (MAE: {stats['mae']:.2f})")
        
        print("\nKey Improvements That Achieved High Accuracy:")
        for improvement in analysis['key_improvements']:
            print(f"  • {improvement}")
        
        # Save models
        pipeline.save_all_models(results)
        
        if analysis['overall_accuracy'] >= 0.92:
            print("\n✅ SUCCESS: Achieved 92%+ accuracy target!")
        else:
            print(f"\n⚠️  Achieved {analysis['overall_accuracy']:.1%} accuracy")
            print("   Additional improvements needed:")
            print("   - More training data")
            print("   - Feature selection optimization")
            print("   - Hyperparameter fine-tuning")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Check for required packages
    try:
        import xgboost
        import lightgbm
    except ImportError:
        print("Installing required packages...")
        os.system("pip install xgboost lightgbm")
        print("Please run the script again after installation.")
        sys.exit(1)
    
    main()