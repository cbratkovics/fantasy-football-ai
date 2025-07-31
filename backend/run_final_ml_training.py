#!/usr/bin/env python3
"""
Final ML Training with All Available Data
Comprehensive training using all data sources with proper error handling
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib
from datetime import datetime
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalMLTrainer:
    """Final comprehensive ML training"""
    
    def __init__(self):
        self.nfl_client = NFLDataPyClient()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_all_data(self):
        """Load all available NFL data"""
        logger.info("Loading comprehensive NFL data...")
        
        # Load seasons 2015-2023
        seasons = list(range(2015, 2024))
        
        all_data = []
        
        for season in seasons:
            logger.info(f"Loading {season} season...")
            
            # Get weekly data
            weekly_df = self.nfl_client.import_weekly_data(season)
            
            if weekly_df is not None and len(weekly_df) > 0:
                # Add season info
                weekly_df['season'] = season
                
                # Try to get Next Gen Stats
                try:
                    # Passing NGS
                    ngs_pass = self.nfl_client.import_ngs_data('passing', season)
                    if ngs_pass is not None and 'player_gsis_id' in ngs_pass.columns:
                        ngs_pass_avg = ngs_pass.groupby('player_gsis_id').agg({
                            'avg_time_to_throw': 'mean',
                            'avg_completed_air_yards': 'mean',
                            'avg_intended_air_yards': 'mean',
                            'aggressiveness': 'mean',
                            'max_completed_air_distance': 'max',
                            'avg_air_yards_differential': 'mean',
                            'avg_air_distance': 'mean',
                            'max_air_distance': 'max'
                        }).reset_index()
                        ngs_pass_avg.columns = ['player_id'] + [f'ngs_{col}' for col in ngs_pass_avg.columns[1:]]
                        weekly_df = weekly_df.merge(ngs_pass_avg, on='player_id', how='left')
                        
                    # Rushing NGS
                    ngs_rush = self.nfl_client.import_ngs_data('rushing', season)
                    if ngs_rush is not None and 'player_gsis_id' in ngs_rush.columns:
                        ngs_rush_avg = ngs_rush.groupby('player_gsis_id').agg({
                            'efficiency': 'mean',
                            'avg_time_to_los': 'mean',
                            'rush_attempts': 'sum',
                            'rush_yards': 'sum',
                            'expected_rush_yards': 'sum',
                            'rush_yards_over_expected': 'sum',
                            'rush_yards_over_expected_per_att': 'mean',
                            'rush_pct_over_expected': 'mean'
                        }).reset_index()
                        ngs_rush_avg.columns = ['player_id'] + [f'ngs_{col}' for col in ngs_rush_avg.columns[1:]]
                        weekly_df = weekly_df.merge(ngs_rush_avg, on='player_id', how='left')
                        
                    # Receiving NGS
                    ngs_rec = self.nfl_client.import_ngs_data('receiving', season)
                    if ngs_rec is not None and 'player_gsis_id' in ngs_rec.columns:
                        ngs_rec_avg = ngs_rec.groupby('player_gsis_id').agg({
                            'avg_cushion': 'mean',
                            'avg_separation': 'mean',
                            'avg_intended_air_yards': 'mean',
                            'catch_percentage': 'mean',
                            'avg_yac': 'mean',
                            'avg_expected_yac': 'mean',
                            'avg_yac_above_expectation': 'mean'
                        }).reset_index()
                        ngs_rec_avg.columns = ['player_id'] + [f'ngs_{col}' for col in ngs_rec_avg.columns[1:]]
                        weekly_df = weekly_df.merge(ngs_rec_avg, on='player_id', how='left')
                        
                except Exception as e:
                    logger.warning(f"Could not load NGS data for {season}: {e}")
                
                all_data.append(weekly_df)
            else:
                logger.warning(f"No data found for {season}")
        
        # Combine all seasons
        data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records loaded: {len(data)}")
        
        return data
    
    def engineer_features(self, data):
        """Engineer comprehensive features"""
        logger.info("Engineering features...")
        
        # Filter for fantasy positions
        positions = ['QB', 'RB', 'WR', 'TE']
        data = data[data['position'].isin(positions)]
        
        # Remove incomplete records
        data = data[data['fantasy_points_ppr'].notna()]
        data = data[data['player_id'].notna()]
        
        # Sort for time series features
        data = data.sort_values(['player_id', 'season', 'week'])
        
        # Player groups
        player_groups = data.groupby('player_id')
        
        # 1. Basic rolling features
        rolling_cols = [
            'fantasy_points_ppr', 'completions', 'attempts', 'passing_yards', 
            'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds', 
            'carries', 'receptions', 'targets', 'receiving_yards', 'receiving_tds'
        ]
        
        for col in rolling_cols:
            if col in data.columns:
                # 3-game averages
                data[f'{col}_avg3'] = player_groups[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
                ).fillna(0)
                
                # 5-game averages
                data[f'{col}_avg5'] = player_groups[col].transform(
                    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                ).fillna(0)
                
                # Standard deviation (consistency)
                data[f'{col}_std3'] = player_groups[col].transform(
                    lambda x: x.rolling(3, min_periods=1).std().shift(1)
                ).fillna(0)
        
        # 2. Trend features
        data['fantasy_trend'] = player_groups['fantasy_points_ppr'].transform(
            lambda x: x.diff().rolling(3, min_periods=1).mean().shift(1)
        ).fillna(0)
        
        data['hot_streak'] = player_groups['fantasy_points_ppr'].transform(
            lambda x: (x > x.rolling(5, min_periods=1).mean()).rolling(3).sum().shift(1)
        ).fillna(0)
        
        # 3. Position-specific features
        # QB
        qb_mask = data['position'] == 'QB'
        if 'attempts' in data.columns:
            data.loc[qb_mask, 'pass_efficiency'] = (
                data.loc[qb_mask, 'passing_yards'] / 
                (data.loc[qb_mask, 'attempts'] + 1)
            )
            data.loc[qb_mask, 'td_rate'] = (
                data.loc[qb_mask, 'passing_tds'] / 
                (data.loc[qb_mask, 'attempts'] + 1)
            )
            data.loc[qb_mask, 'int_rate'] = (
                data.loc[qb_mask, 'interceptions'] / 
                (data.loc[qb_mask, 'attempts'] + 1)
            )
        
        # RB
        rb_mask = data['position'] == 'RB'
        if 'carries' in data.columns:
            data.loc[rb_mask, 'yards_per_carry'] = (
                data.loc[rb_mask, 'rushing_yards'] / 
                (data.loc[rb_mask, 'carries'] + 1)
            )
            data.loc[rb_mask, 'rb_receiving_share'] = (
                data.loc[rb_mask, 'receiving_yards'] / 
                (data.loc[rb_mask, 'rushing_yards'] + data.loc[rb_mask, 'receiving_yards'] + 1)
            )
        
        # WR/TE
        rec_mask = data['position'].isin(['WR', 'TE'])
        if 'targets' in data.columns:
            data.loc[rec_mask, 'catch_rate'] = (
                data.loc[rec_mask, 'receptions'] / 
                (data.loc[rec_mask, 'targets'] + 1)
            )
            data.loc[rec_mask, 'yards_per_reception'] = (
                data.loc[rec_mask, 'receiving_yards'] / 
                (data.loc[rec_mask, 'receptions'] + 1)
            )
            data.loc[rec_mask, 'yards_per_target'] = (
                data.loc[rec_mask, 'receiving_yards'] / 
                (data.loc[rec_mask, 'targets'] + 1)
            )
        
        # 4. Season progress features
        data['games_played'] = player_groups.cumcount() + 1
        data['season_week'] = data['week']
        data['is_early_season'] = (data['week'] <= 4).astype(int)
        data['is_late_season'] = (data['week'] >= 14).astype(int)
        
        # 5. Target share and usage
        if 'target_share' in data.columns:
            data['target_share_avg3'] = player_groups['target_share'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            ).fillna(0)
        
        if 'air_yards_share' in data.columns:
            data['air_yards_share_avg3'] = player_groups['air_yards_share'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            ).fillna(0)
        
        # 6. Fill NaN values
        data = data.fillna(0)
        
        logger.info(f"Features engineered. Final shape: {data.shape}")
        
        return data
    
    def prepare_ml_data(self, data):
        """Prepare final ML dataset"""
        
        # Select feature columns (exclude identifiers and target)
        exclude_cols = [
            'player_id', 'player_name', 'player_display_name', 'position',
            'position_group', 'headshot_url', 'recent_team', 'season', 
            'week', 'season_type', 'fantasy_points_ppr', 'fantasy_points',
            'game_id', 'opponent_team', 'team'
        ]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Keep only numeric columns
        numeric_feature_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_feature_cols.append(col)
        
        # Create position dummies
        position_dummies = pd.get_dummies(data['position'], prefix='position')
        
        # Combine features
        X = pd.concat([
            data[numeric_feature_cols],
            position_dummies
        ], axis=1).fillna(0)
        
        y = data['fantasy_points_ppr']
        
        # Add metadata
        X['_season'] = data['season']
        X['_week'] = data['week']
        X['_position'] = data['position']
        X['_player_id'] = data['player_id']
        
        self.feature_columns = list(X.columns)
        
        return X, y
    
    def train_models(self, X_train, y_train, model_name='all'):
        """Train ensemble models"""
        logger.info(f"Training {model_name} models...")
        
        # Remove metadata
        meta_cols = ['_season', '_week', '_position', '_player_id']
        X_clean = X_train.drop(columns=meta_cols, errors='ignore')
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        self.scalers[model_name] = scaler
        
        # Models
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        trained_models = {}
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=3)
        cv_results = {}
        
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                # Train
                model.fit(X_scaled[train_idx], y_train.iloc[train_idx])
                
                # Validate
                y_pred = model.predict(X_scaled[val_idx])
                mae = mean_absolute_error(y_train.iloc[val_idx], y_pred)
                cv_scores.append(mae)
            
            cv_results[name] = {
                'cv_mae': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
            
            # Final training
            model.fit(X_scaled, y_train)
            trained_models[name] = model
            
            logger.info(f"    {name} CV MAE: {cv_results[name]['cv_mae']:.2f} "
                       f"(+/- {cv_results[name]['cv_std']:.2f})")
        
        self.models[model_name] = trained_models
        
        return trained_models, cv_results
    
    def evaluate_models(self, X_test, y_test, model_name='all'):
        """Evaluate model performance"""
        
        # Remove metadata
        meta_cols = ['_season', '_week', '_position', '_player_id']
        X_clean = X_test.drop(columns=meta_cols, errors='ignore')
        
        # Scale
        X_scaled = self.scalers[model_name].transform(X_clean)
        
        # Get predictions
        predictions = {}
        for name, model in self.models[model_name].items():
            predictions[name] = model.predict(X_scaled)
        
        # Ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Metrics
        metrics = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred),
            'mape': np.mean(np.abs((y_test - ensemble_pred) / (y_test + 1))) * 100
        }
        
        # Individual model metrics
        for name, pred in predictions.items():
            metrics[f'{name}_mae'] = mean_absolute_error(y_test, pred)
            metrics[f'{name}_r2'] = r2_score(y_test, pred)
        
        # Position metrics
        if '_position' in X_test.columns:
            for pos in ['QB', 'RB', 'WR', 'TE']:
                mask = X_test['_position'] == pos
                if mask.sum() > 0:
                    metrics[f'mae_{pos}'] = mean_absolute_error(
                        y_test[mask], ensemble_pred[mask]
                    )
                    metrics[f'count_{pos}'] = mask.sum()
        
        return metrics, ensemble_pred
    
    def run_training(self):
        """Run complete training pipeline"""
        
        logger.info("="*70)
        logger.info("FINAL COMPREHENSIVE ML TRAINING")
        logger.info(f"Model Version: {self.model_version}")
        logger.info("="*70)
        
        # 1. Load data
        data = self.load_all_data()
        
        # 2. Engineer features
        data = self.engineer_features(data)
        
        # 3. Prepare ML data
        X, y = self.prepare_ml_data(data)
        
        logger.info(f"\nDataset shape: {X.shape}")
        logger.info(f"Features: {len(self.feature_columns) - 4} (excluding metadata)")
        
        # 4. Split by season
        train_mask = X['_season'] < 2023
        test_mask = X['_season'] == 2023
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"\nData split:")
        logger.info(f"  Training: {len(X_train)} samples (2015-2022)")
        logger.info(f"  Testing: {len(X_test)} samples (2023)")
        
        # 5. Train overall model
        logger.info("\n" + "-"*50)
        logger.info("OVERALL MODEL TRAINING")
        logger.info("-"*50)
        
        models, cv_results = self.train_models(X_train, y_train, 'all')
        
        # 6. Evaluate
        metrics, predictions = self.evaluate_models(X_test, y_test, 'all')
        
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Ensemble MAE: {metrics['mae']:.2f} points")
        logger.info(f"  Ensemble RMSE: {metrics['rmse']:.2f} points")
        logger.info(f"  Ensemble R²: {metrics['r2']:.3f}")
        logger.info(f"  Ensemble MAPE: {metrics['mape']:.1f}%")
        
        # 7. Position-specific models
        position_results = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_train = X_train[X_train['_position'] == position]
            pos_y_train = y_train[X_train['_position'] == position]
            
            if len(pos_train) > 500:  # Enough samples
                logger.info(f"\n" + "-"*50)
                logger.info(f"{position} MODEL TRAINING")
                logger.info("-"*50)
                logger.info(f"  Training samples: {len(pos_train)}")
                
                pos_models, pos_cv = self.train_models(
                    pos_train, pos_y_train, position
                )
                
                # Evaluate
                pos_test = X_test[X_test['_position'] == position]
                pos_y_test = y_test[X_test['_position'] == position]
                
                if len(pos_test) > 50:
                    pos_metrics, _ = self.evaluate_models(
                        pos_test, pos_y_test, position
                    )
                    
                    position_results[position] = pos_metrics
                    
                    logger.info(f"\n{position} Performance:")
                    logger.info(f"  MAE: {pos_metrics['mae']:.2f} points")
                    logger.info(f"  R²: {pos_metrics['r2']:.3f}")
                    logger.info(f"  Test samples: {len(pos_test)}")
        
        # 8. Save models
        self.save_models()
        
        # 9. Feature importance
        logger.info("\n" + "-"*50)
        logger.info("FEATURE IMPORTANCE (Top 20)")
        logger.info("-"*50)
        
        # Get feature importance from XGBoost
        xgb_model = self.models['all']['xgboost']
        feature_names = [col for col in self.feature_columns if not col.startswith('_')]
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 10. Final summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Model Version: {self.model_version}")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"\nOverall MAE: {metrics['mae']:.2f} points")
        logger.info(f"Overall R²: {metrics['r2']:.3f}")
        logger.info("\nPosition-specific MAE:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if f'mae_{pos}' in metrics:
                logger.info(f"  {pos}: {metrics[f'mae_{pos}']:.2f} points "
                          f"(n={metrics[f'count_{pos}']})")
        logger.info("="*70)
        
        return metrics
    
    def save_models(self):
        """Save all models and metadata"""
        
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_type, models in self.models.items():
            for name, model in models.items():
                filepath = os.path.join(
                    model_dir,
                    f'final_{model_type}_{name}_{self.model_version}.pkl'
                )
                joblib.dump(model, filepath)
        
        # Save scalers
        for model_type, scaler in self.scalers.items():
            filepath = os.path.join(
                model_dir,
                f'scaler_{model_type}_{self.model_version}.pkl'
            )
            joblib.dump(scaler, filepath)
        
        # Save metadata
        metadata = {
            'version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'features': self.feature_columns,
            'model_types': list(self.models.keys()),
            'training_seasons': list(range(2015, 2023)),
            'test_season': 2023
        }
        
        with open(f'{model_dir}/final_training_{self.model_version}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nModels saved to {model_dir}/")

def main():
    """Main execution"""
    trainer = FinalMLTrainer()
    metrics = trainer.run_training()
    return metrics

if __name__ == "__main__":
    metrics = main()
    success = metrics['mae'] < 10  # Reasonable MAE threshold
    sys.exit(0 if success else 1)