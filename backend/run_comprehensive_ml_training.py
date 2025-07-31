#!/usr/bin/env python3
"""
Comprehensive ML Training with All Data Sources
Includes weather data, advanced stats, and multiple seasons
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
import asyncio
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient
from data.sources.weather_client import WeatherClient
from data.sleeper_client import SleeperAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveMLTrainer:
    """Comprehensive ML Training Pipeline"""
    
    def __init__(self):
        self.nfl_client = NFLDataPyClient()
        self.weather_client = None  # Will be initialized async
        self.sleeper_client = SleeperAPIClient()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    async def initialize(self):
        """Initialize async clients"""
        self.weather_client = WeatherClient()
        await self.weather_client.__aenter__()
        
    async def close(self):
        """Close async clients"""
        if self.weather_client:
            await self.weather_client.__aexit__(None, None, None)
    
    def load_nfl_data(self, seasons):
        """Load comprehensive NFL data"""
        logger.info("Loading NFL data...")
        
        dfs = []
        for season in seasons:
            logger.info(f"Loading {season} season...")
            
            # Weekly stats
            weekly_df = self.nfl_client.import_weekly_data(season)
            if weekly_df is not None:
                # Add play-by-play derived features
                pbp_df = self.nfl_client.import_pbp_data(
                    season, 
                    columns=['game_id', 'player_id', 'week', 'epa', 'air_epa', 
                            'yac_epa', 'comp_air_epa', 'comp_yac_epa']
                )
                
                if pbp_df is not None:
                    # Aggregate EPA metrics by player/week
                    epa_agg = pbp_df.groupby(['player_id', 'week']).agg({
                        'epa': 'sum',
                        'air_epa': 'sum',
                        'yac_epa': 'sum'
                    }).reset_index()
                    
                    # Merge with weekly data
                    weekly_df = weekly_df.merge(
                        epa_agg, 
                        on=['player_id', 'week'], 
                        how='left'
                    )
                
                dfs.append(weekly_df)
        
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total NFL records: {len(data)}")
        
        return data
    
    async def add_weather_features(self, data):
        """Add weather features to the dataset"""
        logger.info("Adding weather features...")
        
        # Get unique games
        unique_games = data[['season', 'week', 'recent_team']].drop_duplicates()
        
        weather_data = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(unique_games), batch_size):
            batch = unique_games.iloc[i:i+batch_size]
            
            for _, game in batch.iterrows():
                try:
                    # Get schedule info
                    schedules = self.nfl_client.import_schedules(game['season'])
                    game_info = schedules[
                        ((schedules['home_team'] == game['recent_team']) |
                         (schedules['away_team'] == game['recent_team'])) &
                        (schedules['week'] == game['week'])
                    ]
                    
                    if not game_info.empty:
                        home_team = game_info['home_team'].iloc[0]
                        game_date = pd.to_datetime(game_info['gameday'].iloc[0])
                        
                        weather = await self.weather_client.get_game_weather(
                            home_team, game_date, 13
                        )
                        
                        weather_data.append({
                            'season': game['season'],
                            'week': game['week'],
                            'team': game['recent_team'],
                            **weather
                        })
                        
                except Exception as e:
                    logger.debug(f"Weather error for {game}: {e}")
                    
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Convert to DataFrame and merge
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            data = data.merge(
                weather_df,
                left_on=['season', 'week', 'recent_team'],
                right_on=['season', 'week', 'team'],
                how='left'
            )
            
            # Fill missing weather with defaults
            weather_cols = ['temperature', 'wind_speed', 'precipitation', 'humidity']
            for col in weather_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(65 if col == 'temperature' else 0)
        
        return data
    
    def engineer_features(self, data):
        """Engineer advanced features"""
        logger.info("Engineering features...")
        
        # Sort by player and week for rolling calculations
        data = data.sort_values(['player_id', 'season', 'week'])
        
        # Group by player
        player_groups = data.groupby('player_id')
        
        # 1. Rolling averages (last 3 and 5 games)
        rolling_cols = [
            'fantasy_points_ppr', 'completions', 'passing_yards', 'passing_tds',
            'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 
            'receiving_tds', 'targets'
        ]
        
        for col in rolling_cols:
            if col in data.columns:
                data[f'{col}_avg3'] = player_groups[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
                )
                data[f'{col}_avg5'] = player_groups[col].transform(
                    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                )
                data[f'{col}_std3'] = player_groups[col].transform(
                    lambda x: x.rolling(3, min_periods=1).std().shift(1)
                )
        
        # 2. Trend features
        data['fantasy_trend'] = player_groups['fantasy_points_ppr'].transform(
            lambda x: x.diff().rolling(3, min_periods=1).mean().shift(1)
        )
        
        # 3. Consistency score
        data['consistency_score'] = data['fantasy_points_ppr_std3'].fillna(0) / (
            data['fantasy_points_ppr_avg3'] + 1
        )
        
        # 4. Position-specific features
        # QB features
        qb_mask = data['position'] == 'QB'
        data.loc[qb_mask, 'pass_efficiency'] = (
            data.loc[qb_mask, 'passing_yards'] / 
            (data.loc[qb_mask, 'attempts'] + 1)
        )
        data.loc[qb_mask, 'td_int_ratio'] = (
            data.loc[qb_mask, 'passing_tds'] / 
            (data.loc[qb_mask, 'interceptions'] + 1)
        )
        
        # RB features
        rb_mask = data['position'] == 'RB'
        data.loc[rb_mask, 'yards_per_carry'] = (
            data.loc[rb_mask, 'rushing_yards'] / 
            (data.loc[rb_mask, 'carries'] + 1)
        )
        data.loc[rb_mask, 'rb_target_share'] = (
            data.loc[rb_mask, 'targets'] / 
            (data.loc[rb_mask, 'targets'].rolling(3).mean() + 1)
        )
        
        # WR/TE features
        rec_mask = data['position'].isin(['WR', 'TE'])
        data.loc[rec_mask, 'yards_per_reception'] = (
            data.loc[rec_mask, 'receiving_yards'] / 
            (data.loc[rec_mask, 'receptions'] + 1)
        )
        data.loc[rec_mask, 'catch_rate'] = (
            data.loc[rec_mask, 'receptions'] / 
            (data.loc[rec_mask, 'targets'] + 1)
        )
        
        # 5. EPA features if available
        if 'epa' in data.columns:
            data['epa_per_play'] = data['epa'] / (data['plays'] + 1)
            data['epa_avg3'] = player_groups['epa'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )
        
        # 6. Weather impact features
        if 'wind_speed' in data.columns:
            data['bad_weather'] = (
                (data['wind_speed'] > 15) | 
                (data['precipitation'] > 0.25) |
                (data['temperature'] < 32)
            ).astype(int)
        
        # Fill NaN values
        data = data.fillna(0)
        
        return data
    
    def prepare_ml_data(self, data):
        """Prepare data for ML training"""
        
        # Filter positions
        positions = ['QB', 'RB', 'WR', 'TE']
        data = data[data['position'].isin(positions)]
        
        # Remove rows with missing target
        data = data[data['fantasy_points_ppr'].notna()]
        
        # Select feature columns
        exclude_cols = [
            'player_id', 'player_name', 'player_display_name', 'headshot_url',
            'season', 'week', 'game_id', 'team', 'recent_team', 'opponent_team',
            'fantasy_points_ppr', 'fantasy_points', 'season_type'
        ]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = ['position', 'position_group']
        for col in categorical_cols:
            if col in feature_cols:
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                feature_cols.remove(col)
                feature_cols.extend(dummies.columns.tolist())
        
        # Select only numeric columns
        numeric_cols = []
        for col in feature_cols:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
        
        self.feature_columns = numeric_cols
        
        X = data[numeric_cols].fillna(0)
        y = data['fantasy_points_ppr']
        
        # Add metadata
        X['_season'] = data['season']
        X['_week'] = data['week']
        X['_position'] = data['position']
        
        return X, y, data
    
    def train_models(self, X_train, y_train, model_name='all'):
        """Train ensemble of models"""
        logger.info(f"Training models for {model_name}...")
        
        # Remove metadata
        meta_cols = ['_season', '_week', '_position']
        X_train_clean = X_train.drop(columns=meta_cols, errors='ignore')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_clean)
        self.scalers[model_name] = scaler
        
        # Initialize models
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        trained_models = {}
        cv_scores = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_fold_train = X_scaled[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_scaled[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, y_pred)
                scores.append(mae)
            
            cv_scores[name] = {
                'cv_mae_mean': np.mean(scores),
                'cv_mae_std': np.std(scores)
            }
            
            # Train on full data
            model.fit(X_scaled, y_train)
            trained_models[name] = model
            
            logger.info(f"{name} CV MAE: {cv_scores[name]['cv_mae_mean']:.2f} "
                       f"(+/- {cv_scores[name]['cv_mae_std']:.2f})")
        
        self.models[model_name] = trained_models
        
        return trained_models, cv_scores
    
    def evaluate_models(self, X_test, y_test, model_name='all'):
        """Evaluate model performance"""
        
        # Remove metadata
        meta_cols = ['_season', '_week', '_position']
        X_test_clean = X_test.drop(columns=meta_cols, errors='ignore')
        
        # Scale features
        X_scaled = self.scalers[model_name].transform(X_test_clean)
        
        # Get predictions from each model
        predictions = []
        for name, model in self.models[model_name].items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Ensemble prediction
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred)
        }
        
        # Metrics by position
        if '_position' in X_test.columns:
            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_mask = X_test['_position'] == pos
                if pos_mask.sum() > 0:
                    pos_mae = mean_absolute_error(
                        y_test[pos_mask], 
                        ensemble_pred[pos_mask]
                    )
                    metrics[f'mae_{pos}'] = pos_mae
                    metrics[f'count_{pos}'] = pos_mask.sum()
        
        return metrics, ensemble_pred
    
    async def run_training_pipeline(self):
        """Run complete training pipeline"""
        
        logger.info("="*70)
        logger.info("COMPREHENSIVE ML TRAINING PIPELINE")
        logger.info("="*70)
        
        # 1. Load NFL data
        seasons = [2019, 2020, 2021, 2022, 2023]
        data = self.load_nfl_data(seasons)
        
        # 2. Add weather features
        data = await self.add_weather_features(data)
        
        # 3. Engineer features
        data = self.engineer_features(data)
        
        # 4. Prepare ML data
        X, y, full_data = self.prepare_ml_data(data)
        
        logger.info(f"\nDataset shape: {X.shape}")
        logger.info(f"Features: {len(self.feature_columns)}")
        
        # 5. Split by season
        train_mask = X['_season'] < 2023
        test_mask = X['_season'] == 2023
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"\nTraining samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # 6. Train overall model
        logger.info("\n" + "-"*50)
        logger.info("TRAINING OVERALL MODEL")
        logger.info("-"*50)
        
        models, cv_scores = self.train_models(X_train, y_train, 'all')
        
        # 7. Evaluate overall model
        metrics, predictions = self.evaluate_models(X_test, y_test, 'all')
        
        logger.info(f"\nOverall Model Performance:")
        logger.info(f"  MAE: {metrics['mae']:.2f} points")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} points")
        logger.info(f"  R²: {metrics['r2']:.3f}")
        
        # 8. Train position-specific models
        for position in ['QB', 'RB', 'WR', 'TE']:
            logger.info(f"\n" + "-"*50)
            logger.info(f"TRAINING {position} MODEL")
            logger.info("-"*50)
            
            pos_train = X_train[X_train['_position'] == position]
            pos_y_train = y_train[X_train['_position'] == position]
            
            if len(pos_train) > 100:
                pos_models, pos_cv = self.train_models(
                    pos_train, pos_y_train, position
                )
                
                pos_test = X_test[X_test['_position'] == position]
                pos_y_test = y_test[X_test['_position'] == position]
                
                if len(pos_test) > 10:
                    pos_metrics, _ = self.evaluate_models(
                        pos_test, pos_y_test, position
                    )
                    
                    logger.info(f"\n{position} Model Performance:")
                    logger.info(f"  MAE: {pos_metrics['mae']:.2f} points")
                    logger.info(f"  R²: {pos_metrics['r2']:.3f}")
                    logger.info(f"  Samples: {len(pos_test)}")
        
        # 9. Save models
        logger.info("\n" + "-"*50)
        logger.info("SAVING MODELS")
        logger.info("-"*50)
        
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_type, models in self.models.items():
            for name, model in models.items():
                filepath = os.path.join(
                    model_dir, 
                    f'comprehensive_{model_type}_{name}_{timestamp}.pkl'
                )
                joblib.dump(model, filepath)
                logger.info(f"Saved {model_type}/{name} to {filepath}")
        
        # Save scalers
        for model_type, scaler in self.scalers.items():
            filepath = os.path.join(
                model_dir,
                f'scaler_{model_type}_{timestamp}.pkl'
            )
            joblib.dump(scaler, filepath)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': self.feature_columns,
            'seasons_trained': list(X_train['_season'].unique()),
            'test_season': 2023,
            'overall_metrics': metrics,
            'model_types': list(self.models.keys())
        }
        
        with open(f'{model_dir}/comprehensive_training_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 10. Final summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Total samples: {len(X)} ({len(X_train)} train, {len(X_test)} test)")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Overall MAE: {metrics['mae']:.2f} points")
        logger.info(f"Overall R²: {metrics['r2']:.3f}")
        logger.info("\nPosition-specific performance:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if f'mae_{pos}' in metrics:
                logger.info(f"  {pos}: MAE={metrics[f'mae_{pos}']:.2f} "
                          f"(n={metrics[f'count_{pos}']})")
        logger.info(f"\nModels saved to: {model_dir}/")
        logger.info("="*70)
        
        return True

async def main():
    """Main execution function"""
    trainer = ComprehensiveMLTrainer()
    
    try:
        await trainer.initialize()
        success = await trainer.run_training_pipeline()
        return success
    
    finally:
        await trainer.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)