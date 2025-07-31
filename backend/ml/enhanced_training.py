"""
Enhanced ML Training Pipeline with Multiple Free Data Sources
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sources.data_aggregator import DataAggregator

logger = logging.getLogger(__name__)


class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline using multiple free data sources
    
    Features:
    - Multiple model ensemble (XGBoost, Random Forest, Gradient Boosting)
    - Advanced feature engineering from all data sources
    - Time series validation
    - Incremental learning support
    - Model versioning
    """
    
    def __init__(self):
        """Initialize ML pipeline"""
        self.aggregator = DataAggregator()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"Enhanced ML Pipeline initialized (version: {self.model_version})")
        
    async def prepare_training_data(self, seasons: List[int] = None,
                                  min_samples_per_player: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare comprehensive training data from all sources
        
        Args:
            seasons: List of seasons (default: 2015-2023)
            min_samples_per_player: Minimum games for a player to be included
            
        Returns:
            Tuple of (features_df, target_df)
        """
        if seasons is None:
            # Use all available seasons with good data coverage
            seasons = list(range(2015, 2024))
            
        logger.info(f"Preparing training data for seasons: {seasons}")
        
        # Build comprehensive dataset
        async with self.aggregator as agg:
            df = await agg.build_training_dataset(seasons)
            
        # Filter players with enough samples
        player_counts = df['player_id'].value_counts()
        valid_players = player_counts[player_counts >= min_samples_per_player].index
        df = df[df['player_id'].isin(valid_players)]
        
        logger.info(f"Dataset size after filtering: {len(df)} samples")
        
        # Separate features and target
        target_col = 'stats_fantasy_points_ppr'
        
        # Select feature columns (exclude identifiers and target)
        exclude_cols = ['player_id', 'name', 'season', 'week', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = ['position', 'team']
        for col in categorical_cols:
            if col in feature_cols:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.remove(col)
                feature_cols.extend(dummies.columns.tolist())
                
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Create feature and target DataFrames
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Add metadata for tracking
        X['_season'] = df['season']
        X['_week'] = df['week']
        X['_player_id'] = df['player_id']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y
        
    def create_time_series_splits(self, X: pd.DataFrame, y: pd.Series, 
                                n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series splits for validation
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_splits: Number of splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        # Sort by season and week
        sorted_idx = X.sort_values(['_season', '_week']).index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]
        
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for train_idx, test_idx in tscv.split(X_sorted):
            splits.append((sorted_idx[train_idx], sorted_idx[test_idx]))
            
        return splits
        
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series,
                           position: Optional[str] = None) -> Dict[str, Any]:
        """
        Train ensemble of models
        
        Args:
            X: Features DataFrame
            y: Target Series
            position: Optional position-specific model
            
        Returns:
            Dict with trained models and metrics
        """
        model_key = position or 'all'
        logger.info(f"Training ensemble models for: {model_key}")
        
        # Remove metadata columns for training
        meta_cols = ['_season', '_week', '_player_id']
        X_train = X.drop(columns=meta_cols, errors='ignore')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[model_key] = scaler
        
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
                max_depth=10,
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
        
        # Train models with cross-validation
        cv_results = {}
        trained_models = {}
        
        # Get time series splits
        splits = self.create_time_series_splits(X, y, n_splits=5)
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_fold_train = X_scaled[train_idx]
                y_fold_train = y.iloc[train_idx]
                X_fold_val = X_scaled[val_idx]
                y_fold_val = y.iloc[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Validate
                y_pred = model.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, y_pred)
                cv_scores.append(mae)
                
            cv_results[model_name] = {
                'cv_mae_mean': np.mean(cv_scores),
                'cv_mae_std': np.std(cv_scores)
            }
            
            # Train on full data
            model.fit(X_scaled, y)
            trained_models[model_name] = model
            
            logger.info(f"{model_name} CV MAE: {cv_results[model_name]['cv_mae_mean']:.2f} "
                       f"(+/- {cv_results[model_name]['cv_mae_std']:.2f})")
            
        # Store models
        self.models[model_key] = trained_models
        
        # Calculate feature importance (average across models)
        feature_importance = self._calculate_feature_importance(
            trained_models, X_train.columns
        )
        
        return {
            'models': trained_models,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'scaler': scaler
        }
        
    def _calculate_feature_importance(self, models: Dict[str, Any], 
                                    feature_names: List[str]) -> pd.DataFrame:
        """Calculate average feature importance across models"""
        importance_dict = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
                
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            importance_df['mean'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('mean', ascending=False)
            
            return importance_df
            
        return pd.DataFrame()
        
    def predict_ensemble(self, X: pd.DataFrame, position: Optional[str] = None) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features DataFrame
            position: Optional position-specific model
            
        Returns:
            Array of predictions
        """
        model_key = position or 'all'
        
        if model_key not in self.models:
            raise ValueError(f"No model trained for: {model_key}")
            
        # Remove metadata columns
        meta_cols = ['_season', '_week', '_player_id']
        X_pred = X.drop(columns=meta_cols, errors='ignore')
        
        # Scale features
        X_scaled = self.scalers[model_key].transform(X_pred)
        
        # Get predictions from each model
        predictions = []
        
        for model_name, model in self.models[model_key].items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            
        # Average predictions (simple ensemble)
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                      position: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            position: Optional position-specific model
            
        Returns:
            Dict of metrics
        """
        y_pred = self.predict_ensemble(X_test, position)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test.replace(0, 1))) * 100
        }
        
        # Calculate metrics by range
        ranges = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
        
        for low, high in ranges:
            mask = (y_test >= low) & (y_test < high)
            if mask.sum() > 0:
                range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                metrics[f'mae_{low}_{high}'] = range_mae
                
        return metrics
        
    def save_models(self, path: str = 'models/'):
        """Save all trained models and metadata"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for position, models in self.models.items():
            for model_name, model in models.items():
                filename = f"{path}/model_{self.model_version}_{position}_{model_name}.pkl"
                joblib.dump(model, filename)
                
        # Save scalers
        for position, scaler in self.scalers.items():
            filename = f"{path}/scaler_{self.model_version}_{position}.pkl"
            joblib.dump(scaler, filename)
            
        # Save metadata
        metadata = {
            'version': self.model_version,
            'feature_columns': self.feature_columns,
            'positions': list(self.models.keys()),
            'model_types': list(next(iter(self.models.values())).keys())
        }
        
        import json
        with open(f"{path}/metadata_{self.model_version}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Models saved to {path} (version: {self.model_version})")
        
    async def run_training_pipeline(self, seasons: List[int] = None,
                                  test_season: int = 2024):
        """
        Run complete training pipeline
        
        Args:
            seasons: Training seasons (default: 2015-2023)
            test_season: Season to use for testing
        """
        logger.info("Starting Enhanced ML Training Pipeline")
        logger.info("="*60)
        
        # 1. Prepare data
        X, y = await self.prepare_training_data(seasons)
        
        # 2. Split by season for testing
        train_mask = X['_season'] < test_season
        test_mask = X['_season'] == test_season
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # 3. Train overall model
        results = self.train_ensemble_model(X_train, y_train)
        
        # 4. Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        logger.info("\nOverall Model Performance:")
        logger.info(f"  MAE: {metrics['mae']:.2f} points")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} points")
        logger.info(f"  R²: {metrics['r2']:.3f}")
        logger.info(f"  MAPE: {metrics['mape']:.1f}%")
        
        # 5. Train position-specific models
        for position in ['QB', 'RB', 'WR', 'TE']:
            # Filter by position
            pos_mask_train = X_train[f'position_{position}'] == 1
            pos_mask_test = X_test[f'position_{position}'] == 1
            
            if pos_mask_train.sum() > 100:  # Enough samples
                logger.info(f"\nTraining {position} model...")
                
                pos_results = self.train_ensemble_model(
                    X_train[pos_mask_train],
                    y_train[pos_mask_train],
                    position=position
                )
                
                # Evaluate position model
                if pos_mask_test.sum() > 10:
                    pos_metrics = self.evaluate_model(
                        X_test[pos_mask_test],
                        y_test[pos_mask_test],
                        position=position
                    )
                    
                    logger.info(f"{position} Model Performance:")
                    logger.info(f"  MAE: {pos_metrics['mae']:.2f} points")
                    logger.info(f"  R²: {pos_metrics['r2']:.3f}")
                    
        # 6. Feature importance
        if 'feature_importance' in results:
            logger.info("\nTop 20 Features:")
            top_features = results['feature_importance'].head(20)
            for idx, (feat, row) in enumerate(top_features.iterrows(), 1):
                logger.info(f"  {idx}. {feat}: {row['mean']:.4f}")
                
        # 7. Save models
        self.save_models()
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline completed successfully!")
        
        return metrics


# Example usage
async def test_enhanced_pipeline():
    """Test the enhanced ML pipeline"""
    pipeline = EnhancedMLPipeline()
    
    # Run training on subset of data for testing
    metrics = await pipeline.run_training_pipeline(
        seasons=[2022, 2023],  # Limited seasons for testing
        test_season=2023
    )
    
    print(f"\nFinal test metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_pipeline())