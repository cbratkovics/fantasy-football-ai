"""
ML Model Training Pipeline using Historical Data
Trains both GMM clustering for draft tiers and Neural Networks for predictions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from backend.models.database import Player, PlayerStats
from backend.ml.features import FeatureEngineer
from backend.ml.gmm_clustering import GMMDraftOptimizer
from backend.ml.neural_network import FantasyNeuralNetwork

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


class ModelTrainer:
    """Orchestrates the training of all ML models using historical data"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        # Initialize GMM with proper settings for 7 features
        self.gmm_clusterer = GMMDraftOptimizer(n_components=16, n_pca_components=7)
        # Initialize with default dimensions, will be updated per position
        self.nn_predictor = None
        
        # Position groups for separate models
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'FLEX': ['RB', 'WR', 'TE'],
            'K': ['K']
        }
    
    def load_historical_data(self, positions: List[str], min_games: int = 6) -> pd.DataFrame:
        """Load historical data for specified positions"""
        logger.info(f"Loading historical data for positions: {positions}")
        
        with self.SessionLocal() as db:
            # Query to get aggregated stats per player per season
            query = db.query(
                Player.player_id,
                Player.first_name,
                Player.last_name,
                Player.position,
                Player.age,
                Player.years_exp,
                PlayerStats.season,
                func.count(PlayerStats.week).label('games_played'),
                func.sum(PlayerStats.fantasy_points_ppr).label('total_points_ppr'),
                func.avg(PlayerStats.fantasy_points_ppr).label('avg_points_ppr'),
                func.stddev(PlayerStats.fantasy_points_ppr).label('std_points_ppr'),
                func.max(PlayerStats.fantasy_points_ppr).label('max_points_ppr'),
                func.min(PlayerStats.fantasy_points_ppr).label('min_points_ppr'),
                func.sum(PlayerStats.fantasy_points_std).label('total_points_std'),
                func.avg(PlayerStats.fantasy_points_std).label('avg_points_std')
            ).join(
                Player, Player.player_id == PlayerStats.player_id
            ).filter(
                Player.position.in_(positions),
                PlayerStats.fantasy_points_ppr > 0  # Only games where player actually played
            ).group_by(
                Player.player_id,
                Player.first_name,
                Player.last_name,
                Player.position,
                Player.age,
                Player.years_exp,
                PlayerStats.season
            ).having(
                func.count(PlayerStats.week) >= min_games  # Minimum games threshold
            )
            
            df = pd.read_sql(query.statement, db.bind)
            
        logger.info(f"Loaded {len(df)} player-season records")
        return df
    
    def prepare_features_for_gmm(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare features for GMM clustering"""
        logger.info("Preparing features for GMM clustering")
        
        # Calculate additional features
        df['points_per_game'] = df['avg_points_ppr']
        df['consistency_score'] = 1 - (df['std_points_ppr'] / (df['avg_points_ppr'] + 1e-6))
        df['ceiling'] = df['max_points_ppr']
        df['floor'] = df['min_points_ppr']
        df['games_percentage'] = df['games_played'] / 17  # Assuming 17 game season
        
        # Position-specific adjustments
        position_values = {'QB': 1.0, 'RB': 0.8, 'WR': 0.8, 'TE': 0.6, 'K': 0.4}
        df['position_value'] = df['position'].map(position_values)
        
        # Select features for clustering
        feature_columns = [
            'points_per_game', 'consistency_score', 'ceiling', 'floor',
            'games_percentage', 'position_value', 'total_points_ppr'
        ]
        
        # Handle missing values
        features_df = df[feature_columns].fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Save scaler
        joblib.dump(scaler, self.models_dir / 'gmm_scaler.pkl')
        
        return features_scaled, df
    
    def train_gmm_model(self) -> Dict[str, Any]:
        """Train GMM clustering model for draft tiers"""
        logger.info("Training GMM clustering model...")
        
        # Load data for all offensive positions
        all_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        df = self.load_historical_data(all_positions)
        
        # Get most recent season data for each player
        df_recent = df.loc[df.groupby('player_id')['season'].idxmax()]
        
        # Prepare features
        features_scaled, df_with_features = self.prepare_features_for_gmm(df_recent)
        
        # Define feature names
        feature_names = [
            'points_per_game', 'consistency_score', 'ceiling', 'floor',
            'games_percentage', 'position_value', 'total_points_ppr'
        ]
        
        # Train GMM
        self.gmm_clusterer.fit(features_scaled, feature_names)
        
        # Create player names from first and last names
        player_names = (df_with_features['first_name'] + ' ' + df_with_features['last_name']).tolist()
        
        # Get cluster assignments
        tiers = self.gmm_clusterer.predict_tiers(
            features=features_scaled,
            player_ids=df_with_features['player_id'].tolist(),
            player_names=player_names,
            positions=df_with_features['position'].tolist(),
            expected_points=df_with_features['avg_points_ppr'].tolist()
        )
        df_with_features['cluster'] = [t.tier for t in tiers]
        
        # Analyze clusters
        cluster_stats = df_with_features.groupby('cluster').agg({
            'avg_points_ppr': ['mean', 'std', 'count'],
            'position': lambda x: x.value_counts().to_dict()
        })
        
        logger.info(f"Created {self.gmm_clusterer.n_components} draft tiers")
        logger.info(f"Cluster statistics:\n{cluster_stats}")
        
        # Save model
        model_path = self.models_dir / 'gmm_draft_tiers.pkl'
        joblib.dump(self.gmm_clusterer, model_path)
        logger.info(f"Saved GMM model to {model_path}")
        
        # Store tiers in database
        try:
            from backend.ml.draft_tier_storage import DraftTierStorage
            storage = DraftTierStorage()
            storage.store_draft_tiers(tiers, season=2024)
            logger.info("Stored draft tiers in database")
        except Exception as e:
            logger.error(f"Failed to store draft tiers: {str(e)}")
        
        return {
            'n_clusters': self.gmm_clusterer.n_components,
            'cluster_stats': cluster_stats.to_dict(),
            'model_path': str(model_path),
            'tiers_stored': len(tiers)
        }
    
    def prepare_features_for_nn(self, position: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare features for neural network training"""
        logger.info(f"Preparing features for {position} neural network")
        
        # Load detailed stats
        with self.SessionLocal() as db:
            query = db.query(
                PlayerStats.player_id,
                PlayerStats.season,
                PlayerStats.week,
                PlayerStats.fantasy_points_ppr,
                PlayerStats.stats,
                Player.position,
                Player.age,
                Player.years_exp
            ).join(
                Player, Player.player_id == PlayerStats.player_id
            ).filter(
                Player.position == position,
                PlayerStats.fantasy_points_ppr > 0
            )
            
            stats_df = pd.read_sql(query.statement, db.bind)
        
        # Extract features from JSONB stats
        stats_features = pd.json_normalize(stats_df['stats'])
        
        # Combine with player features
        features_df = pd.concat([
            stats_df[['player_id', 'season', 'week', 'age', 'years_exp', 'fantasy_points_ppr']],
            stats_features
        ], axis=1)
        
        # Create lag features (previous week performance)
        features_df = features_df.sort_values(['player_id', 'season', 'week'])
        lag_columns = ['fantasy_points_ppr', 'pts_ppr', 'pts_std']
        
        for col in lag_columns:
            if col in features_df.columns:
                features_df[f'{col}_lag1'] = features_df.groupby('player_id')[col].shift(1)
                features_df[f'{col}_lag2'] = features_df.groupby('player_id')[col].shift(2)
                features_df[f'{col}_rolling_avg'] = features_df.groupby('player_id')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Remove rows with NaN in target
        features_df = features_df.dropna(subset=['fantasy_points_ppr'])
        
        # Select relevant features based on position
        feature_columns = self.feature_engineer.get_position_features(position)
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        X = features_df[available_features].fillna(0).values
        y = features_df['fantasy_points_ppr'].values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return features_df, X, y
    
    def train_position_model(self, position: str) -> Dict[str, Any]:
        """Train neural network for a specific position"""
        logger.info(f"Training neural network for {position}")
        
        # Prepare data
        df, X, y = self.prepare_features_for_nn(position)
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for {position} ({len(X)} samples)")
            return {'status': 'skipped', 'reason': 'insufficient_data'}
        
        # Time-based split (don't use future data to predict past)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.models_dir / f'nn_scaler_{position}.pkl'
        joblib.dump(scaler, scaler_path)
        
        # Create and train model
        self.nn_predictor = FantasyNeuralNetwork(input_dim=X_train_scaled.shape[1])
        
        # Combine train and test data since fit will do its own split
        X_all = np.vstack([X_train_scaled, X_test_scaled])
        y_all = np.hstack([y_train, y_test])
        
        history = self.nn_predictor.fit(
            X_all, y_all,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=0  # Reduce verbosity
        )
        
        # Evaluate
        predictions = self.nn_predictor.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"{position} Model Performance:")
        logger.info(f"  MAE: {mae:.2f} points")
        logger.info(f"  RMSE: {rmse:.2f} points")
        logger.info(f"  R2: {r2:.3f}")
        
        # Save model
        model_path = self.models_dir / f'nn_model_{position}.h5'
        self.nn_predictor.save_model(model_path)
        
        return {
            'position': position,
            'samples': len(X),
            'features': X.shape[1],
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all models (GMM and position-specific NNs)"""
        logger.info("Starting full model training pipeline...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gmm': {},
            'neural_networks': {}
        }
        
        # Train GMM for draft tiers
        try:
            results['gmm'] = self.train_gmm_model()
        except Exception as e:
            logger.error(f"GMM training failed: {str(e)}")
            results['gmm'] = {'status': 'failed', 'error': str(e)}
        
        # Train neural networks for each position
        for position in ['QB', 'RB', 'WR', 'TE', 'K']:
            try:
                results['neural_networks'][position] = self.train_position_model(position)
            except Exception as e:
                logger.error(f"Neural network training failed for {position}: {str(e)}")
                results['neural_networks'][position] = {'status': 'failed', 'error': str(e)}
        
        # Save training results
        import json
        results_path = self.models_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training complete! Results saved to {results_path}")
        return results
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all trained models"""
        logger.info("Evaluating all models...")
        
        evaluation_results = {}
        
        # Evaluate each position model
        for position in ['QB', 'RB', 'WR', 'TE', 'K']:
            model_path = self.models_dir / f'nn_model_{position}.h5'
            if not model_path.exists():
                continue
                
            # Load model and data
            df, X, y = self.prepare_features_for_nn(position)
            
            # Load scaler
            scaler_path = self.models_dir / f'nn_scaler_{position}.pkl'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X)
                
                # Load model and predict
                self.nn_predictor.load_model(model_path)
                predictions = self.nn_predictor.predict(X_scaled)
                
                evaluation_results[position] = {
                    'mae': float(mean_absolute_error(y, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                    'r2': float(r2_score(y, predictions)),
                    'samples': len(y)
                }
        
        return evaluation_results


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    print("Training results:", results)