"""
ETL Pipeline for Fantasy Football AI Assistant.

This module orchestrates data collection, transformation, and loading
for machine learning model training and prediction.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from .sources.nfl import ESPNDataCollector, create_espn_collector
from .storage.models import DatabaseManager, get_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FantasyDataETL:
    """
    Complete ETL pipeline for fantasy football data processing.
    
    Handles data extraction from ESPN API, transformation for ML features,
    and loading into structured formats for model training.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize ETL pipeline.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or get_database()
        self.espn_collector = create_espn_collector(db_path=self.db_manager.db_path)
        
        # Feature engineering configuration
        self.feature_config = {
            'rolling_windows': [3, 4, 6],  # Windows for rolling averages
            'min_games_threshold': 4,      # Minimum games for valid stats
            'positions': ['QB', 'RB', 'WR', 'TE'],  # Positions to process
            'outlier_threshold': 3.0       # Standard deviations for outlier detection
        }
        
        logger.info("Fantasy Data ETL Pipeline initialized")
    
    def run_full_pipeline(
        self, 
        collect_fresh_data: bool = True,
        process_features: bool = True,
        save_training_data: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete ETL pipeline.
        
        Args:
            collect_fresh_data: Whether to collect new data from API
            process_features: Whether to generate ML features
            save_training_data: Whether to save processed data for training
            
        Returns:
            Pipeline execution summary
        """
        pipeline_start = datetime.now()
        logger.info("Starting Full ETL Pipeline")
        
        summary = {
            'start_time': pipeline_start,
            'steps_completed': [],
            'data_stats': {},
            'errors': []
        }
        
        try:
            # Step 1: Data Collection
            if collect_fresh_data:
                logger.info("Step 1: Collecting fresh data from ESPN API")
                collection_result = self._collect_data()
                summary['steps_completed'].append('data_collection')
                summary['data_stats']['raw_records'] = len(collection_result) if not collection_result.empty else 0
            else:
                logger.info("Step 1: Skipping data collection (using existing data)")
                summary['steps_completed'].append('data_collection_skipped')
            
            # Step 2: Data Validation and Cleaning
            logger.info("Step 2: Validating and cleaning data")
            validation_result = self._validate_and_clean_data()
            summary['steps_completed'].append('data_validation')
            summary['data_stats'].update(validation_result)
            
            # Step 3: Feature Engineering
            if process_features:
                logger.info("Step 3: Engineering ML features")
                feature_result = self._engineer_features()
                summary['steps_completed'].append('feature_engineering')
                summary['data_stats'].update(feature_result)
            else:
                logger.info("Step 3: Skipping feature engineering")
                summary['steps_completed'].append('feature_engineering_skipped')
            
            # Step 4: Training Data Preparation
            if save_training_data:
                logger.info("Step 4: Preparing and saving training datasets")
                training_result = self._prepare_training_data()
                summary['steps_completed'].append('training_data_prep')
                summary['data_stats'].update(training_result)
            else:
                logger.info("Step 4: Skipping training data preparation")
                summary['steps_completed'].append('training_data_prep_skipped')
            
            # Pipeline completion
            pipeline_end = datetime.now()
            summary['end_time'] = pipeline_end
            summary['duration'] = (pipeline_end - pipeline_start).total_seconds()
            summary['status'] = 'SUCCESS'
            
            logger.info(f"ETL Pipeline completed successfully in {summary['duration']:.2f} seconds")
            self._log_pipeline_summary(summary)
            
            return summary
            
        except Exception as e:
            summary['status'] = 'FAILED'
            summary['errors'].append(str(e))
            logger.error(f"ETL Pipeline failed: {e}")
            raise
    
    def _collect_data(self) -> pd.DataFrame:
        """Collect fresh data from ESPN API."""
        try:
            data = self.espn_collector.collect_all_seasons(
                save_to_db=True,
                save_to_csv=True
            )
            logger.info(f"Collected {len(data)} records from ESPN API")
            return data
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
    
    def _validate_and_clean_data(self) -> Dict[str, Any]:
        """Validate data integrity and clean issues."""
        validation_stats = {}
        
        # Check database integrity
        integrity_issues = self.db_manager.validate_data_integrity()
        validation_stats['integrity_issues'] = integrity_issues
        
        # Get overall database statistics
        db_stats = self.db_manager.get_database_stats()
        validation_stats.update(db_stats)
        
        # Log any significant issues
        for issue_type, count in integrity_issues.items():
            if count > 0:
                logger.warning(f"Found {count} {issue_type}")
        
        logger.info("Data validation completed")
        return validation_stats
    
    def _engineer_features(self) -> Dict[str, Any]:
        """Engineer ML features from raw data."""
        feature_stats = {}
        
        try:
            # Get training seasons data
            seasons = [2022, 2023, 2024]
            raw_data = self.db_manager.get_training_data(seasons)
            
            if raw_data.empty:
                logger.warning("No data available for feature engineering")
                return {'features_created': 0}
            
            logger.info(f"Engineering features for {len(raw_data)} records")
            
            # Create features by position
            all_features = []
            for position in self.feature_config['positions']:
                position_data = raw_data[raw_data['position'] == position].copy()
                if len(position_data) > 0:
                    position_features = self._create_player_features(position_data)
                    all_features.append(position_features)
                    logger.info(f"Created features for {len(position_features)} {position} records")
            
            if all_features:
                # Combine all position features
                feature_df = pd.concat(all_features, ignore_index=True)
                feature_stats['features_created'] = len(feature_df)
                feature_stats['unique_players'] = feature_df['player_id'].nunique()
                feature_stats['feature_columns'] = len([col for col in feature_df.columns if col.startswith('feat_')])
                
                # Save engineered features
                self._save_features(feature_df)
                
                logger.info(f"Feature engineering completed: {feature_stats['features_created']} records with {feature_stats['feature_columns']} features")
            else:
                logger.warning("No features could be created")
                feature_stats['features_created'] = 0
            
            return feature_stats
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _create_player_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features for players in the dataset.
        
        Args:
            player_data: DataFrame with player statistics
            
        Returns:
            DataFrame with engineered features
        """
        # Sort by player and time
        player_data = player_data.sort_values(['player_id', 'season', 'week'])
        
        features_list = []
        
        # Group by player for feature creation
        for player_id, player_df in player_data.groupby('player_id'):
            if len(player_df) < self.feature_config['min_games_threshold']:
                continue  # Skip players with insufficient data
            
            player_features = self._calculate_player_features(player_df)
            features_list.append(player_features)
        
        return pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
    
    def _calculate_player_features(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for a single player."""
        player_df = player_df.copy().reset_index(drop=True)
        
        # Basic information
        player_df['feat_games_played'] = len(player_df)
        
        # Basic statistics
        player_df['feat_avg_points'] = player_df['fantasy_points'].mean()
        player_df['feat_std_points'] = player_df['fantasy_points'].std()
        player_df['feat_consistency'] = player_df['feat_avg_points'] / (player_df['feat_std_points'] + 0.01)  # Avoid division by zero
        
        # Projection accuracy
        player_df['feat_avg_projection_error'] = (player_df['fantasy_points'] - player_df['projected_points']).mean()
        player_df['feat_projection_accuracy'] = 1.0 - abs(player_df['feat_avg_projection_error']) / (player_df['feat_avg_points'] + 0.01)
        
        # Performance trends (rolling averages)
        for window in self.feature_config['rolling_windows']:
            if len(player_df) >= window:
                rolling_avg = player_df['fantasy_points'].rolling(window=window, min_periods=1).mean()
                player_df[f'feat_rolling_avg_{window}'] = rolling_avg
                
                # Momentum (recent vs overall average)
                recent_avg = rolling_avg.iloc[-min(window, len(player_df)):].mean()
                player_df[f'feat_momentum_{window}'] = recent_avg - player_df['feat_avg_points']
        
        # Boom/Bust analysis
        boom_threshold = player_df['feat_avg_points'] + player_df['feat_std_points']
        bust_threshold = player_df['feat_avg_points'] - player_df['feat_std_points']
        
        player_df['feat_boom_rate'] = (player_df['fantasy_points'] > boom_threshold).mean()
        player_df['feat_bust_rate'] = (player_df['fantasy_points'] < bust_threshold).mean()
        
        # Recent performance (last 25% of games)
        recent_games = max(1, len(player_df) // 4)
        recent_data = player_df.tail(recent_games)
        player_df['feat_recent_avg'] = recent_data['fantasy_points'].mean()
        player_df['feat_recent_trend'] = player_df['feat_recent_avg'] - player_df['feat_avg_points']
        
        # Efficiency metrics
        player_df['feat_points_per_projection'] = player_df['fantasy_points'] / (player_df['projected_points'] + 0.01)
        player_df['feat_avg_efficiency'] = player_df['feat_points_per_projection'].mean()
        
        # Season-based features
        for season, season_df in player_df.groupby('season'):
            if len(season_df) >= 2:  # Minimum games in season
                season_avg = season_df['fantasy_points'].mean()
                player_df.loc[player_df['season'] == season, f'feat_season_{season}_avg'] = season_avg
        
        # Target variable (next game performance)
        player_df['target_next_game_points'] = player_df['fantasy_points'].shift(-1)
        
        return player_df
    
    def _save_features(self, feature_df: pd.DataFrame) -> None:
        """Save engineered features to file."""
        try:
            # Create data directory
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            feature_file = "data/processed/engineered_features.csv"
            feature_df.to_csv(feature_file, index=False)
            logger.info(f"Features saved to {feature_file}")
            
            # Save feature metadata
            feature_columns = [col for col in feature_df.columns if col.startswith('feat_')]
            metadata = {
                'total_features': len(feature_columns),
                'feature_columns': feature_columns,
                'created_at': datetime.now().isoformat(),
                'total_records': len(feature_df),
                'unique_players': feature_df['player_id'].nunique()
            }
            
            metadata_file = "data/processed/feature_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Feature metadata saved to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            raise
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """Prepare datasets for ML model training."""
        training_stats = {}
        
        try:
            # Load engineered features
            feature_file = "data/processed/engineered_features.csv"
            if not Path(feature_file).exists():
                logger.warning("No engineered features found, skipping training data preparation")
                return {'training_datasets': 0}
            
            features_df = pd.read_csv(feature_file)
            logger.info(f"Loaded {len(features_df)} feature records")
            
            # Create training datasets by position
            datasets_created = 0
            
            for position in self.feature_config['positions']:
                position_data = features_df[features_df['position'] == position].copy()
                
                if len(position_data) > 50:  # Minimum threshold for training
                    # Prepare training/validation split
                    train_data, val_data = self._split_training_data(position_data)
                    
                    # Save position-specific datasets
                    train_file = f"data/processed/train_{position.lower()}.csv"
                    val_file = f"data/processed/val_{position.lower()}.csv"
                    
                    train_data.to_csv(train_file, index=False)
                    val_data.to_csv(val_file, index=False)
                    
                    datasets_created += 1
                    logger.info(f"Created {position} training dataset: {len(train_data)} train, {len(val_data)} validation")
            
            training_stats['training_datasets'] = datasets_created
            
            # Create combined dataset for multi-position models
            if datasets_created > 0:
                combined_file = "data/processed/train_combined.csv"
                features_df.to_csv(combined_file, index=False)
                training_stats['combined_dataset_size'] = len(features_df)
                logger.info(f"Created combined training dataset with {len(features_df)} records")
            
            return training_stats
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise
    
    def _split_training_data(self, data: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            data: DataFrame to split
            val_ratio: Ratio of data to use for validation
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        # Sort by season and week for temporal split
        data = data.sort_values(['season', 'week'])
        
        # Use temporal split (most recent data for validation)
        split_idx = int(len(data) * (1 - val_ratio))
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        
        return train_data, val_data
    
    def _log_pipeline_summary(self, summary: Dict[str, Any]) -> None:
        """Log pipeline execution summary."""
        logger.info("=" * 50)
        logger.info("ETL PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Status: {summary.get('status', 'UNKNOWN')}")
        logger.info(f"Duration: {summary.get('duration', 0):.2f} seconds")
        logger.info(f"Steps completed: {', '.join(summary.get('steps_completed', []))}")
        
        if summary.get('data_stats'):
            logger.info("\nData Statistics:")
            for key, value in summary['data_stats'].items():
                logger.info(f"  {key}: {value}")
        
        if summary.get('errors'):
            logger.error("\nErrors encountered:")
            for error in summary['errors']:
                logger.error(f"  {error}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of data pipeline."""
        status = {
            'database_stats': self.db_manager.get_database_stats(),
            'integrity_check': self.db_manager.validate_data_integrity(),
            'data_files': self._check_data_files(),
            'last_update': self._get_last_update_time()
        }
        return status
    
    def _check_data_files(self) -> Dict[str, bool]:
        """Check existence of key data files."""
        files_to_check = {
            'raw_combined_csv': Path("data/fantasy_weekly_stats_combined.csv").exists(),
            'engineered_features': Path("data/processed/engineered_features.csv").exists(),
            'feature_metadata': Path("data/processed/feature_metadata.json").exists(),
            'training_combined': Path("data/processed/train_combined.csv").exists()
        }
        return files_to_check
    
    def _get_last_update_time(self) -> Optional[str]:
        """Get timestamp of last data update."""
        try:
            metadata_file = Path("data/processed/feature_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('created_at')
        except Exception:
            pass
        return None


def run_etl_pipeline(
    collect_data: bool = True,
    process_features: bool = True,
    save_training: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run the ETL pipeline.
    
    Args:
        collect_data: Whether to collect fresh data
        process_features: Whether to engineer features
        save_training: Whether to prepare training datasets
        
    Returns:
        Pipeline execution summary
    """
    etl = FantasyDataETL()
    return etl.run_full_pipeline(
        collect_fresh_data=collect_data,
        process_features=process_features,
        save_training_data=save_training
    )


if __name__ == "__main__":
    # Example usage
    try:
        # Run full pipeline
        result = run_etl_pipeline(
            collect_data=True,
            process_features=True,
            save_training=True
        )
        
        print("\nETL Pipeline completed successfully!")
        print(f"Execution time: {result.get('duration', 0):.2f} seconds")
        
        # Print key statistics
        if result.get('data_stats'):
            print("\nKey Statistics:")
            stats = result['data_stats']
            if 'total_players' in stats:
                print(f"  Total players in database: {stats['total_players']}")
            if 'features_created' in stats:
                print(f"  ML features created: {stats['features_created']}")
            if 'training_datasets' in stats:
                print(f"  Training datasets prepared: {stats['training_datasets']}")
        
    except Exception as e:
        print(f"ETL Pipeline failed: {e}")
        raise