"""
ML Model Versioning System
Manages model lifecycle, A/B testing, and performance tracking
"""
import os
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import shutil
import sqlite3

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for model versions"""
    model_id: str
    version: str
    model_type: str  # 'random_forest', 'neural_network', 'ensemble'
    position: str
    created_at: datetime
    created_by: str
    description: str
    
    # Training metrics
    training_mae: float
    training_rmse: float
    training_r2: float
    validation_mae: float
    validation_rmse: float
    validation_r2: float
    
    # Model configuration
    hyperparameters: Dict[str, Any]
    features_used: List[str]
    training_data_size: int
    training_data_hash: str
    
    # Deployment info
    status: str = "development"  # development, staging, production, archived
    deployed_at: Optional[datetime] = None
    is_champion: bool = False
    performance_history: List[Dict] = field(default_factory=list)
    
    # A/B testing
    traffic_percentage: float = 0.0
    ab_test_group: Optional[str] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model evaluation"""
    model_id: str
    version: str
    evaluation_date: datetime
    
    # Accuracy metrics
    mae: float
    rmse: float
    r2: float
    mape: float  # Mean Absolute Percentage Error
    
    # Business metrics
    prediction_accuracy_rate: float  # % of predictions within threshold
    prediction_latency_ms: float
    prediction_count: int
    
    # Position-specific metrics
    position_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Comparative metrics
    improvement_vs_baseline: float = 0.0
    improvement_vs_previous: float = 0.0


class ModelVersioningSystem:
    """
    Comprehensive model versioning system with A/B testing capabilities
    """
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize model registry database
        self.db_path = self.base_path / "model_registry.db"
        self._init_database()
        
        # Model cache
        self._model_cache = {}
        self._metadata_cache = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_r2": 0.7,
            "max_mae": 5.0,
            "min_accuracy_rate": 0.8,
            "max_latency_ms": 200
        }
    
    def _init_database(self):
        """Initialize SQLite database for model registry"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT,
                    version TEXT,
                    model_type TEXT,
                    position TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    description TEXT,
                    training_mae REAL,
                    training_rmse REAL,
                    training_r2 REAL,
                    validation_mae REAL,
                    validation_rmse REAL,
                    validation_r2 REAL,
                    hyperparameters TEXT,
                    features_used TEXT,
                    training_data_size INTEGER,
                    training_data_hash TEXT,
                    status TEXT,
                    deployed_at TEXT,
                    is_champion INTEGER,
                    traffic_percentage REAL,
                    ab_test_group TEXT,
                    PRIMARY KEY (model_id, version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_id TEXT,
                    version TEXT,
                    evaluation_date TEXT,
                    mae REAL,
                    rmse REAL,
                    r2 REAL,
                    mape REAL,
                    prediction_accuracy_rate REAL,
                    prediction_latency_ms REAL,
                    prediction_count INTEGER,
                    position_metrics TEXT,
                    improvement_vs_baseline REAL,
                    improvement_vs_previous REAL,
                    PRIMARY KEY (model_id, version, evaluation_date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT,
                    model_a_id TEXT,
                    model_a_version TEXT,
                    model_b_id TEXT,
                    model_b_version TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    sample_size INTEGER,
                    winner TEXT,
                    confidence REAL,
                    metrics TEXT,
                    PRIMARY KEY (test_id)
                )
            """)
    
    def register_model(
        self,
        model: BaseEstimator,
        model_type: str,
        position: str,
        hyperparameters: Dict[str, Any],
        features_used: List[str],
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        created_by: str = "system",
        description: str = "",
        version: Optional[str] = None
    ) -> str:
        """
        Register a new model version
        
        Returns:
            Model version ID
        """
        # Generate model ID and version
        model_id = f"{model_type}_{position}".lower()
        if version is None:
            version = self._generate_version(model_id)
        
        # Calculate training metrics
        train_pred = model.predict(training_data[features_used])
        val_pred = model.predict(validation_data[features_used])
        
        training_mae = mean_absolute_error(y_train, train_pred)
        training_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        training_r2 = r2_score(y_train, train_pred)
        
        validation_mae = mean_absolute_error(y_val, val_pred)
        validation_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        validation_r2 = r2_score(y_val, val_pred)
        
        # Calculate data hash for reproducibility
        data_hash = self._calculate_data_hash(training_data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            model_type=model_type,
            position=position,
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description,
            training_mae=training_mae,
            training_rmse=training_rmse,
            training_r2=training_r2,
            validation_mae=validation_mae,
            validation_rmse=validation_rmse,
            validation_r2=validation_r2,
            hyperparameters=hyperparameters,
            features_used=features_used,
            training_data_size=len(training_data),
            training_data_hash=data_hash
        )
        
        # Save model and metadata
        self._save_model(model, metadata)
        self._save_metadata(metadata)
        
        logger.info(f"Registered model {model_id} version {version}")
        return f"{model_id}:{version}"
    
    def _generate_version(self, model_id: str) -> str:
        """Generate new version number for model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version FROM model_metadata WHERE model_id = ? ORDER BY created_at DESC LIMIT 1",
                (model_id,)
            )
            result = cursor.fetchone()
            
            if result:
                # Increment version
                current_version = result[0]
                if current_version.startswith("v"):
                    version_num = int(current_version[1:]) + 1
                else:
                    version_num = int(current_version) + 1
                return f"v{version_num}"
            else:
                return "v1"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for reproducibility"""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _save_model(self, model: BaseEstimator, metadata: ModelMetadata):
        """Save model to filesystem"""
        model_dir = self.base_path / metadata.model_id / metadata.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save features list
        features_path = model_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump(metadata.features_used, f)
        
        # Save hyperparameters
        params_path = model_dir / "hyperparameters.json"
        with open(params_path, 'w') as f:
            json.dump(metadata.hyperparameters, f, default=str)
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_metadata VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                metadata.model_id,
                metadata.version,
                metadata.model_type,
                metadata.position,
                metadata.created_at.isoformat(),
                metadata.created_by,
                metadata.description,
                metadata.training_mae,
                metadata.training_rmse,
                metadata.training_r2,
                metadata.validation_mae,
                metadata.validation_rmse,
                metadata.validation_r2,
                json.dumps(metadata.hyperparameters, default=str),
                json.dumps(metadata.features_used),
                metadata.training_data_size,
                metadata.training_data_hash,
                metadata.status,
                metadata.deployed_at.isoformat() if metadata.deployed_at else None,
                int(metadata.is_champion),
                metadata.traffic_percentage,
                metadata.ab_test_group
            ))
    
    def load_model(self, model_id: str, version: str = "latest") -> Tuple[BaseEstimator, ModelMetadata]:
        """Load model and metadata"""
        # Get version if "latest"
        if version == "latest":
            version = self._get_latest_version(model_id)
        
        cache_key = f"{model_id}:{version}"
        
        # Check cache
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], self._metadata_cache[cache_key]
        
        # Load from filesystem
        model_dir = self.base_path / model_id / version
        
        if not model_dir.exists():
            raise ValueError(f"Model {model_id} version {version} not found")
        
        # Load model
        model_path = model_dir / "model.pkl"
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = self._load_metadata(model_id, version)
        
        # Cache
        self._model_cache[cache_key] = model
        self._metadata_cache[cache_key] = metadata
        
        return model, metadata
    
    def _get_latest_version(self, model_id: str) -> str:
        """Get latest version for model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version FROM model_metadata WHERE model_id = ? ORDER BY created_at DESC LIMIT 1",
                (model_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"No versions found for model {model_id}")
            
            return result[0]
    
    def _load_metadata(self, model_id: str, version: str) -> ModelMetadata:
        """Load metadata from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM model_metadata WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"Metadata not found for {model_id}:{version}")
            
            # Parse result
            return ModelMetadata(
                model_id=result[0],
                version=result[1],
                model_type=result[2],
                position=result[3],
                created_at=datetime.fromisoformat(result[4]),
                created_by=result[5],
                description=result[6],
                training_mae=result[7],
                training_rmse=result[8],
                training_r2=result[9],
                validation_mae=result[10],
                validation_rmse=result[11],
                validation_r2=result[12],
                hyperparameters=json.loads(result[13]),
                features_used=json.loads(result[14]),
                training_data_size=result[15],
                training_data_hash=result[16],
                status=result[17],
                deployed_at=datetime.fromisoformat(result[18]) if result[18] else None,
                is_champion=bool(result[19]),
                traffic_percentage=result[20],
                ab_test_group=result[21]
            )
    
    def promote_to_production(
        self,
        model_id: str,
        version: str,
        traffic_percentage: float = 100.0
    ) -> bool:
        """Promote model version to production"""
        # Validate model meets thresholds
        metadata = self._load_metadata(model_id, version)
        
        if not self._validate_performance(metadata):
            logger.error(f"Model {model_id}:{version} does not meet performance thresholds")
            return False
        
        # Update status
        with sqlite3.connect(self.db_path) as conn:
            # Demote current champion
            conn.execute(
                "UPDATE model_metadata SET is_champion = 0, traffic_percentage = 0 WHERE model_id = ? AND is_champion = 1",
                (model_id,)
            )
            
            # Promote new champion
            conn.execute("""
                UPDATE model_metadata 
                SET status = 'production', is_champion = 1, traffic_percentage = ?, deployed_at = ?
                WHERE model_id = ? AND version = ?
            """, (traffic_percentage, datetime.utcnow().isoformat(), model_id, version))
        
        logger.info(f"Promoted {model_id}:{version} to production with {traffic_percentage}% traffic")
        return True
    
    def _validate_performance(self, metadata: ModelMetadata) -> bool:
        """Validate model meets performance thresholds"""
        thresholds = self.performance_thresholds
        
        if metadata.validation_r2 < thresholds["min_r2"]:
            return False
        
        if metadata.validation_mae > thresholds["max_mae"]:
            return False
        
        return True
    
    def start_ab_test(
        self,
        model_a_id: str,
        model_a_version: str,
        model_b_id: str,
        model_b_version: str,
        traffic_split: float = 0.5,
        test_duration_days: int = 7
    ) -> str:
        """Start A/B test between two models"""
        test_id = f"ab_test_{int(datetime.utcnow().timestamp())}"
        
        # Update traffic allocation
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE model_metadata 
                SET traffic_percentage = ?, ab_test_group = 'A'
                WHERE model_id = ? AND version = ?
            """, (traffic_split * 100, model_a_id, model_a_version))
            
            conn.execute("""
                UPDATE model_metadata 
                SET traffic_percentage = ?, ab_test_group = 'B'
                WHERE model_id = ? AND version = ?
            """, ((1 - traffic_split) * 100, model_b_id, model_b_version))
            
            # Record test
            end_date = datetime.utcnow() + timedelta(days=test_duration_days)
            conn.execute("""
                INSERT INTO ab_test_results (
                    test_id, model_a_id, model_a_version, model_b_id, model_b_version,
                    start_date, end_date, sample_size, winner, confidence, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id, model_a_id, model_a_version, model_b_id, model_b_version,
                datetime.utcnow().isoformat(), end_date.isoformat(),
                0, None, 0.0, "{}"
            ))
        
        logger.info(f"Started A/B test {test_id}")
        return test_id
    
    def evaluate_model_performance(
        self,
        model_id: str,
        version: str,
        test_data: pd.DataFrame,
        y_true: np.ndarray,
        features_used: List[str]
    ) -> ModelPerformanceMetrics:
        """Evaluate model performance on test data"""
        model, metadata = self.load_model(model_id, version)
        
        # Make predictions
        start_time = datetime.utcnow()
        y_pred = model.predict(test_data[features_used])
        prediction_latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        # Accuracy rate (predictions within 20% of actual)
        accuracy_rate = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6) <= 0.2)
        
        # Create performance metrics
        performance = ModelPerformanceMetrics(
            model_id=model_id,
            version=version,
            evaluation_date=datetime.utcnow(),
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            prediction_accuracy_rate=accuracy_rate,
            prediction_latency_ms=prediction_latency,
            prediction_count=len(y_pred)
        )
        
        # Save performance metrics
        self._save_performance_metrics(performance)
        
        return performance
    
    def _save_performance_metrics(self, performance: ModelPerformanceMetrics):
        """Save performance metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_performance VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                performance.model_id,
                performance.version,
                performance.evaluation_date.isoformat(),
                performance.mae,
                performance.rmse,
                performance.r2,
                performance.mape,
                performance.prediction_accuracy_rate,
                performance.prediction_latency_ms,
                performance.prediction_count,
                json.dumps(performance.position_metrics),
                performance.improvement_vs_baseline,
                performance.improvement_vs_previous
            ))
    
    def get_model_registry(self) -> pd.DataFrame:
        """Get all registered models"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(
                "SELECT * FROM model_metadata ORDER BY created_at DESC",
                conn
            )
    
    def get_champion_models(self) -> Dict[str, Dict]:
        """Get current champion models for each position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT model_id, version, position FROM model_metadata WHERE is_champion = 1"
            )
            results = cursor.fetchall()
            
            champions = {}
            for model_id, version, position in results:
                champions[position] = {
                    "model_id": model_id,
                    "version": version
                }
            
            return champions
    
    def archive_model(self, model_id: str, version: str):
        """Archive old model version"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_metadata SET status = 'archived' WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
        
        logger.info(f"Archived model {model_id}:{version}")
    
    def cleanup_old_versions(self, keep_versions: int = 5):
        """Clean up old model versions, keeping only the most recent"""
        with sqlite3.connect(self.db_path) as conn:
            # Get all models
            cursor = conn.execute("SELECT DISTINCT model_id FROM model_metadata")
            model_ids = [row[0] for row in cursor.fetchall()]
            
            for model_id in model_ids:
                # Get versions ordered by date
                cursor = conn.execute(
                    "SELECT version FROM model_metadata WHERE model_id = ? ORDER BY created_at DESC",
                    (model_id,)
                )
                versions = [row[0] for row in cursor.fetchall()]
                
                # Archive old versions
                if len(versions) > keep_versions:
                    old_versions = versions[keep_versions:]
                    for version in old_versions:
                        if not self._is_champion(model_id, version):
                            self.archive_model(model_id, version)
                            
                            # Delete files
                            model_dir = self.base_path / model_id / version
                            if model_dir.exists():
                                shutil.rmtree(model_dir)
    
    def _is_champion(self, model_id: str, version: str) -> bool:
        """Check if model version is current champion"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT is_champion FROM model_metadata WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
            result = cursor.fetchone()
            return bool(result[0]) if result else False


# Example usage
if __name__ == "__main__":
    # Initialize versioning system
    versioning = ModelVersioningSystem()
    
    # Example: Register a model (would be called from training script)
    # model = RandomForestRegressor()
    # versioning.register_model(
    #     model=model,
    #     model_type="random_forest",
    #     position="QB",
    #     hyperparameters={"n_estimators": 100, "max_depth": 10},
    #     features_used=["feature1", "feature2"],
    #     training_data=train_df,
    #     validation_data=val_df,
    #     y_train=y_train,
    #     y_val=y_val,
    #     description="Baseline RandomForest model"
    # )
    
    # Get model registry
    registry = versioning.get_model_registry()
    print("Model Registry:")
    print(registry.head())
    
    # Get champion models
    champions = versioning.get_champion_models()
    print("\nChampion Models:")
    print(champions)