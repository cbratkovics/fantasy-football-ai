#!/usr/bin/env python3
"""
Train Neural Network models for fantasy football predictions
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.neural_network import FantasyNeuralNetwork
from backend.ml.train import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_neural_networks():
    """Train neural network models for each position"""
    logger.info("Training Neural Network Models")
    logger.info("="*60)
    
    trainer = ModelTrainer()
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K']
    results = {}
    
    for position in positions:
        logger.info(f"\nTraining Neural Network for {position}")
        try:
            # Prepare data
            df, X, y = trainer.prepare_features_for_nn(position)
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for {position} ({len(X)} samples)")
                continue
            
            logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train neural network
            nn = FantasyNeuralNetwork(input_dim=X_train_scaled.shape[1])
            
            # Build model (internally done by fit)
            model = nn._build_model(position=position)
            logger.info(f"Model architecture: {model.summary()}")
            
            # Train - using all data with validation split
            X_all = np.vstack([X_train_scaled, X_test_scaled])
            y_all = np.hstack([y_train, y_test])
            
            history = nn.fit(
                X_all, y_all,
                positions=[position] * len(X_all),  # Position info for each sample
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            # Save model
            model_path = models_dir / f'nn_model_{position}.h5'
            nn.save_model(str(model_path))
            
            # Save scaler
            scaler_path = models_dir / f'nn_scaler_{position}.pkl'
            joblib.dump(scaler, scaler_path)
            
            # Get final metrics
            if position in history:
                final_metrics = history[position]
                results[position] = {
                    'status': 'success',
                    'final_loss': final_metrics['final_loss'],
                    'final_mae': final_metrics['final_mae'],
                    'samples': final_metrics['samples']
                }
                logger.info(f"{position} training complete - MAE: {final_metrics['final_mae']:.2f}")
            else:
                # Single model trained
                results[position] = {
                    'status': 'success',
                    'final_loss': history['final_loss'],
                    'final_mae': history['final_mae'],
                    'samples': history['samples']
                }
                logger.info(f"{position} training complete - MAE: {history['final_mae']:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to train {position}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[position] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(models_dir / 'nn_training_results.csv')
    
    logger.info("\n" + "="*60)
    logger.info("Neural Network Training Summary:")
    logger.info(results_df)
    
    return results


if __name__ == "__main__":
    train_neural_networks()