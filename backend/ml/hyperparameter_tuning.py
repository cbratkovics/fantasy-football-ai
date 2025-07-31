"""
Hyperparameter Tuning System using Optuna
Automatically finds optimal hyperparameters for all models
"""

import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple, Callable
import logging
import json
from datetime import datetime

from .advanced_models import (
    FantasyFootballTransformer,
    FantasyFootballLSTM,
    FantasyFootballCNN,
    HybridFantasyModel
)

logger = logging.getLogger(__name__)


class FantasyModelTuner:
    """
    Comprehensive hyperparameter tuning for fantasy football models
    """
    
    def __init__(self, 
                 model_type: str = 'hybrid',
                 n_trials: int = 100,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Best parameters storage
        self.best_params = {}
        self.best_score = float('inf')
        
        # Model builders
        self.model_builders = {
            'transformer': self._build_transformer,
            'lstm': self._build_lstm,
            'cnn': self._build_cnn,
            'hybrid': self._build_hybrid,
            'traditional': self._build_traditional_nn
        }
        
    def objective(self, trial: optuna.Trial, 
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for Optuna optimization
        """
        # Clear previous models to free memory
        keras.backend.clear_session()
        
        # Get model based on type
        model = self.model_builders[self.model_type](trial, X_train.shape)
        
        # Compile model with suggested optimizer parameters
        optimizer = self._suggest_optimizer(trial)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Training parameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 20, 100)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=trial.suggest_int('early_stopping_patience', 5, 20),
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=trial.suggest_float('lr_reduction_factor', 0.3, 0.7),
                patience=trial.suggest_int('lr_patience', 3, 10),
                min_lr=1e-6
            ),
            TFKerasPruningCallback(trial, 'val_loss')
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        # Return best validation loss
        return min(history.history['val_loss'])
    
    def _suggest_optimizer(self, trial: optuna.Trial) -> keras.optimizers.Optimizer:
        """Suggest optimizer and its parameters"""
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
        
        if optimizer_name == 'adam':
            return keras.optimizers.Adam(
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                beta_1=trial.suggest_float('beta_1', 0.8, 0.99),
                beta_2=trial.suggest_float('beta_2', 0.9, 0.999),
                epsilon=trial.suggest_float('epsilon', 1e-8, 1e-6, log=True)
            )
        elif optimizer_name == 'adamw':
            return keras.optimizers.AdamW(
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            )
        elif optimizer_name == 'sgd':
            return keras.optimizers.SGD(
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                momentum=trial.suggest_float('momentum', 0.8, 0.99),
                nesterov=trial.suggest_categorical('nesterov', [True, False])
            )
        else:  # rmsprop
            return keras.optimizers.RMSprop(
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                rho=trial.suggest_float('rho', 0.8, 0.99)
            )
    
    def _build_transformer(self, trial: optuna.Trial, input_shape: Tuple) -> keras.Model:
        """Build transformer model with suggested hyperparameters"""
        # Model architecture parameters
        embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        ff_dim = trial.suggest_int('ff_dim', 128, 1024, step=128)
        num_blocks = trial.suggest_int('num_transformer_blocks', 2, 6)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Build model
        model = FantasyFootballTransformer(
            num_features=input_shape[-1],
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_blocks,
            dropout=dropout
        )
        
        return model
    
    def _build_lstm(self, trial: optuna.Trial, input_shape: Tuple) -> keras.Model:
        """Build LSTM model with suggested hyperparameters"""
        # Architecture parameters
        num_layers = trial.suggest_int('num_lstm_layers', 1, 4)
        lstm_units = []
        for i in range(num_layers):
            units = trial.suggest_int(f'lstm_units_layer_{i}', 32, 256, step=32)
            lstm_units.append(units)
        
        # Dense layers
        num_dense = trial.suggest_int('num_dense_layers', 1, 3)
        dense_units = []
        for i in range(num_dense):
            units = trial.suggest_int(f'dense_units_layer_{i}', 32, 256, step=32)
            dense_units.append(units)
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.3)
        
        # Build model
        model = FantasyFootballLSTM(
            num_features=input_shape[-1],
            lstm_units=lstm_units,
            dense_units=dense_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
        
        return model
    
    def _build_cnn(self, trial: optuna.Trial, input_shape: Tuple) -> keras.Model:
        """Build CNN model with suggested hyperparameters"""
        # Architecture parameters
        num_conv_layers = trial.suggest_int('num_conv_layers', 2, 5)
        filters = []
        kernel_sizes = []
        
        for i in range(num_conv_layers):
            filters.append(trial.suggest_int(f'filters_layer_{i}', 32, 256, step=32))
            kernel_sizes.append(trial.suggest_int(f'kernel_size_layer_{i}', 2, 5))
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Build model
        model = FantasyFootballCNN(
            num_features=input_shape[-1],
            filters=filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        return model
    
    def _build_hybrid(self, trial: optuna.Trial, input_shape: Tuple) -> keras.Model:
        """Build hybrid model with suggested hyperparameters"""
        # CNN parameters
        cnn_filters = [
            trial.suggest_int('cnn_filters_1', 16, 64, step=16),
            trial.suggest_int('cnn_filters_2', 32, 128, step=32)
        ]
        
        # LSTM parameters
        lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
        
        # Transformer parameters
        transformer_heads = trial.suggest_categorical('transformer_heads', [2, 4, 8])
        embed_dim = trial.suggest_categorical('embed_dim', [32, 64, 128])
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Build model
        model = HybridFantasyModel(
            num_features=input_shape[-1],
            num_static_features=input_shape[-1] // 2,  # Assume half are static
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            transformer_heads=transformer_heads,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        return model
    
    def _build_traditional_nn(self, trial: optuna.Trial, input_shape: Tuple) -> keras.Model:
        """Build traditional feedforward neural network"""
        # Number of hidden layers
        n_layers = trial.suggest_int('n_layers', 2, 6)
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape[1:]))
        
        # Hidden layers
        for i in range(n_layers):
            n_units = trial.suggest_int(f'n_units_l{i}', 32, 512, step=32)
            activation = trial.suggest_categorical(f'activation_l{i}', ['relu', 'elu', 'selu'])
            
            model.add(keras.layers.Dense(n_units, activation=activation))
            
            # Batch normalization
            if trial.suggest_categorical(f'batch_norm_l{i}', [True, False]):
                model.add(keras.layers.BatchNormalization())
            
            # Dropout
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(1))
        
        return model
    
    def tune(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2) -> Dict:
        """
        Run hyperparameter tuning
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type} model")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5
            ),
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create objective with fixed data
        def objective_with_data(trial):
            return self.objective(trial, X_train, y_train, X_val, y_val)
        
        # Run optimization
        study.optimize(
            objective_with_data,
            n_trials=self.n_trials,
            callbacks=[self._optuna_callback]
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save results
        results = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial': i,
                    'value': trial.value,
                    'params': trial.params
                }
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }
        
        return results
    
    def _optuna_callback(self, study: optuna.Study, trial: optuna.FrozenTrial):
        """Callback for Optuna optimization"""
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: {trial.value:.4f}")
    
    def tune_ensemble(self, X: np.ndarray, y: np.ndarray,
                     model_types: List[str] = ['transformer', 'lstm', 'hybrid']) -> Dict:
        """
        Tune multiple models for ensemble
        """
        ensemble_results = {}
        
        for model_type in model_types:
            logger.info(f"Tuning {model_type} for ensemble")
            self.model_type = model_type
            results = self.tune(X, y)
            ensemble_results[model_type] = results
        
        # Find best ensemble weights
        ensemble_weights = self._optimize_ensemble_weights(X, y, ensemble_results)
        
        return {
            'individual_models': ensemble_results,
            'ensemble_weights': ensemble_weights
        }
    
    def _optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray,
                                  model_results: Dict) -> Dict[str, float]:
        """
        Optimize ensemble weights using Bayesian optimization
        """
        def ensemble_objective(trial):
            # Suggest weights for each model
            weights = {}
            weight_sum = 0
            
            for model_type in model_results.keys():
                weight = trial.suggest_float(f'weight_{model_type}', 0, 1)
                weights[model_type] = weight
                weight_sum += weight
            
            # Normalize weights
            weights = {k: v/weight_sum for k, v in weights.items()}
            
            # Calculate ensemble performance (simplified)
            ensemble_score = sum(
                weights[model_type] * model_results[model_type]['best_score']
                for model_type in weights
            )
            
            return ensemble_score
        
        # Create study for ensemble optimization
        ensemble_study = optuna.create_study(direction='minimize')
        ensemble_study.optimize(ensemble_objective, n_trials=50)
        
        # Get normalized weights
        best_weights = ensemble_study.best_params
        weight_sum = sum(best_weights.values())
        normalized_weights = {
            k.replace('weight_', ''): v/weight_sum 
            for k, v in best_weights.items()
        }
        
        return normalized_weights
    
    def save_results(self, results: Dict, filepath: str):
        """Save tuning results to file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Tuning results saved to {filepath}")
    
    def load_and_build_best_model(self, filepath: str, input_shape: Tuple) -> keras.Model:
        """Load best parameters and build model"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.model_type = results['model_type']
        
        # Create a mock trial with best parameters
        study = optuna.create_study()
        trial = optuna.trial.FixedTrial(self.best_params)
        
        # Build model with best parameters
        model = self.model_builders[self.model_type](trial, input_shape)
        
        # Compile with best optimizer
        optimizer = self._suggest_optimizer(trial)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model


def automated_hyperparameter_search(X: np.ndarray, y: np.ndarray,
                                   model_types: List[str] = None,
                                   n_trials: int = 100) -> Dict:
    """
    Convenience function for automated hyperparameter search
    """
    if model_types is None:
        model_types = ['transformer', 'lstm', 'cnn', 'hybrid', 'traditional']
    
    all_results = {}
    
    for model_type in model_types:
        tuner = FantasyModelTuner(
            model_type=model_type,
            n_trials=n_trials
        )
        
        results = tuner.tune(X, y)
        all_results[model_type] = results
        
        # Save individual results
        tuner.save_results(
            results,
            f'hyperparameter_results_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    # Find best overall model
    best_model = min(all_results.items(), key=lambda x: x[1]['best_score'])
    logger.info(f"Best model: {best_model[0]} with score: {best_model[1]['best_score']:.4f}")
    
    return all_results