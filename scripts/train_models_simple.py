#!/usr/bin/env python3
"""
Simple ML model training script to get basic models working
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.database import create_engine, Player, PlayerStats
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football")


def train_simple_models():
    """Train simple RandomForest models for each position"""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K']
    results = {}
    
    with SessionLocal() as db:
        for position in positions:
            logger.info(f"Training model for {position}")
            
            # Get player stats
            query = db.query(
                PlayerStats.player_id,
                PlayerStats.season,
                PlayerStats.week,
                PlayerStats.fantasy_points_ppr,
                PlayerStats.stats,
                Player.age,
                Player.years_exp
            ).join(
                Player, Player.player_id == PlayerStats.player_id
            ).filter(
                Player.position == position,
                PlayerStats.fantasy_points_ppr > 0
            )
            
            df = pd.read_sql(query.statement, db.bind)
            
            if len(df) < 100:
                logger.warning(f"Insufficient data for {position}")
                continue
            
            # Extract basic features from stats
            stats_features = pd.json_normalize(df['stats'])
            
            # Select features based on position
            if position == 'QB':
                feature_cols = ['pass_att', 'pass_cmp', 'pass_yd', 'pass_td', 'pass_int', 'rush_att', 'rush_yd']
            elif position in ['RB', 'WR', 'TE']:
                feature_cols = ['rush_att', 'rush_yd', 'rec', 'rec_yd', 'rec_td', 'rec_tgt']
            elif position == 'K':
                feature_cols = ['fgm', 'fga', 'xpm', 'xpa']
            else:
                feature_cols = []
            
            # Get available features
            available_features = [col for col in feature_cols if col in stats_features.columns]
            
            if not available_features:
                logger.warning(f"No features available for {position}")
                continue
            
            # Prepare features
            X = pd.concat([
                df[['season', 'week', 'age', 'years_exp']],
                stats_features[available_features]
            ], axis=1).fillna(0)
            
            y = df['fantasy_points_ppr'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train RandomForest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            logger.info(f"{position} Model - MAE: {mae:.2f}, R²: {r2:.3f}")
            
            # Save model and scaler
            model_path = models_dir / f'rf_model_{position}.pkl'
            scaler_path = models_dir / f'rf_scaler_{position}.pkl'
            feature_path = models_dir / f'rf_features_{position}.pkl'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(X.columns.tolist(), feature_path)
            
            results[position] = {
                'mae': mae,
                'r2': r2,
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(models_dir / 'training_results_simple.csv')
    logger.info(f"Training complete! Results:\n{results_df}")
    
    return results


if __name__ == "__main__":
    train_simple_models()