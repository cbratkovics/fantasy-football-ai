#!/usr/bin/env python3
"""
Train ML models on real fantasy football data
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment
os.environ["DATABASE_URL"] = "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres"

from sqlalchemy import create_engine, text
from backend.models.database import Player, PlayerStats

class RealDataTrainer:
    def __init__(self):
        self.engine = create_engine(os.environ["DATABASE_URL"])
        
    def fetch_and_store_data(self):
        """Fetch data from Sleeper API if needed"""
        print("📊 Phase 1: Data Collection")
        print("="*60)
        
        with self.engine.connect() as conn:
            # Check current data
            player_count = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
            stats_count = conn.execute(text("SELECT COUNT(*) FROM player_stats")).scalar()
            
            print(f"Current database status:")
            print(f"  Players: {player_count}")
            print(f"  Stats: {stats_count}")
            
            if stats_count < 1000:
                print("\nFetching additional stats data...")
                self._fetch_stats_data()
            else:
                print("✅ Sufficient data available")
                
    def _fetch_stats_data(self):
        """Fetch stats from Sleeper API"""
        stored = 0
        for week in range(1, 6):  # First 5 weeks of 2023
            try:
                url = f"https://api.sleeper.app/v1/stats/nfl/regular/2023/{week}"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    stats = response.json()
                    print(f"  Week {week}: {len(stats)} player stats")
                    
                    # Store in database
                    with self.engine.begin() as conn:
                        for player_id, stat_data in stats.items():
                            if stat_data.get('pts_ppr', 0) > 0:
                                # Check if player exists
                                player_exists = conn.execute(
                                    text("SELECT 1 FROM players WHERE player_id = :pid"),
                                    {"pid": player_id}
                                ).scalar()
                                
                                if player_exists:
                                    # Check if stats exist
                                    stats_exist = conn.execute(
                                        text("""
                                            SELECT 1 FROM player_stats 
                                            WHERE player_id = :pid 
                                            AND season = 2023 
                                            AND week = :week
                                        """),
                                        {"pid": player_id, "week": week}
                                    ).scalar()
                                    
                                    if not stats_exist:
                                        conn.execute(
                                            text("""
                                                INSERT INTO player_stats 
                                                (player_id, season, week, stats, 
                                                 fantasy_points_std, fantasy_points_ppr, fantasy_points_half)
                                                VALUES (:pid, 2023, :week, :stats::jsonb, :std, :ppr, :half)
                                            """),
                                            {
                                                "pid": player_id,
                                                "week": week,
                                                "stats": str(stat_data).replace("'", '"'),
                                                "std": stat_data.get('pts_std', 0),
                                                "ppr": stat_data.get('pts_ppr', 0),
                                                "half": stat_data.get('pts_half_ppr', 0)
                                            }
                                        )
                                        stored += 1
                                        
            except Exception as e:
                print(f"  Error fetching week {week}: {str(e)}")
                
        print(f"✅ Stored {stored} new stat records")
        
    def prepare_training_data(self):
        """Load and prepare data for training"""
        print("\n🔧 Phase 2: Data Preparation")
        print("="*60)
        
        # Load data with engineered features
        query = """
            WITH player_averages AS (
                SELECT 
                    player_id,
                    AVG(fantasy_points_ppr) as avg_points,
                    COUNT(*) as games_played
                FROM player_stats
                WHERE season = 2023
                GROUP BY player_id
            )
            SELECT 
                p.position,
                ps.player_id,
                ps.season,
                ps.week,
                ps.fantasy_points_ppr as target,
                COALESCE((ps.stats->>'pass_yd')::float, 0) as pass_yards,
                COALESCE((ps.stats->>'pass_td')::float, 0) as pass_tds,
                COALESCE((ps.stats->>'rush_yd')::float, 0) as rush_yards,
                COALESCE((ps.stats->>'rush_td')::float, 0) as rush_tds,
                COALESCE((ps.stats->>'rec')::float, 0) as receptions,
                COALESCE((ps.stats->>'rec_yd')::float, 0) as rec_yards,
                COALESCE((ps.stats->>'rec_td')::float, 0) as rec_tds,
                pa.avg_points as player_avg_points,
                pa.games_played
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            LEFT JOIN player_averages pa ON ps.player_id = pa.player_id
            WHERE ps.season = 2023
            AND p.position IN ('QB', 'RB', 'WR', 'TE')
            AND ps.fantasy_points_ppr > 0
        """
        
        df = pd.read_sql(query, self.engine)
        print(f"✅ Loaded {len(df)} samples")
        
        # Create additional features
        df['total_tds'] = df['pass_tds'] + df['rush_tds'] + df['rec_tds']
        df['total_yards'] = df['pass_yards'] + df['rush_yards'] + df['rec_yards']
        
        # Position encoding
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        print("\nData distribution:")
        print(df['position'].value_counts())
        
        return df
        
    def train_models(self, df):
        """Train multiple models and ensemble"""
        print("\n🤖 Phase 3: Model Training")
        print("="*60)
        
        # Prepare features
        feature_cols = [
            'pass_yards', 'pass_tds', 'rush_yards', 'rush_tds',
            'receptions', 'rec_yards', 'rec_tds', 'total_tds',
            'total_yards', 'player_avg_points', 'games_played',
            'is_qb', 'is_rb', 'is_wr', 'is_te'
        ]
        
        X = df[feature_cols]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model': model
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.3f}")
            
        # Ensemble predictions
        print("\nCreating ensemble...")
        ensemble_pred = np.mean([predictions[m] for m in predictions], axis=0)
        
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Ensemble Performance:")
        print(f"  MAE: {ensemble_mae:.2f}")
        print(f"  RMSE: {ensemble_rmse:.2f}")
        print(f"  R²: {ensemble_r2:.3f}")
        
        # Save best model
        best_model = models['random_forest']
        joblib.dump(best_model, 'fantasy_model_trained.pkl')
        print("\n✅ Model saved to fantasy_model_trained.pkl")
        
        return results, X_test, y_test, ensemble_pred
        
    def evaluate_results(self, results, X_test, y_test, ensemble_pred):
        """Detailed evaluation of results"""
        print("\n📊 Phase 4: Detailed Evaluation")
        print("="*60)
        
        # Performance summary
        print("\nModel Comparison:")
        print("-"*40)
        print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-"*40)
        
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} {metrics['r2']:<10.3f}")
            
        print(f"{'Ensemble':<15} {mean_absolute_error(y_test, ensemble_pred):<10.2f} "
              f"{np.sqrt(mean_squared_error(y_test, ensemble_pred)):<10.2f} "
              f"{r2_score(y_test, ensemble_pred):<10.3f}")
        
        # Error analysis
        errors = np.abs(ensemble_pred - y_test)
        
        print("\n\nError Distribution:")
        print(f"  Mean error: {errors.mean():.2f} points")
        print(f"  Median error: {np.median(errors):.2f} points")
        print(f"  90th percentile error: {np.percentile(errors, 90):.2f} points")
        
        # Feature importance (Random Forest)
        rf_model = results['random_forest']['model']
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n\nTop Feature Importances:")
        print("-"*40)
        for _, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:<20} {row['importance']:.3f}")
            
        # Save evaluation report
        report = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'ensemble_performance': {
                'mae': float(mean_absolute_error(y_test, ensemble_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
                'r2': float(r2_score(y_test, ensemble_pred))
            },
            'sample_size': len(y_test),
            'feature_importance': feature_importance.to_dict()
        }
        
        import json
        with open('model_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\n✅ Evaluation report saved to model_evaluation_report.json")
        
    def run(self):
        """Run complete training pipeline"""
        print("🚀 FANTASY FOOTBALL ML TRAINING PIPELINE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Fetch data if needed
        self.fetch_and_store_data()
        
        # Prepare data
        df = self.prepare_training_data()
        
        # Train models
        results, X_test, y_test, ensemble_pred = self.train_models(df)
        
        # Evaluate
        self.evaluate_results(results, X_test, y_test, ensemble_pred)
        
        print("\n" + "="*80)
        print(f"✅ Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    trainer = RealDataTrainer()
    trainer.run()