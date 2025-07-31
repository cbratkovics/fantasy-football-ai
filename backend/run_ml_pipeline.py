#!/usr/bin/env python3
"""
Complete ML Pipeline: Data Collection, Training, and Evaluation
"""

import os
import sys
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "postgresql://postgres:-Pv95h_SjeXf%21Dt@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models.database import Player, PlayerStats
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.ensemble_predictions import EnsemblePredictor
from backend.ml.predictions import MLPredictor

class MLPipeline:
    def __init__(self):
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.Session = sessionmaker(bind=self.engine)
        self.feature_engineer = FeatureEngineer()
        self.ensemble_predictor = EnsemblePredictor()
        
    def collect_player_data(self):
        """Fetch and store player data from Sleeper API"""
        print("\n📊 PHASE 1: Data Collection")
        print("="*60)
        
        db = self.Session()
        try:
            # Check existing data
            player_count = db.query(Player).count()
            print(f"Current players in database: {player_count}")
            
            if player_count < 1000:
                print("Fetching player data from Sleeper API...")
                response = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60)
                
                if response.status_code == 200:
                    players_data = response.json()
                    print(f"✅ Fetched {len(players_data)} players")
                    
                    # Store relevant players
                    stored = 0
                    for player_id, player_info in players_data.items():
                        if player_info.get('position') in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                            existing = db.query(Player).filter(Player.player_id == player_id).first()
                            if not existing and player_info.get('first_name'):
                                player = Player(
                                    player_id=player_id,
                                    first_name=player_info.get('first_name', ''),
                                    last_name=player_info.get('last_name', ''),
                                    position=player_info.get('position'),
                                    team=player_info.get('team'),
                                    fantasy_positions=player_info.get('fantasy_positions', []),
                                    age=player_info.get('age'),
                                    years_exp=player_info.get('years_exp'),
                                    status=player_info.get('status', 'Unknown'),
                                    meta_data=player_info
                                )
                                db.add(player)
                                stored += 1
                                
                                if stored % 100 == 0:
                                    db.commit()
                                    print(f"  Stored {stored} players...")
                    
                    db.commit()
                    print(f"✅ Stored {stored} new players")
            else:
                print("✅ Sufficient player data already exists")
                
        except Exception as e:
            print(f"❌ Error collecting player data: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
    def collect_stats_data(self):
        """Fetch and store player stats for recent seasons"""
        print("\n📈 Collecting Player Stats")
        print("-"*60)
        
        db = self.Session()
        seasons_weeks = [
            (2023, range(1, 18)),  # 2023 season
            (2022, range(1, 18)),  # 2022 season
        ]
        
        try:
            total_stored = 0
            for season, weeks in seasons_weeks:
                for week in weeks:
                    # Check if data exists
                    existing_count = db.query(PlayerStats).filter(
                        PlayerStats.season == season,
                        PlayerStats.week == week
                    ).count()
                    
                    if existing_count < 100:  # Fetch if we have less than 100 records
                        print(f"Fetching stats for {season} Week {week}...")
                        url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            stats_data = response.json()
                            stored = 0
                            
                            for player_id, stats in stats_data.items():
                                if any([stats.get('pts_std', 0), stats.get('pts_ppr', 0)]):
                                    # Check if player exists
                                    player = db.query(Player).filter(Player.player_id == player_id).first()
                                    if player:
                                        # Check if stats exist
                                        existing = db.query(PlayerStats).filter(
                                            PlayerStats.player_id == player_id,
                                            PlayerStats.season == season,
                                            PlayerStats.week == week
                                        ).first()
                                        
                                        if not existing:
                                            player_stats = PlayerStats(
                                                player_id=player_id,
                                                season=season,
                                                week=week,
                                                stats=stats,
                                                fantasy_points_std=stats.get('pts_std', 0),
                                                fantasy_points_ppr=stats.get('pts_ppr', 0),
                                                fantasy_points_half=stats.get('pts_half_ppr', 0),
                                                opponent=stats.get('opponent'),
                                                is_home=stats.get('home') == 1 if 'home' in stats else None
                                            )
                                            db.add(player_stats)
                                            stored += 1
                                            
                            if stored > 0:
                                db.commit()
                                total_stored += stored
                                print(f"  ✅ Stored {stored} stat records")
                    else:
                        print(f"  ✓ {season} Week {week}: Data already exists ({existing_count} records)")
                        
            print(f"\n✅ Total new stat records stored: {total_stored}")
            
            # Show data summary
            total_stats = db.query(PlayerStats).count()
            print(f"Total stat records in database: {total_stats}")
            
        except Exception as e:
            print(f"❌ Error collecting stats: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
    def prepare_training_data(self):
        """Prepare data for ML training"""
        print("\n🔧 PHASE 2: Data Preparation")
        print("="*60)
        
        db = self.Session()
        try:
            # Get training data
            query = text("""
                SELECT 
                    p.player_id,
                    p.position,
                    p.team,
                    ps.season,
                    ps.week,
                    ps.stats,
                    ps.fantasy_points_ppr,
                    ps.opponent,
                    ps.is_home
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE ps.season IN (2022, 2023)
                AND p.position IN ('QB', 'RB', 'WR', 'TE')
                AND ps.fantasy_points_ppr > 0
                ORDER BY ps.season, ps.week
            """)
            
            result = db.execute(query)
            data = []
            for row in result:
                data.append({
                    'player_id': row.player_id,
                    'position': row.position,
                    'team': row.team,
                    'season': row.season,
                    'week': row.week,
                    'stats': row.stats,
                    'fantasy_points': row.fantasy_points_ppr,
                    'opponent': row.opponent,
                    'is_home': row.is_home
                })
                
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df)} training samples")
            print(f"\nData by position:")
            print(df['position'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"❌ Error preparing data: {str(e)}")
            return None
        finally:
            db.close()
            
    def train_models(self, df):
        """Train ML models on the data"""
        print("\n🤖 PHASE 3: Model Training")
        print("="*60)
        
        if df is None or len(df) == 0:
            print("❌ No data available for training")
            return None
            
        try:
            # Engineer features
            print("Engineering features...")
            features_df = self.feature_engineer.engineer_features(df)
            print(f"✅ Created {len(features_df.columns)} features")
            
            # Prepare for training
            feature_cols = [col for col in features_df.columns if col not in 
                          ['player_id', 'season', 'week', 'fantasy_points', 'position']]
            
            X = features_df[feature_cols]
            y = features_df['fantasy_points']
            
            # Split by season (2022 for training, 2023 for testing)
            train_mask = features_df['season'] == 2022
            test_mask = features_df['season'] == 2023
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            print(f"\nTraining set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Train ensemble model
            print("\nTraining ensemble model...")
            self.ensemble_predictor.train(X_train, y_train)
            
            # Make predictions
            predictions = self.ensemble_predictor.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            results = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions,
                'actuals': y_test.values,
                'test_data': features_df[test_mask]
            }
            
            print(f"\n📊 Model Performance:")
            print(f"  MAE: {mae:.2f} points")
            print(f"  RMSE: {rmse:.2f} points")
            print(f"  R²: {r2:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def evaluate_by_position(self, results):
        """Evaluate model performance by position"""
        print("\n📋 PHASE 4: Detailed Evaluation")
        print("="*60)
        
        if results is None:
            print("❌ No results to evaluate")
            return
            
        test_data = results['test_data']
        test_data['predictions'] = results['predictions']
        test_data['errors'] = np.abs(test_data['predictions'] - test_data['fantasy_points'])
        
        print("\nPerformance by Position:")
        print("-"*40)
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = test_data[test_data['position'] == position]
            if len(pos_data) > 0:
                mae = pos_data['errors'].mean()
                r2 = 1 - (pos_data['errors']**2).sum() / ((pos_data['fantasy_points'] - pos_data['fantasy_points'].mean())**2).sum()
                print(f"\n{position}:")
                print(f"  Samples: {len(pos_data)}")
                print(f"  MAE: {mae:.2f} points")
                print(f"  R²: {r2:.3f}")
                print(f"  Avg Actual: {pos_data['fantasy_points'].mean():.1f} points")
                print(f"  Avg Predicted: {pos_data['predictions'].mean():.1f} points")
                
        # Top predictions vs actuals
        print("\n\nTop 10 Performances (Actual vs Predicted):")
        print("-"*60)
        top_performances = test_data.nlargest(10, 'fantasy_points')[['position', 'fantasy_points', 'predictions', 'errors']]
        for idx, row in top_performances.iterrows():
            print(f"{row['position']}: Actual {row['fantasy_points']:.1f}, Predicted {row['predictions']:.1f}, Error {row['errors']:.1f}")
            
        # Save model
        print("\n💾 Saving trained model...")
        self.ensemble_predictor.save_model('ensemble_model_real_data.pkl')
        print("✅ Model saved successfully")
        
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("\n🚀 FANTASY FOOTBALL ML PIPELINE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Phase 1: Data Collection
        self.collect_player_data()
        self.collect_stats_data()
        
        # Phase 2: Data Preparation
        df = self.prepare_training_data()
        
        # Phase 3: Model Training
        results = self.train_models(df)
        
        # Phase 4: Evaluation
        self.evaluate_by_position(results)
        
        print("\n" + "="*80)
        print(f"✅ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run_pipeline()