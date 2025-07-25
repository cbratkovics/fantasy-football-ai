"""
Fantasy Football AI - CLI ML Integration
Integrates ML models with existing CLI system for production use.
"""


import click
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List
import asyncio
from datetime import datetime
import json

# Import the ML system
from ..models.ml_integration import FantasyFootballAI

logger = logging.getLogger(__name__)

@click.group()
def ml():
    """Machine Learning model commands for Fantasy Football AI."""
    pass

@ml.command()
@click.option('--seasons', default='2022,2023,2024', help='Comma-separated seasons to train on')
@click.option('--epochs', default=100, help='Neural network training epochs')
@click.option('--validation-split', default=0.2, help='Validation data fraction')
@click.option('--model-dir', default='models/', help='Directory to save models')
@click.option('--force', is_flag=True, help='Overwrite existing models')
@click.option('--fantasy-scoring', default='standard', help='Fantasy scoring system (standard/ppr/half_ppr)')
def train(seasons: str, epochs: int, validation_split: float, model_dir: str, force: bool, fantasy_scoring: str):
    """
    Train the complete ML system on historical NFL data.
    
    Example:
        python src/fantasy_ai/cli/main.py ml train --seasons 2022,2023,2024 --epochs 100
    """
    click.echo("Starting Fantasy Football AI Training Pipeline")
    click.echo("="*60)
    
    try:
        # Parse seasons
        season_list = [int(s.strip()) for s in seasons.split(',')]
        click.echo(f"Training on seasons: {season_list}")
        click.echo(f"Fantasy scoring: {fantasy_scoring}")
        
        # Check if models exist
        model_path = Path(model_dir)
        if model_path.exists() and any(model_path.iterdir()) and not force:
            if not click.confirm(f"Models exist in {model_dir}. Overwrite?"):
                click.echo("Training cancelled.")
                return
        
        # Load training data from database
        click.echo("Loading training data from database...")
        training_data = _load_training_data(season_list, fantasy_scoring)
        
        if training_data.empty:
            click.echo("No training data found. Run data collection first:")
            click.echo("  python src/fantasy_ai/cli/main.py collect quick")
            return
        
        click.echo(f"Loaded {len(training_data)} player-week records")
        
        # Show data distribution
        pos_counts = training_data['position'].value_counts()
        click.echo(f"Position distribution: {pos_counts.to_dict()}")
        
        # Initialize and train AI system
        click.echo("Initializing AI system...")
        ai_system = FantasyFootballAI(model_dir=model_dir)
        
        click.echo("Starting training process...")
        with click.progressbar(length=100, label='Training progress') as bar:
            training_results = ai_system.train_system(
                training_data, 
                validation_split=validation_split,
                epochs=epochs
            )
            bar.update(100)
        
        # Display results
        click.echo("\nTraining Complete!")
        click.echo("="*40)
        
        perf = training_results['system_performance']
        click.echo(f"System Accuracy: {perf['accuracy_percentage']:.1f}%")
        click.echo(f"Prediction MAE: {perf['prediction_mae']:.3f}")
        click.echo(f"Training Samples: {perf['n_samples']:,}")
        
        # Save models
        model_save_path = ai_system.save_models()
        click.echo(f"Models saved to: {model_save_path}")
        
        # Accuracy check
        if perf['accuracy_percentage'] >= 89.0:
            click.echo("TARGET ACCURACY ACHIEVED!")
        else:
            click.echo("Consider more training data or hyperparameter tuning")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Training failed: {e}")
        raise

@ml.command()
@click.option('--model-path', required=True, help='Path to trained models')
@click.option('--season', default=2024, help='Season to predict for')
@click.option('--week', help='Specific week to predict (optional)')
@click.option('--position', help='Filter by position (QB/RB/WR/TE)')
@click.option('--output', default='predictions.csv', help='Output CSV file')
@click.option('--top-n', default=50, help='Number of top predictions to show')
@click.option('--fantasy-scoring', default='standard', help='Fantasy scoring system (standard/ppr/half_ppr)')
def predict(model_path: str, season: int, week: Optional[int], 
           position: Optional[str], output: str, top_n: int, fantasy_scoring: str):
    """
    Generate fantasy football predictions using trained models.
    
    Example:
        python src/fantasy_ai/cli/main.py ml predict --model-path models/fantasy_ai_v1.0.0_20241225 --season 2024
    """
    click.echo("Generating Fantasy Football Predictions")
    click.echo("="*50)
    
    try:
        # Load trained models
        click.echo(f"Loading models from: {model_path}")
        ai_system = FantasyFootballAI()
        ai_system.load_models(model_path)
        
        # Load current season data
        click.echo(f"Loading {season} season data...")
        prediction_data = _load_prediction_data(season, week, position, fantasy_scoring)
        
        if prediction_data.empty:
            click.echo("No data found for predictions. Run data collection first:")
            click.echo("  python src/fantasy_ai/cli/main.py collect quick")
            return
        
        click.echo(f"Loaded {len(prediction_data)} player records")
        
        # Generate predictions
        click.echo("Generating predictions...")
        with click.progressbar(prediction_data.iterrows(), label='Processing players') as bar:
            predictions = ai_system.predict(prediction_data)
            for _ in bar:
                pass
        
        # Get draft recommendations
        recommendations = ai_system.get_draft_recommendations(predictions, position)
        
        # Display top predictions
        click.echo(f"\nTop {min(top_n, len(recommendations))} Predictions:")
        click.echo("-" * 80)
        
        top_recs = recommendations.head(top_n)
        for _, player in top_recs.iterrows():
            confidence_icon = "HIGH" if player['confidence'] > 0.8 else "MED" if player['confidence'] > 0.6 else "LOW"
            click.echo(f"{confidence_icon:<4} {player['player_id']:<15} {player['position']:<3} "
                      f"Tier {player['tier']:<2} {player['predicted_points']:<6.1f}pts "
                      f"({player['confidence']:.2f} conf)")
        
        # Position breakdown
        if not position:
            click.echo(f"\nPosition Breakdown:")
            pos_summary = recommendations.groupby('position').agg({
                'predicted_points': ['mean', 'count'],
                'confidence': 'mean'
            }).round(2)
            click.echo(pos_summary.to_string())
        
        # Save to CSV
        recommendations.to_csv(output, index=False)
        click.echo(f"\nPredictions saved to: {output}")
        
        # Summary stats
        click.echo(f"\nSummary:")
        click.echo(f"Total predictions: {len(predictions)}")
        click.echo(f"Average confidence: {recommendations['confidence'].mean():.2f}")
        click.echo(f"High-confidence picks: {sum(recommendations['confidence'] > 0.8)}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"Prediction failed: {e}")
        raise

@ml.command()
@click.option('--model-path', required=True, help='Path to trained models')
@click.option('--test-season', default=2024, help='Season to test on')
@click.option('--week-start', default=1, help='Starting week for evaluation')
@click.option('--week-end', default=17, help='Ending week for evaluation')
@click.option('--fantasy-scoring', default='standard', help='Fantasy scoring system (standard/ppr/half_ppr)')
def evaluate(model_path: str, test_season: int, week_start: int, week_end: int, fantasy_scoring: str):
    """
    Evaluate model performance on test data.
    
    Example:
        python src/fantasy_ai/cli/main.py ml evaluate --model-path models/latest --test-season 2024
    """
    click.echo("Evaluating Model Performance")
    click.echo("="*40)
    
    try:
        # Load models
        ai_system = FantasyFootballAI()
        ai_system.load_models(model_path)
        
        # Load test data
        click.echo(f"Loading test data for {test_season} weeks {week_start}-{week_end}...")
        test_data = _load_evaluation_data(test_season, week_start, week_end, fantasy_scoring)
        
        if test_data.empty:
            click.echo("No test data found.")
            return
        
        # Generate predictions
        click.echo("Generating predictions for evaluation...")
        predictions = ai_system.predict(test_data)
        
        # Calculate accuracy metrics
        click.echo("Calculating accuracy metrics...")
        
        # Compare predictions to actual results
        actual_vs_predicted = []
        for pred in predictions:
            actual_row = test_data[
                (test_data['player_id'] == pred.player_id) &
                (test_data['week'] == pred.week) &
                (test_data['season'] == pred.season)
            ]
            
            if not actual_row.empty:
                actual_points = float(actual_row['fantasy_points'].iloc[0])
                actual_vs_predicted.append({
                    'actual': actual_points,
                    'predicted': pred.prediction.predicted_points,
                    'position': pred.position,
                    'player_id': pred.player_id
                })
        
        if not actual_vs_predicted:
            click.echo("No matching predictions found for evaluation.")
            return
        
        eval_df = pd.DataFrame(actual_vs_predicted)
        
        # Calculate metrics
        mae = np.mean(np.abs(eval_df['actual'] - eval_df['predicted']))
        rmse = np.sqrt(np.mean((eval_df['actual'] - eval_df['predicted']) ** 2))
        
        # Accuracy within acceptable ranges
        within_3 = np.sum(np.abs(eval_df['actual'] - eval_df['predicted']) <= 3) / len(eval_df)
        within_5 = np.sum(np.abs(eval_df['actual'] - eval_df['predicted']) <= 5) / len(eval_df)
        
        # Display results
        click.echo("\nEvaluation Results:")
        click.echo("-" * 30)
        click.echo(f"Test samples: {len(eval_df):,}")
        click.echo(f"Mean Absolute Error: {mae:.3f}")
        click.echo(f"Root Mean Square Error: {rmse:.3f}")
        click.echo(f"Accuracy within ±3 pts: {within_3:.1%}")
        click.echo(f"Accuracy within ±5 pts: {within_5:.1%}")
        
        # Position-specific metrics
        click.echo(f"\nPosition-Specific Performance:")
        pos_metrics = eval_df.groupby('position').apply(
            lambda x: pd.Series({
                'MAE': np.mean(np.abs(x['actual'] - x['predicted'])),
                'Count': len(x)
            })
        ).round(3)
        click.echo(pos_metrics.to_string())
        
        # Performance assessment
        if mae <= 0.45:
            click.echo("\nEXCELLENT: Target accuracy achieved!")
        elif mae <= 1.0:
            click.echo("\nGOOD: Performance within acceptable range")
        else:
            click.echo("\nNEEDS IMPROVEMENT: Consider retraining with more data")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"Evaluation failed: {e}")
        raise

@ml.command()
@click.option('--model-path', help='Path to specific model version')
def status(model_path: Optional[str]):
    """
    Show ML system status and model information.
    
    Example:
        python src/fantasy_ai/cli/main.py ml status
    """
    click.echo("Fantasy Football AI System Status")
    click.echo("="*45)
    
    try:
        if model_path:
            # Show specific model info
            ai_system = FantasyFootballAI()
            ai_system.load_models(model_path)
            summary = ai_system.get_system_summary()
            
            click.echo(f"Model Path: {model_path}")
            click.echo(f"Status: {summary['status']}")
            click.echo(f"Version: {summary['model_version']}")
            
            # Component details
            click.echo(f"\nComponents:")
            fe = summary['components']['feature_engineering']
            click.echo(f"  Features: {fe['feature_count']} ({', '.join(fe['feature_names'])})")
            
            gmm = summary['components']['gmm_clustering']
            click.echo(f"  GMM Models: {len(gmm['positions_trained'])} positions")
            
            nn = summary['components']['neural_network']
            click.echo(f"  Neural Network: {nn['total_parameters']:,} parameters")
            
            # Performance
            if 'performance_metrics' in summary and summary['performance_metrics']:
                perf = summary['performance_metrics']
                click.echo(f"\nPerformance:")
                click.echo(f"  Accuracy: {perf['accuracy_percentage']:.1f}%")
                click.echo(f"  MAE: {perf['prediction_mae']:.3f}")
                
        else:
            # Show database status and available models
            db_status = asyncio.run(check_database_status())
            click.echo(f"Database Status: {db_status}")
            
            models_dir = Path("models/")
            if models_dir.exists():
                model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                
                if model_dirs:
                    click.echo(f"\nAvailable Models ({len(model_dirs)}):")
                    for model_dir in sorted(model_dirs, reverse=True):
                        metadata_file = model_dir / "system_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                            
                            version = metadata.get('model_version', 'unknown')
                            training_date = metadata.get('training_date', 'unknown')
                            click.echo(f"  {model_dir.name:<30} v{version:<8} {training_date}")
                        else:
                            click.echo(f"  {model_dir.name:<30} (incomplete)")
                else:
                    click.echo("\nNo trained models found.")
            else:
                click.echo("\nModels directory does not exist.")
                
            # System requirements check
            click.echo(f"\nSystem Check:")
            
            # Check required Python packages
            required_packages = ['pandas', 'numpy', 'scikit-learn', 'tensorflow']
            for package in required_packages:
                try:
                    __import__(package)
                    status_icon = "OK"
                except ImportError:
                    status_icon = "MISSING"
                click.echo(f"  {package}: {status_icon}")
                
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"Status check failed: {e}")

# Database integration functions - Updated with your actual schema

def _load_training_data(seasons: List[int], fantasy_scoring: str = 'standard') -> pd.DataFrame:
    """Load training data from your database for ML model training."""
    async def load_data():
        logger.info(f"Loading training data for seasons: {seasons}, scoring: {fantasy_scoring}")
        
        try:
            from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
            from fantasy_ai.core.data.storage.models import Player, WeeklyStats
            from sqlalchemy import select, and_
            
            db_manager = get_simple_db_manager()
            
            # Select fantasy points column based on scoring system
            fantasy_points_col = {
                'standard': WeeklyStats.fantasy_points_standard,
                'ppr': WeeklyStats.fantasy_points_ppr,
                'half_ppr': WeeklyStats.fantasy_points_half_ppr
            }.get(fantasy_scoring, WeeklyStats.fantasy_points_standard)
            
            async with db_manager.get_session() as session:
                query = (
                    select(
                        Player.id.label('player_id'),
                        Player.name.label('player_name'),
                        Player.position,
                        WeeklyStats.week,
                        WeeklyStats.season,
                        fantasy_points_col.label('fantasy_points'),
                        # Additional stats for feature engineering
                        WeeklyStats.passing_yards,
                        WeeklyStats.passing_touchdowns,
                        WeeklyStats.interceptions,
                        WeeklyStats.rushing_yards,
                        WeeklyStats.rushing_touchdowns,
                        WeeklyStats.receiving_yards,
                        WeeklyStats.receiving_touchdowns,
                        WeeklyStats.receptions,
                        WeeklyStats.receiving_targets,
                        WeeklyStats.fumbles,
                        WeeklyStats.fumbles_lost
                    )
                    .select_from(Player)
                    .join(WeeklyStats, Player.id == WeeklyStats.player_id)
                    .where(
                        and_(
                            WeeklyStats.season.in_(seasons),
                            fantasy_points_col.is_not(None),
                            fantasy_points_col >= 0,
                            Player.position.in_(['QB', 'RB', 'WR', 'TE'])
                        )
                    )
                    .order_by(Player.id, WeeklyStats.season, WeeklyStats.week)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'player_id': str(row.player_id),
                        'player_name': row.player_name,
                        'position': row.position,
                        'week': int(row.week),
                        'season': int(row.season),
                        'fantasy_points': float(row.fantasy_points) if row.fantasy_points else 0.0,
                        # Additional stats
                        'passing_yards': float(row.passing_yards) if row.passing_yards else 0,
                        'passing_touchdowns': int(row.passing_touchdowns) if row.passing_touchdowns else 0,
                        'interceptions': int(row.interceptions) if row.interceptions else 0,
                        'rushing_yards': float(row.rushing_yards) if row.rushing_yards else 0,
                        'rushing_touchdowns': int(row.rushing_touchdowns) if row.rushing_touchdowns else 0,
                        'receiving_yards': float(row.receiving_yards) if row.receiving_yards else 0,
                        'receiving_touchdowns': int(row.receiving_touchdowns) if row.receiving_touchdowns else 0,
                        'receptions': int(row.receptions) if row.receptions else 0,
                        'receiving_targets': int(row.receiving_targets) if row.receiving_targets else 0,
                        'fumbles': int(row.fumbles) if row.fumbles else 0,
                        'fumbles_lost': int(row.fumbles_lost) if row.fumbles_lost else 0,
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} training records")
                
                if len(df) > 0:
                    # Data quality checks
                    missing_points = df['fantasy_points'].isna().sum()
                    if missing_points > 0:
                        logger.warning(f"Removing {missing_points} records with missing fantasy_points")
                        df = df.dropna(subset=['fantasy_points'])
                    
                    # Check position distribution
                    pos_counts = df['position'].value_counts()
                    logger.info(f"Position distribution: {pos_counts.to_dict()}")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return pd.DataFrame()
    
    return asyncio.run(load_data())

def _load_prediction_data(season: int, week: Optional[int] = None, 
                         position: Optional[str] = None, fantasy_scoring: str = 'standard') -> pd.DataFrame:
    """Load current season data for generating predictions."""
    async def load_data():
        logger.info(f"Loading prediction data: season={season}, week={week}, position={position}")
        
        try:
            from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
            from fantasy_ai.core.data.storage.models import Player, WeeklyStats
            from sqlalchemy import select, and_
            
            db_manager = get_simple_db_manager()
            
            # Select fantasy points column based on scoring system
            fantasy_points_col = {
                'standard': WeeklyStats.fantasy_points_standard,
                'ppr': WeeklyStats.fantasy_points_ppr,
                'half_ppr': WeeklyStats.fantasy_points_half_ppr
            }.get(fantasy_scoring, WeeklyStats.fantasy_points_standard)
            
            async with db_manager.get_session() as session:
                query = (
                    select(
                        Player.id.label('player_id'),
                        Player.name.label('player_name'),
                        Player.position,
                        WeeklyStats.week,
                        WeeklyStats.season,
                        fantasy_points_col.label('fantasy_points')
                    )
                    .select_from(Player)
                    .join(WeeklyStats, Player.id == WeeklyStats.player_id)
                    .where(
                        and_(
                            WeeklyStats.season == season,
                            Player.position.in_(['QB', 'RB', 'WR', 'TE'])
                        )
                    )
                )
                
                # Add optional filters
                if week is not None:
                    query = query.where(WeeklyStats.week == week)
                
                if position is not None:
                    query = query.where(Player.position == position)
                
                query = query.order_by(Player.id, WeeklyStats.week)
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'player_id': str(row.player_id),
                        'player_name': row.player_name,
                        'position': row.position,
                        'week': int(row.week),
                        'season': int(row.season),
                        'fantasy_points': float(row.fantasy_points) if row.fantasy_points else 0.0,
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} prediction records")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load prediction data: {e}")
            return pd.DataFrame()
    
    return asyncio.run(load_data())

def _load_evaluation_data(season: int, week_start: int, week_end: int, fantasy_scoring: str = 'standard') -> pd.DataFrame:
    """Load evaluation data for model performance testing."""
    async def load_data():
        logger.info(f"Loading evaluation data: season={season}, weeks {week_start}-{week_end}")
        
        try:
            from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
            from fantasy_ai.core.data.storage.models import Player, WeeklyStats
            from sqlalchemy import select, and_
            
            db_manager = get_simple_db_manager()
            
            # Select fantasy points column based on scoring system
            fantasy_points_col = {
                'standard': WeeklyStats.fantasy_points_standard,
                'ppr': WeeklyStats.fantasy_points_ppr,
                'half_ppr': WeeklyStats.fantasy_points_half_ppr
            }.get(fantasy_scoring, WeeklyStats.fantasy_points_standard)
            
            async with db_manager.get_session() as session:
                query = (
                    select(
                        Player.id.label('player_id'),
                        Player.name.label('player_name'),
                        Player.position,
                        WeeklyStats.week,
                        WeeklyStats.season,
                        fantasy_points_col.label('fantasy_points')
                    )
                    .select_from(Player)
                    .join(WeeklyStats, Player.id == WeeklyStats.player_id)
                    .where(
                        and_(
                            WeeklyStats.season == season,
                            WeeklyStats.week >= week_start,
                            WeeklyStats.week <= week_end,
                            fantasy_points_col.is_not(None),
                            Player.position.in_(['QB', 'RB', 'WR', 'TE'])
                        )
                    )
                    .order_by(Player.id, WeeklyStats.week)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'player_id': str(row.player_id),
                        'player_name': row.player_name,
                        'position': row.position,
                        'week': int(row.week),
                        'season': int(row.season),
                        'fantasy_points': float(row.fantasy_points),
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} evaluation records")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            return pd.DataFrame()
    
    return asyncio.run(load_data())

async def check_database_status() -> str:
    """Check database status and data availability."""
    try:
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        from fantasy_ai.core.data.storage.models import Player, WeeklyStats
        from sqlalchemy import select, func
        
        db_manager = get_simple_db_manager()
        
        async with db_manager.get_session() as session:
            # Get record counts
            result = await session.execute(select(func.count()).select_from(Player))
            player_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(WeeklyStats))
            stats_count = result.scalar()
            
            return f"Connected - {player_count} players, {stats_count} weekly stats"
            
    except Exception as e:
        return f"Error: {str(e)[:50]}..."