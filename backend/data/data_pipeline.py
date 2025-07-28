"""
Main Data Pipeline for Fantasy Football AI System
Orchestrates data fetching, processing, ML training, and predictions
Production-ready with error handling, logging, and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import schedule
import time
from dataclasses import dataclass
import json

# Import our modules (these would be actual imports in production)
# from sleeper_api_client import SleeperAPIClient, Player
# from database_models import DatabaseManager, Player as DBPlayer, PlayerStats, Prediction, DraftTier
# from fantasy_scoring import FantasyScorer, ScoringSettings
# from feature_engineering import FeatureEngineer, PlayerFeatures
# from gmm_clustering import GMMDraftOptimizer, DraftTier as GMMTier
# from neural_network_predictor import FantasyNeuralNetwork, PredictionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    database_url: str
    redis_host: str = 'localhost'
    redis_port: int = 6379
    
    # API settings
    sleeper_rate_limit: int = 900  # per minute
    
    # ML settings
    retrain_interval_days: int = 7
    min_samples_for_training: int = 100
    
    # Data settings
    seasons_to_fetch: List[int] = None
    current_season: int = 2024
    
    def __post_init__(self):
        if self.seasons_to_fetch is None:
            # Default: last 3 seasons
            self.seasons_to_fetch = [2022, 2023, 2024]


class FantasyFootballPipeline:
    """
    Main orchestration pipeline for the Fantasy Football AI system
    Handles data fetching, processing, ML training, and predictions
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        self.config = config
        
        # Initialize components
        self.db = DatabaseManager(config.database_url)
        self.sleeper_client = SleeperAPIClient(
            redis_host=config.redis_host,
            redis_port=config.redis_port
        )
        
        # ML components
        self.feature_engineer = FeatureEngineer()
        self.gmm_optimizer = GMMDraftOptimizer()
        self.neural_network = None  # Initialized during training
        
        # Scoring systems
        self.scorers = {
            'standard': FantasyScorer(ScoringSettings.standard()),
            'ppr': FantasyScorer(ScoringSettings.ppr()),
            'half_ppr': FantasyScorer(ScoringSettings.half_ppr())
        }
        
        # State tracking
        self.last_model_update = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize database and load existing models"""
        logger.info("Initializing Fantasy Football Pipeline")
        
        # Create database tables
        await self.db.create_tables()
        
        # Load existing models if available
        try:
            self._load_models()
            logger.info("Loaded existing models")
        except Exception as e:
            logger.info(f"No existing models found: {e}")
        
        self.is_initialized = True
        logger.info("Pipeline initialization complete")
    
    async def run_full_update(self):
        """Run complete data update and ML pipeline"""
        logger.info("Starting full pipeline update")
        
        try:
            # Step 1: Update player data
            await self._update_players()
            
            # Step 2: Fetch historical stats (if needed)
            await self._fetch_historical_stats()
            
            # Step 3: Calculate fantasy points
            await self._calculate_fantasy_points()
            
            # Step 4: Engineer features
            features_df = await self._engineer_features()
            
            # Step 5: Train/update ML models
            if self._should_retrain_models():
                await self._train_models(features_df)
            
            # Step 6: Generate predictions
            await self._generate_predictions()
            
            # Step 7: Update draft tiers
            await self._update_draft_tiers()
            
            logger.info("Full pipeline update complete")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise
    
    async def _update_players(self):
        """Fetch and update player data from Sleeper API"""
        logger.info("Updating player data")
        
        async with self.sleeper_client as client:
            # Get all players
            players = await client.get_all_players()
            
            # Update database
            async with self.db.async_session() as session:
                for player_id, player in players.items():
                    # Skip defenses and inactive players
                    if player.position == 'DEF' or player.status == 'Inactive':
                        continue
                    
                    # Check if player exists
                    result = await session.execute(
                        select(DBPlayer).where(DBPlayer.player_id == player_id)
                    )
                    db_player = result.scalar_one_or_none()
                    
                    if db_player:
                        # Update existing player
                        db_player.team = player.team
                        db_player.status = player.status
                        db_player.metadata = player.metadata
                    else:
                        # Create new player
                        db_player = DBPlayer(
                            player_id=player_id,
                            first_name=player.first_name,
                            last_name=player.last_name,
                            position=player.position,
                            team=player.team,
                            fantasy_positions=player.fantasy_positions,
                            age=player.age,
                            years_exp=player.years_exp,
                            status=player.status,
                            metadata=player.metadata
                        )
                        session.add(db_player)
                
                await session.commit()
        
        logger.info(f"Updated {len(players)} players")
    
    async def _fetch_historical_stats(self):
        """Fetch historical stats (mock implementation)"""
        logger.info("Fetching historical stats")
        
        # In production, this would fetch from a stats API
        # For now, we'll generate synthetic data for demonstration
        
        async with self.db.async_session() as session:
            # Get all active players
            result = await session.execute(
                select(DBPlayer).where(
                    and_(
                        DBPlayer.status == 'Active',
                        DBPlayer.position.in_(['QB', 'RB', 'WR', 'TE'])
                    )
                )
            )
            players = result.scalars().all()
            
            # Generate synthetic stats for each season/week
            for season in self.config.seasons_to_fetch:
                for week in range(1, 18):  # 17 week season
                    for player in players[:100]:  # Limit for demo
                        # Check if stats already exist
                        result = await session.execute(
                            select(PlayerStats).where(
                                and_(
                                    PlayerStats.player_id == player.player_id,
                                    PlayerStats.season == season,
                                    PlayerStats.week == week
                                )
                            )
                        )
                        if result.scalar_one_or_none():
                            continue
                        
                        # Generate synthetic stats based on position
                        stats = self._generate_synthetic_stats(
                            player.position,
                            player.player_id,
                            season,
                            week
                        )
                        
                        # Create stats record
                        player_stats = PlayerStats(
                            player_id=player.player_id,
                            season=season,
                            week=week,
                            stats=stats,
                            opponent=self._get_random_opponent(),
                            is_home=bool(week % 2),
                            game_date=datetime(season, 9, 1) + timedelta(weeks=week-1)
                        )
                        session.add(player_stats)
                
                await session.commit()
                logger.info(f"Generated stats for season {season}")
    
    def _generate_synthetic_stats(
        self, 
        position: str, 
        player_id: str,
        season: int,
        week: int
    ) -> Dict[str, Any]:
        """Generate realistic synthetic stats for demonstration"""
        np.random.seed(hash(f"{player_id}_{season}_{week}") % 2**32)
        
        if position == 'QB':
            return {
                'passing_yards': np.random.normal(250, 75),
                'passing_tds': np.random.poisson(1.8),
                'passing_int': np.random.poisson(0.8),
                'rushing_yards': np.random.normal(15, 10),
                'rushing_tds': np.random.poisson(0.1),
                'fumbles_lost': np.random.poisson(0.1)
            }
        elif position == 'RB':
            return {
                'rushing_yards': np.random.normal(70, 30),
                'rushing_tds': np.random.poisson(0.6),
                'receptions': np.random.poisson(3),
                'receiving_yards': np.random.normal(25, 15),
                'receiving_tds': np.random.poisson(0.1),
                'fumbles_lost': np.random.poisson(0.05)
            }
        elif position == 'WR':
            return {
                'receptions': np.random.poisson(5),
                'receiving_yards': np.random.normal(65, 25),
                'receiving_tds': np.random.poisson(0.4),
                'rushing_yards': np.random.normal(2, 5),
                'rushing_tds': np.random.poisson(0.02),
                'fumbles_lost': np.random.poisson(0.02)
            }
        elif position == 'TE':
            return {
                'receptions': np.random.poisson(3.5),
                'receiving_yards': np.random.normal(40, 20),
                'receiving_tds': np.random.poisson(0.3),
                'fumbles_lost': np.random.poisson(0.02)
            }
        
        return {}
    
    def _get_random_opponent(self) -> str:
        """Get random opponent team"""
        teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT',
                'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC',
                'DAL', 'NYG', 'PHI', 'WAS', 'CHI', 'DET', 'GB', 'MIN',
                'ATL', 'CAR', 'NO', 'TB', 'ARI', 'LAR', 'SF', 'SEA']
        return np.random.choice(teams)
    
    async def _calculate_fantasy_points(self):
        """Calculate fantasy points for all stats"""
        logger.info("Calculating fantasy points")
        
        async with self.db.async_session() as session:
            # Get all stats without calculated points
            result = await session.execute(
                select(PlayerStats).where(
                    PlayerStats.fantasy_points_ppr.is_(None)
                ).limit(1000)  # Process in batches
            )
            stats_to_update = result.scalars().all()
            
            for stat in stats_to_update:
                # Get player position
                result = await session.execute(
                    select(DBPlayer.position).where(
                        DBPlayer.player_id == stat.player_id
                    )
                )
                position = result.scalar_one()
                
                # Calculate points for each scoring system
                stat.fantasy_points_std = float(
                    self.scorers['standard'].calculate_points(stat.stats, position)
                )
                stat.fantasy_points_ppr = float(
                    self.scorers['ppr'].calculate_points(stat.stats, position)
                )
                stat.fantasy_points_half = float(
                    self.scorers['half_ppr'].calculate_points(stat.stats, position)
                )
            
            await session.commit()
            logger.info(f"Calculated fantasy points for {len(stats_to_update)} stats")
    
    async def _engineer_features(self) -> pd.DataFrame:
        """Engineer features for ML models"""
        logger.info("Engineering features")
        
        # Fetch data from database
        async with self.db.async_session() as session:
            # Get player stats for feature engineering
            query = """
                SELECT 
                    ps.player_id,
                    p.first_name || ' ' || p.last_name as player_name,
                    p.position,
                    ps.season,
                    ps.week,
                    ps.fantasy_points_ppr,
                    ps.opponent,
                    ps.is_home
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE ps.fantasy_points_ppr IS NOT NULL
                ORDER BY ps.player_id, ps.season, ps.week
            """
            
            result = await session.execute(query)
            data = result.fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(
            data, 
            columns=['player_id', 'player_name', 'position', 'season', 
                    'week', 'fantasy_points', 'opponent', 'is_home']
        )
        
        # Group by player for feature engineering
        features_list = []
        
        for player_id in df['player_id'].unique():
            player_df = df[df['player_id'] == player_id].copy()
            
            # Need enough history
            if len(player_df) < 5:
                continue
            
            # Mock feature engineering (simplified)
            # In production, use the full FeatureEngineer
            features = {
                'player_id': player_id,
                'player_name': player_df.iloc[-1]['player_name'],
                'position': player_df.iloc[-1]['position'],
                'points_per_game': player_df['fantasy_points'].mean(),
                'last_3_games_avg': player_df.tail(3)['fantasy_points'].mean(),
                'season_std_dev': player_df['fantasy_points'].std(),
                'games_played': len(player_df)
            }
            
            # Add 20+ engineered features here
            for i in range(20):
                features[f'feature_{i}'] = np.random.randn()
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Engineered features for {len(features_df)} players")
        
        return features_df
    
    def _should_retrain_models(self) -> bool:
        """Check if models need retraining"""
        if self.last_model_update is None:
            return True
        
        days_since_update = (datetime.now() - self.last_model_update).days
        return days_since_update >= self.config.retrain_interval_days
    
    async def _train_models(self, features_df: pd.DataFrame):
        """Train ML models"""
        logger.info("Training ML models")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col.startswith('feature_') or col in [
                           'points_per_game', 'last_3_games_avg', 
                           'season_std_dev', 'games_played'
                       ]]
        
        X = features_df[feature_cols].values
        y = features_df['points_per_game'].values
        positions = features_df['position'].values
        
        # Train GMM for draft tiers
        logger.info("Training GMM clustering model")
        self.gmm_optimizer.fit(X, feature_cols)
        
        # Train neural network
        logger.info("Training neural network")
        self.neural_network = FantasyNeuralNetwork(
            input_dim=len(feature_cols),
            position_specific=True
        )
        
        self.neural_network.fit(
            X, y,
            positions=positions,
            feature_names=feature_cols,
            epochs=50,
            verbose=0
        )
        
        # Save models
        self._save_models()
        self.last_model_update = datetime.now()
        
        logger.info("Model training complete")
    
    async def _generate_predictions(self):
        """Generate weekly predictions"""
        logger.info("Generating predictions")
        
        # Get current week
        async with self.sleeper_client as client:
            nfl_state = await client.get_nfl_state()
            current_week = nfl_state['week']
            current_season = int(nfl_state['season'])
        
        # Get features for prediction
        features_df = await self._engineer_features()
        
        if self.neural_network is None:
            logger.warning("No trained model available for predictions")
            return
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col.startswith('feature_') or col in [
                           'points_per_game', 'last_3_games_avg', 
                           'season_std_dev', 'games_played'
                       ]]
        
        X = features_df[feature_cols].values
        positions = features_df['position'].values
        
        # Generate predictions
        predictions = self.neural_network.predict(X, positions)
        
        # Store predictions in database
        async with self.db.async_session() as session:
            for i, pred in enumerate(predictions):
                player_id = features_df.iloc[i]['player_id']
                
                # Check if prediction exists
                result = await session.execute(
                    select(Prediction).where(
                        and_(
                            Prediction.player_id == player_id,
                            Prediction.season == current_season,
                            Prediction.week == current_week
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing prediction
                    existing.predicted_points = pred.predicted_points
                    existing.confidence_interval = {
                        'low': pred.confidence_interval[0],
                        'high': pred.confidence_interval[1]
                    }
                    existing.prediction_std = pred.prediction_std
                else:
                    # Create new prediction
                    new_pred = Prediction(
                        player_id=player_id,
                        season=current_season,
                        week=current_week,
                        predicted_points=pred.predicted_points,
                        confidence_interval={
                            'low': pred.confidence_interval[0],
                            'high': pred.confidence_interval[1]
                        },
                        prediction_std=pred.prediction_std,
                        model_version=self.neural_network.model_version,
                        model_type='neural_network',
                        features_used=pred.feature_importance
                    )
                    session.add(new_pred)
            
            await session.commit()
        
        logger.info(f"Generated {len(predictions)} predictions for week {current_week}")
    
    async def _update_draft_tiers(self):
        """Update draft tier assignments"""
        logger.info("Updating draft tiers")
        
        # Get features
        features_df = await self._engineer_features()
        
        if not self.gmm_optimizer.is_fitted:
            logger.warning("GMM model not fitted")
            return
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col.startswith('feature_') or col in [
                           'points_per_game', 'last_3_games_avg', 
                           'season_std_dev', 'games_played'
                       ]]
        
        X = features_df[feature_cols].values
        
        # Predict tiers
        draft_tiers = self.gmm_optimizer.predict_tiers(
            X,
            features_df['player_id'].tolist(),
            features_df['player_name'].tolist(),
            features_df['position'].tolist(),
            features_df['points_per_game'].tolist()
        )
        
        # Store in database
        async with self.db.async_session() as session:
            for tier in draft_tiers:
                # Create tier record
                db_tier = DraftTier(
                    player_id=tier.player_id,
                    season=self.config.current_season,
                    tier=tier.tier,
                    probability=tier.probability,
                    cluster_features={},  # Simplified
                    tier_label=tier.tier_label,
                    alt_tiers=tier.alternative_tiers,
                    model_version=self.gmm_optimizer.model_version
                )
                session.add(db_tier)
            
            await session.commit()
        
        logger.info(f"Updated {len(draft_tiers)} draft tier assignments")
    
    def _save_models(self):
        """Save trained models to disk"""
        import os
        
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save GMM
        self.gmm_optimizer.save_model(os.path.join(models_dir, 'gmm_model.pkl'))
        
        # Save neural network
        if self.neural_network:
            self.neural_network.save_model(os.path.join(models_dir, 'neural_network'))
        
        logger.info("Models saved successfully")
    
    def _load_models(self):
        """Load existing models from disk"""
        import os
        
        models_dir = 'models'
        
        # Load GMM
        gmm_path = os.path.join(models_dir, 'gmm_model.pkl')
        if os.path.exists(gmm_path):
            self.gmm_optimizer.load_model(gmm_path)
        
        # Load neural network
        nn_path = os.path.join(models_dir, 'neural_network')
        if os.path.exists(nn_path):
            self.neural_network = FantasyNeuralNetwork(input_dim=26)
            self.neural_network.load_model(nn_path)
        
        logger.info("Models loaded successfully")
    
    async def run_weekly_update(self):
        """Run weekly update (lighter than full update)"""
        logger.info("Running weekly update")
        
        # Update current week stats
        await self._update_players()
        await self._calculate_fantasy_points()
        
        # Generate new predictions
        await self._generate_predictions()
        
        logger.info("Weekly update complete")
    
    def schedule_updates(self):
        """Schedule automatic updates"""
        # Weekly updates every Tuesday
        schedule.every().tuesday.at("06:00").do(
            lambda: asyncio.run(self.run_weekly_update())
        )
        
        # Full update monthly
        schedule.every(4).weeks.do(
            lambda: asyncio.run(self.run_full_update())
        )
        
        logger.info("Update schedule configured")
    
    async def get_player_rankings(
        self,
        position: Optional[str] = None,
        scoring: str = 'ppr'
    ) -> pd.DataFrame:
        """Get current player rankings"""
        async with self.db.async_session() as session:
            query = """
                SELECT 
                    p.player_id,
                    p.first_name || ' ' || p.last_name as name,
                    p.position,
                    p.team,
                    dt.tier,
                    dt.tier_label,
                    pred.predicted_points,
                    pred.confidence_interval
                FROM players p
                LEFT JOIN draft_tiers dt ON p.player_id = dt.player_id
                LEFT JOIN predictions pred ON p.player_id = pred.player_id
                WHERE p.status = 'Active'
            """
            
            if position:
                query += f" AND p.position = '{position}'"
            
            query += " ORDER BY pred.predicted_points DESC NULLS LAST"
            
            result = await session.execute(query)
            data = result.fetchall()
        
        df = pd.DataFrame(
            data,
            columns=['player_id', 'name', 'position', 'team', 
                    'tier', 'tier_label', 'predicted_points', 'confidence_interval']
        )
        
        return df


# Main execution
async def main():
    """Main execution for testing"""
    config = PipelineConfig(
        database_url="postgresql://user:pass@localhost/fantasy_football",
        seasons_to_fetch=[2022, 2023, 2024]
    )
    
    pipeline = FantasyFootballPipeline(config)
    
    # Initialize
    await pipeline.initialize()
    
    # Run full update
    await pipeline.run_full_update()
    
    # Get rankings
    rankings = await pipeline.get_player_rankings(position='QB')
    print("\nTop 10 QBs:")
    print(rankings.head(10))


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main())