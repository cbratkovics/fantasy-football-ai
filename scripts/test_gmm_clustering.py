#!/usr/bin/env python3
"""
Test GMM clustering implementation
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.train import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gmm_clustering():
    """Test GMM clustering for draft tiers"""
    logger.info("Testing GMM Clustering Implementation")
    logger.info("="*60)
    
    trainer = ModelTrainer()
    
    try:
        # Train GMM model
        result = trainer.train_gmm_model()
        
        logger.info(f"\nGMM Training Successful!")
        logger.info(f"Number of tiers created: {result['n_clusters']}")
        logger.info(f"Model saved to: {result['model_path']}")
        
        # Load and analyze the saved tiers
        logger.info("\nAnalyzing tier assignments...")
        
        # Store tier assignments in database
        logger.info("\nStoring tier assignments in database...")
        trainer.store_draft_tiers()
        
        return True
        
    except Exception as e:
        logger.error(f"GMM clustering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def store_draft_tiers(self):
    """Store draft tier assignments in database"""
    from backend.models.database import DraftTier as DraftTierModel
    
    with self.SessionLocal() as db:
        # Clear existing tiers
        db.query(DraftTierModel).delete()
        
        # Load the most recent tier assignments
        df = self.load_historical_data(['QB', 'RB', 'WR', 'TE', 'K'])
        df_recent = df.loc[df.groupby('player_id')['season'].idxmax()]
        
        # Prepare features and predict tiers
        features_scaled, df_with_features = self.prepare_features_for_gmm(df_recent)
        
        tiers = self.gmm_clusterer.predict_tiers(
            features=features_scaled,
            player_ids=df_with_features['player_id'].tolist(),
            player_names=df_with_features['player_name'].tolist(),
            positions=df_with_features['position'].tolist(),
            expected_points=df_with_features['avg_points_ppr'].tolist()
        )
        
        # Store each tier assignment
        for tier in tiers:
            db_tier = DraftTierModel(
                player_id=tier.player_id,
                tier_number=tier.tier,
                tier_label=tier.tier_label,
                confidence_score=tier.probability,
                alternative_tiers=tier.alternative_tiers,
                season=2024,  # Current season
                created_at=datetime.utcnow()
            )
            db.add(db_tier)
        
        db.commit()
        logger.info(f"Stored {len(tiers)} tier assignments in database")


# Add the method to ModelTrainer
ModelTrainer.store_draft_tiers = store_draft_tiers


if __name__ == "__main__":
    from datetime import datetime
    test_gmm_clustering()