#!/usr/bin/env python3
"""
Minimal ML training test with a small dataset
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.enhanced_training import EnhancedMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_minimal_training():
    """Test ML training with minimal data"""
    
    logger.info("Starting minimal ML training test...")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = EnhancedMLPipeline()
    
    # Run training on single season to verify it works
    logger.info("Training on 2022 season only (minimal test)...")
    
    try:
        # Train on just 2022, test on partial 2023
        metrics = await pipeline.run_training_pipeline(
            seasons=[2022],  # Just one season
            test_season=2023
        )
        
        logger.info("\n✅ Minimal training completed successfully!")
        logger.info(f"Test metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal_training())
    sys.exit(0 if success else 1)