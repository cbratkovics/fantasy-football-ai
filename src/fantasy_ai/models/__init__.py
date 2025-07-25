"""
Fantasy Football AI - Models Package
Machine Learning components for player analysis and prediction.
"""

from .feature_engineering import FeatureEngineer, PlayerFeatures
from .gmm_clustering import FantasyGMM, PlayerTier
from .neural_network import FantasyNeuralNetwork, PredictionResult
from .ml_integration import FantasyFootballAI, PlayerAnalysis

__version__ = "1.0.0"

__all__ = [
    # Core ML Components
    "FeatureEngineer",
    "PlayerFeatures", 
    "FantasyGMM",
    "PlayerTier",
    "FantasyNeuralNetwork", 
    "PredictionResult",
    
    # Integrated System
    "FantasyFootballAI",
    "PlayerAnalysis",
]

# Package metadata
PACKAGE_INFO = {
    "name": "fantasy-football-ai-models",
    "version": __version__,
    "description": "ML models for fantasy football prediction and analysis",
    "components": {
        "feature_engineering": "Transforms raw NFL stats into ML features",
        "gmm_clustering": "Player tier classification using Gaussian Mixture Models", 
        "neural_network": "Weekly fantasy point prediction using deep learning",
        "ml_integration": "Unified system combining all ML components"
    },
    "target_accuracy": "89.2%",
    "target_mae": 0.45
}

def get_package_info():
    """Return package information and component details."""
    return PACKAGE_INFO