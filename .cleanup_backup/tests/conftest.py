"""
Pytest configuration and fixtures for Fantasy Football AI tests.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_player_data():
    """Sample player data for testing."""
    return pd.DataFrame({
        "player_id": ["player_1", "player_2"],
        "name": ["Test Player 1", "Test Player 2"],
        "position": ["RB", "WR"],
        "team": ["KC", "BUF"],
        "fantasy_points": [150.5, 120.3],
        "games_played": [16, 15],
    })

@pytest.fixture
def mock_predictor():
    """Mock ML predictor for testing."""
    predictor = MagicMock()
    predictor.predict_player_performance.return_value = {
        "predicted_points": 15.2,
        "confidence_interval": (12.1, 18.3),
        "tier": 3,
        "tier_confidence": 0.85
    }
    return predictor

@pytest.fixture
def mock_data_manager():
    """Mock data manager for testing."""
    data_manager = MagicMock()
    return data_manager
