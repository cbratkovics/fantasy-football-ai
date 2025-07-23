def test_basic():
    """Basic test to ensure testing works."""
    assert True

def test_import():
    """Test that main package imports correctly."""
    import fantasy_ai
    assert fantasy_ai.__version__ == "1.0.0"
