"""
Simple Database Check
Quick check to see what's happening with the database.
"""

import sys
import os
from pathlib import Path

# Add your src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def check_database_file():
    """Check if database file exists and basic info."""
    
    print("Fantasy Football AI - Simple Database Check")
    print("=" * 50)
    
    expected_db_path = Path("src/data/fantasy_football.db")
    
    print(f"Expected database path: {expected_db_path.absolute()}")
    print(f"Database file exists: {expected_db_path.exists()}")
    
    if expected_db_path.exists():
        stat = expected_db_path.stat()
        print(f"File size: {stat.st_size} bytes")
        print(f"Created: {stat.st_ctime}")
    
    # Check if data directory exists
    data_dir = Path("src/data")
    print(f"Data directory exists: {data_dir.exists()}")
    
    if data_dir.exists():
        files = list(data_dir.iterdir())
        print(f"Files in data directory: {[f.name for f in files]}")
    
    # Try to import database module
    print(f"\nTrying to import database modules...")
    
    try:
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        print("✓ simple_database module imported successfully")
        
        db_manager = get_simple_db_manager()
        print(f"✓ Database manager created")
        print(f"  Database path: {db_manager.db_path}")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Check if the module path is correct")
        
    except Exception as e:
        print(f"✗ Database manager creation failed: {e}")
    
    # Try to import models
    try:
        from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
        print("✓ Database models imported successfully")
        
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        
    except Exception as e:
        print(f"✗ Models import error: {e}")
    
    # Check project structure
    print(f"\nProject structure check:")
    
    expected_paths = [
        "src/fantasy_ai/__init__.py",
        "src/fantasy_ai/core/__init__.py", 
        "src/fantasy_ai/core/data/__init__.py",
        "src/fantasy_ai/core/data/storage/__init__.py",
        "src/fantasy_ai/core/data/storage/simple_database.py",
        "src/fantasy_ai/core/data/storage/models.py",
        "src/fantasy_ai/cli/main.py",
        "src/fantasy_ai/models/__init__.py"
    ]
    
    for path in expected_paths:
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")
    
    print(f"\nEnvironment check:")
    print(f"  Python version: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  NFL_API_KEY set: {'Yes' if os.getenv('NFL_API_KEY') else 'No'}")

if __name__ == "__main__":
    check_database_file()