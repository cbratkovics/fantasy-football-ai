#!/usr/bin/env python3
"""
Setup and initialization script for Fantasy Football AI Assistant.

This script handles initial setup, database initialization, and data collection.
Run this script after setting up your environment to get started quickly.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def check_imports():
    """Check if required modules can be imported."""
    try:
        import pandas
        import numpy
        import sqlite3
        from dotenv import load_dotenv
        # Load environment variables early
        load_dotenv()
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you've installed all requirements: pip install -r requirements.txt")
        return False

def check_espn_api():
    """Check if ESPN API is available."""
    try:
        from espn_api.football import League
        return True
    except ImportError:
        print("espn_api package required. Install with: pip install espn_api")
        return False

# Check basic imports first
if not check_imports():
    sys.exit(1)

if not check_espn_api():
    sys.exit(1)

# Try to import project modules (they might not exist yet)
try:
    from fantasy_ai.core.data.storage.models import DatabaseManager
    from fantasy_ai.core.data.sources.nfl import create_espn_collector
    from fantasy_ai.core.data.etl import FantasyDataETL
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Project modules not found: {e}")
    print("This is normal if you haven't created the source files yet.")
    print("Run this script with --create-files flag to create the required files first.")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FantasyFootballSetup:
    """Handles initial setup and configuration for the Fantasy Football AI system."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.src_dir = self.project_root / "src" / "fantasy_ai" / "core"
        
    def create_source_files(self) -> bool:
        """Create the required source files with template content."""
        logger.info("Creating source files...")
        
        # Create directory structure
        directories = [
            self.src_dir / "data" / "storage",
            self.src_dir / "data" / "sources", 
            self.src_dir / "models",
            self.src_dir / "api"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created directory: {directory}")
        
        # Create __init__.py files
        init_files = [
            self.project_root / "src" / "fantasy_ai" / "__init__.py",
            self.src_dir / "__init__.py",
            self.src_dir / "data" / "__init__.py",
            self.src_dir / "data" / "storage" / "__init__.py",
            self.src_dir / "data" / "sources" / "__init__.py",
            self.src_dir / "models" / "__init__.py",
            self.src_dir / "api" / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.write_text('"""Package initialization."""\n')
                logger.info(f"  Created: {init_file}")
        
        # Create placeholder files with basic content
        files_to_create = {
            self.src_dir / "data" / "storage" / "models.py": '''"""Database models - PLACEHOLDER
Replace this file with the complete SQLite models from the artifacts."""

class DatabaseManager:
    """Placeholder DatabaseManager"""
    def __init__(self, db_path="data/fantasy_football.db"):
        pass
        
    def get_database_stats(self):
        return {"status": "placeholder"}
''',
            self.src_dir / "data" / "sources" / "nfl.py": '''"""ESPN API client - PLACEHOLDER  
Replace this file with the complete ESPN API client from the artifacts."""

def create_espn_collector():
    """Placeholder function"""
    return None
''',
            self.src_dir / "data" / "etl.py": '''"""ETL Pipeline - PLACEHOLDER
Replace this file with the complete ETL pipeline from the artifacts."""

class FantasyDataETL:
    """Placeholder ETL class"""
    def __init__(self):
        pass
        
    def run_full_pipeline(self, **kwargs):
        return {"status": "placeholder"}
'''
        }
        
        for file_path, content in files_to_create.items():
            if not file_path.exists():
                file_path.write_text(content)
                logger.info(f"  Created placeholder: {file_path}")
        
        logger.info("Source files created. Replace placeholders with actual code from artifacts.")
        return True
        
    def create_directory_structure(self) -> None:
        """Create necessary directories."""
        logger.info("Creating directory structure...")
        
        directories = [
            self.data_dir,
            self.processed_dir,
            self.project_root / "logs",
            self.project_root / "models",
            self.project_root / "models" / "saved",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {directory}")
    
    def check_environment_variables(self) -> Dict[str, bool]:
        """Check if required environment variables are set."""
        # Ensure environment variables are loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        logger.info("Checking environment variables...")
        
        required_vars = {
            'ESPN_S2': os.getenv('ESPN_S2'),
            'SWID': os.getenv('SWID'),
            'LEAGUE_ID': os.getenv('LEAGUE_ID', '1099505687')  # Default provided
        }
        
        results = {}
        for var, value in required_vars.items():
            if value:
                results[var] = True
                logger.info(f"  Found: {var}: {value[:6]}..." if len(value) > 6 else f"  Found: {var}: {value}")
            else:
                results[var] = False
                logger.warning(f"  Missing: {var}: Not set")
        
        return results
    
    def create_env_file_template(self) -> None:
        """Create a .env file template."""
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            logger.info("Creating .env file template...")
            
            template = """# ESPN Fantasy Football API Credentials
# Get these from your ESPN Fantasy Football league page:
# 1. Open browser developer tools (F12)
# 2. Go to Application/Storage > Cookies
# 3. Find espn.com cookies

ESPN_S2=your_espn_s2_cookie_here
SWID=your_swid_cookie_here

# Your ESPN League ID (from URL when viewing your league)
LEAGUE_ID=1099505687

# Optional: Database configuration
DATABASE_PATH=data/fantasy_football.db

# Optional: Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
"""
            
            with open(env_file, 'w') as f:
                f.write(template)
                
            logger.info(f"  Created: {env_file}")
            logger.info("  Please edit .env file with your ESPN credentials")
        else:
            logger.info("  .env file already exists")
    
    def initialize_database(self) -> bool:
        """Initialize the SQLite database."""
        if not MODULES_AVAILABLE:
            logger.warning("Skipping database initialization - modules not available")
            return False
            
        logger.info("Initializing database...")
        
        try:
            db_manager = DatabaseManager()
            
            # Test database connection
            stats = db_manager.get_database_stats()
            logger.info("  Database initialized successfully")
            logger.info(f"  Current stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"  Database initialization failed: {e}")
            return False
    
    def test_espn_connection(self) -> bool:
        """Test ESPN API connection."""
        if not MODULES_AVAILABLE:
            logger.warning("Skipping ESPN connection test - modules not available")
            return False
            
        logger.info("Testing ESPN API connection...")
        
        try:
            collector = create_espn_collector()
            
            # Try to collect just a small sample to test connection
            test_data = collector.collect_season_data(2024)
            
            if not test_data.empty:
                logger.info(f"  ESPN API connection successful")
                logger.info(f"  Test data: {len(test_data)} records collected")
                return True
            else:
                logger.warning("  ESPN API connected but no data returned")
                return False
                
        except Exception as e:
            logger.error(f"  ESPN API connection failed: {e}")
            return False
    
    def run_initial_data_collection(self) -> bool:
        """Run initial data collection."""
        if not MODULES_AVAILABLE:
            logger.warning("Skipping data collection - modules not available")
            return False
            
        logger.info("Running initial data collection...")
        
        try:
            etl = FantasyDataETL()
            
            # Run ETL pipeline with data collection
            result = etl.run_full_pipeline(
                collect_fresh_data=True,
                process_features=True,
                save_training_data=True
            )
            
            if result.get('status') == 'SUCCESS':
                logger.info("  Initial data collection completed successfully")
                logger.info(f"  Collected {result.get('data_stats', {}).get('total_players', 0)} players")
                return True
            else:
                logger.error("  Initial data collection failed")
                return False
                
        except Exception as e:
            logger.error(f"  Initial data collection failed: {e}")
            return False
    
    def verify_setup(self) -> Dict[str, bool]:
        """Verify that setup was completed successfully."""
        logger.info("Verifying setup...")
        
        checks = {
            'directories': self._check_directories(),
            'database': self._check_database(),
            'data_files': self._check_data_files(),
            'environment': self._check_environment()
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info("  All setup checks passed!")
        else:
            logger.warning("  Some setup checks failed")
            
        return checks
    
    def _check_directories(self) -> bool:
        """Check if all required directories exist."""
        required_dirs = [self.data_dir, self.processed_dir]
        return all(d.exists() for d in required_dirs)
    
    def _check_database(self) -> bool:
        """Check if database is accessible and has data."""
        if not MODULES_AVAILABLE:
            return False
        try:
            db_manager = DatabaseManager()
            stats = db_manager.get_database_stats()
            return stats.get('total_players', 0) > 0
        except Exception:
            return False
    
    def _check_data_files(self) -> bool:
        """Check if key data files exist."""
        key_files = [
            self.data_dir / "fantasy_weekly_stats_combined.csv",
            self.processed_dir / "engineered_features.csv"
        ]
        return any(f.exists() for f in key_files)  # At least one should exist
    
    def _check_environment(self) -> bool:
        """Check if environment is properly configured."""
        from dotenv import load_dotenv
        load_dotenv()
        required_vars = ['ESPN_S2', 'SWID']
        return all(os.getenv(var) for var in required_vars)
    
    def print_next_steps(self) -> None:
        """Print next steps for the user."""
        logger.info("Setup completed! Next steps:")
        logger.info("")
        logger.info("1. Develop ML Models:")
        logger.info("   python -c \"from src.fantasy_ai.core.models.predictor import FantasyMLPredictor; print('Ready for ML development')\"")
        logger.info("")
        logger.info("2. Start Web Interface:")
        logger.info("   streamlit run src/fantasy_ai/web/app.py")
        logger.info("")
        logger.info("3. View Data:")
        logger.info("   python -c \"from src.fantasy_ai.core.data.etl import FantasyDataETL; etl = FantasyDataETL(); print(etl.get_pipeline_status())\"")
        logger.info("")
        logger.info("4. Update Data:")
        logger.info("   python -c \"from src.fantasy_ai.core.data.etl import run_etl_pipeline; run_etl_pipeline()\"")
        logger.info("")
        logger.info("Check the README.md for detailed documentation")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Fantasy Football AI Assistant')
    parser.add_argument('--skip-data', action='store_true', help='Skip initial data collection')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    parser.add_argument('--force-env', action='store_true', help='Force recreation of .env file')
    parser.add_argument('--create-files', action='store_true', help='Create placeholder source files')
    
    args = parser.parse_args()
    
    setup = FantasyFootballSetup()
    
    logger.info("Fantasy Football AI Assistant Setup")
    logger.info("=" * 50)
    
    # Handle create-files flag first
    if args.create_files:
        logger.info("Creating source file structure...")
        try:
            setup.create_directory_structure()
            setup.create_source_files()
            logger.info("Source files created successfully!")
            logger.info("Now replace the placeholder files with actual code from the artifacts.")
            logger.info("Then run setup.py again without --create-files flag.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to create source files: {e}")
            sys.exit(1)
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        logger.error("Project modules not available.")
        logger.info("Run: python setup.py --create-files")
        logger.info("Then replace placeholder files with actual code from artifacts.")
        sys.exit(1)
    
    if args.verify_only:
        # Only run verification
        checks = setup.verify_setup()
        if all(checks.values()):
            logger.info("All systems ready!")
            sys.exit(0)
        else:
            logger.error("Setup verification failed")
            sys.exit(1)
    
    # Full setup process
    success_steps = []
    
    # Step 1: Directory structure
    try:
        setup.create_directory_structure()
        success_steps.append("directories")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
    
    # Step 2: Environment file
    try:
        if args.force_env or not (Path(__file__).parent / ".env").exists():
            setup.create_env_file_template()
        success_steps.append("environment")
    except Exception as e:
        logger.error(f"Failed to create environment file: {e}")
    
    # Step 3: Check environment variables
    env_vars = setup.check_environment_variables()
    if not all(env_vars.values()):
        logger.warning("⚠️  Some environment variables are missing")
        logger.warning("Please edit the .env file with your ESPN credentials before continuing")
        
        if not env_vars.get('ESPN_S2') or not env_vars.get('SWID'):
            logger.error("ESPN_S2 and SWID are required. Setup cannot continue.")
            logger.info("Edit .env file and run setup again")
            sys.exit(1)
    
    # Step 4: Database initialization
    try:
        if setup.initialize_database():
            success_steps.append("database")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Step 5: Test ESPN connection
    try:
        if setup.test_espn_connection():
            success_steps.append("espn_connection")
    except Exception as e:
        logger.error(f"ESPN connection test failed: {e}")
    
    # Step 6: Initial data collection (optional)
    if not args.skip_data and "espn_connection" in success_steps:
        try:
            if setup.run_initial_data_collection():
                success_steps.append("data_collection")
        except Exception as e:
            logger.error(f"Initial data collection failed: {e}")
    elif args.skip_data:
        logger.info("Skipping initial data collection (--skip-data flag)")
    
    # Final verification
    logger.info("")
    verification = setup.verify_setup()
    
    logger.info("")
    logger.info("Setup Summary:")
    logger.info(f"  Successful steps: {len(success_steps)}")
    logger.info(f"  Steps completed: {', '.join(success_steps)}")
    
    if all(verification.values()) or len(success_steps) >= 4:
        logger.info("Setup completed successfully!")
        setup.print_next_steps()
        sys.exit(0)
    else:
        logger.warning("Setup completed with some issues")
        logger.info("You can still proceed with development, but some features may not work")
        setup.print_next_steps()
        sys.exit(0)


if __name__ == "__main__":
    main()