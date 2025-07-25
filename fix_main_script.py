"""
Script to fix the main.py file
Run this to replace your corrupted main.py
"""

main_py_content = '''"""
Command Line Interface for Fantasy Football AI Data Collection System.
Location: src/fantasy_ai/cli/main.py
"""

import asyncio
import click
import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import os

# Set up import path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fantasy_ai.log')
    ]
)

logger = logging.getLogger(__name__)

def async_command(f):
    """Decorator to make Click commands async-compatible."""
    import functools
    
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# CLI Group
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config_file):
    """Fantasy Football AI Data Collection CLI."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file
    
    click.echo("Fantasy Football AI Data Collection System")
    click.echo("=" * 50)

# Import and add ML commands (fixed import)
try:
    # Try relative import first (when run as module)
    from .cli_commands import ml
    cli.add_command(ml)
except ImportError:
    # If relative import fails, try absolute import
    try:
        from fantasy_ai.cli.cli_commands import ml
        cli.add_command(ml)
    except ImportError:
        # If both fail, create a placeholder command
        @cli.group()
        def ml():
            """Machine Learning commands (not available - check installation)."""
            click.echo("ML commands not available. Check if cli_commands.py exists.")

# Database Commands
@cli.group()
def database():
    """Database management commands."""
    pass

@database.command()
@click.option('--drop-existing', is_flag=True, help='Drop existing tables before creating')
@async_command
async def init(drop_existing):
    """Initialize database and create tables."""
    
    click.echo("Initializing database...")
    
    try:
        # Lazy import - only loads when this command runs
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        
        db_manager = get_simple_db_manager()
        await db_manager.create_tables(drop_existing=drop_existing)
        
        click.echo("Database initialized successfully")
        click.echo(f"Database location: {db_manager.db_path}")
        
    except Exception as e:
        click.echo(f"Database initialization failed: {e}")
        sys.exit(1)

@database.command()
@async_command
async def stats():
    """Show database statistics."""
    
    click.echo("Database Statistics")
    click.echo("-" * 30)
    
    try:
        # Lazy import
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        
        db_manager = get_simple_db_manager()
        
        click.echo(f"Database URL: {db_manager.db_path}")
        click.echo(f"Engine Type: SQLite")
        
        async with db_manager.get_session() as session:
            from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
            from sqlalchemy import select, func
            
            # Get table counts
            result = await session.execute(select(func.count()).select_from(Team))
            team_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(Player))
            player_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(WeeklyStats))
            stats_count = result.scalar()
            
            click.echo("\\nTable Counts:")
            click.echo(f"  Teams: {team_count}")
            click.echo(f"  Players: {player_count}")
            click.echo(f"  Weekly Stats: {stats_count}")
            
    except Exception as e:
        click.echo(f"Failed to get database stats: {e}")

# API Commands
@cli.group()
def api():
    """NFL API management commands."""
    pass

@api.command()
@async_command
async def test():
    """Test NFL API connectivity."""
    
    click.echo("Testing NFL API connectivity...")
    
    # Check if API key is set
    if not os.getenv('NFL_API_KEY'):
        click.echo("NFL_API_KEY environment variable not set")
        click.echo("Please set your API key: export NFL_API_KEY='your_key_here'")
        return
    
    try:
        # Lazy import - only loads when this command runs
        from fantasy_ai.core.data.sources.nfl_comprehensive import create_nfl_client
        
        client = await create_nfl_client()
        
        try:
            health_status = await client.health_check()
            
            if health_status['api_accessible']:
                click.echo("NFL API connection successful")
                click.echo(f"Response time: {health_status['response_time']:.2f}s")
            else:
                click.echo("NFL API connection failed")
                if health_status.get('last_error'):
                    click.echo(f"Error: {health_status['last_error']}")
        finally:
            await client.close()
                    
    except Exception as e:
        click.echo(f"API test failed: {e}")
        if "NFL_API_KEY" in str(e):
            click.echo("Please ensure your NFL_API_KEY environment variable is set correctly")

# Collection Commands
@cli.group()
def collect():
    """Data collection commands."""
    pass

@collect.command()
@click.option('--max-calls', default=50, help='Maximum API calls to make')
@click.option('--positions', default='QB,RB,WR,TE', help='Positions to collect (comma-separated)')
@click.option('--seasons', default='2023', help='Seasons to collect (comma-separated)')
@click.option('--train-models', is_flag=True, help='Train ML models after collection')
@async_command
async def quick(max_calls, positions, seasons, train_models):
    """Run quick data collection for testing."""
    
    click.echo(f"Starting quick data collection (max {max_calls} API calls)")
    
    try:
        # Lazy import heavy modules only when needed
        from fantasy_ai.core.data.etl import FantasyFootballETL, CollectionConfig
        
        # Parse positions and seasons
        position_list = [p.strip().upper() for p in positions.split(',')]
        season_list = [int(s.strip()) for s in seasons.split(',')]
        
        click.echo(f"Positions: {', '.join(position_list)}")
        click.echo(f"Seasons: {', '.join(map(str, season_list))}")
        
        # Run quick ETL
        config = CollectionConfig(
            api_calls_per_day=max_calls,
            priority_positions=position_list,
            target_seasons=season_list,
            max_concurrent_tasks=2
        )
        
        etl = FantasyFootballETL(config)
        
        with click.progressbar(length=100, label='Collecting data') as bar:
            # Run ETL and update progress
            metrics = await etl.run_full_pipeline()
            bar.update(100)
        
        # Show results
        click.echo("\\nQuick collection completed!")
        click.echo(f"Players processed: {metrics.total_players_processed}")
        click.echo(f"Stats collected: {metrics.total_stats_collected}")
        click.echo(f"API calls made: {metrics.total_api_calls}")
        click.echo(f"Quality score: {metrics.validation_score:.3f}")
        
        # Optionally train ML models
        if train_models:
            click.echo("\\nTraining ML models on collected data...")
            
            try:
                # Import ML functions
                try:
                    from .cli_commands import _load_training_data
                    from ..models import FantasyFootballAI
                except ImportError:
                    from fantasy_ai.cli.cli_commands import _load_training_data
                    from fantasy_ai.models import FantasyFootballAI
                
                # Load the data we just collected
                training_data = _load_training_data(season_list)
                
                if not training_data.empty:
                    # Initialize and train AI system
                    ai_system = FantasyFootballAI(model_dir="models/")
                    results = ai_system.train_system(training_data, epochs=30)
                    
                    # Save the trained models
                    model_path = ai_system.save_models()
                    
                    click.echo(f"ML training complete!")
                    click.echo(f"Accuracy: {results['system_performance']['accuracy_percentage']:.1f}%")
                    click.echo(f"Models saved to: {model_path}")
                else:
                    click.echo("No training data available for ML models")
                    
            except Exception as e:
                click.echo(f"ML training failed: {e}")
                logger.exception("ML training error during quick collection")
        
    except Exception as e:
        click.echo(f"Quick collection failed: {e}")
        logger.exception("Quick collection error")

@collect.command()
@async_command
async def status():
    """Show current collection status."""
    
    click.echo("Collection Status")
    click.echo("-" * 30)
    
    try:
        # Lazy import
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        
        db_manager = get_simple_db_manager()
        
        async with db_manager.get_session() as session:
            from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
            from sqlalchemy import select, func
            
            # Get basic counts
            result = await session.execute(select(func.count()).select_from(Team))
            team_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(Player))
            player_count = result.scalar()
            
            result = await session.execute(select(func.count()).select_from(WeeklyStats))
            stats_count = result.scalar()
            
            click.echo("Database Contents:")
            click.echo(f"  Teams: {team_count}")
            click.echo(f"  Players: {player_count}")
            click.echo(f"  Weekly Stats: {stats_count}")
        
        click.echo(f"\\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        click.echo(f"Failed to get collection status: {e}")

@cli.command()
def version():
    """Show version information."""
    
    click.echo("Fantasy Football AI Data Collection System")
    click.echo("Version: 1.0.0")
    click.echo("Author: Christopher Bratkovics")
    click.echo("https://github.com/cbratkovics/fantasy-football-ai")

if __name__ == '__main__':
    cli()
'''

# Write the fixed content to main.py
with open('src/fantasy_ai/cli/main.py', 'w') as f:
    f.write(main_py_content)

print("Fixed main.py file has been written!")
print("You can now run:")
print("  python src/fantasy_ai/cli/main.py collect quick --max-calls 30 --seasons 2023")