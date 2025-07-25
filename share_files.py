# Files to share with AI for Fantasy Football API integration fixes
# Copy and paste the contents of these files:

FILES_TO_SHARE = [
    "src/fantasy_ai/cli/cli_commands.py",
    "src/fantasy_ai/cli/main.py",
    "src/fantasy_ai/core/data/etl.py",
    "src/fantasy_ai/core/data/orchestrator.py",
    "src/fantasy_ai/core/data/sources/nfl.py",
    "src/fantasy_ai/core/data/sources/nfl_comprehensive.py",
    "src/fantasy_ai/core/data/storage/database.py",
    "src/fantasy_ai/core/data/storage/models.py",
    "src/fantasy_ai/core/data/storage/simple_database.py",
    "src/fantasy_ai/models/feature_engineering.py",
    "src/fantasy_ai/models/gmm_clustering.py",
    "src/fantasy_ai/models/ml_integration.py",
    "src/fantasy_ai/models/neural_network.py",
]

# Run this to see file contents:
for file_path in FILES_TO_SHARE:
    print(f"\n{'='*60}")
    print(f"FILE: {file_path}")
    print('='*60)
    try:
        with open(file_path, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("❌ FILE NOT FOUND")
    except Exception as e:
        print(f"❌ ERROR: {e}")
