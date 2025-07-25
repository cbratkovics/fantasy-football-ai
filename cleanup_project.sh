#!/bin/bash
# Fantasy Football AI Project Cleanup Script
# Generated automatically - review before running

set -e  # Exit on error

cd '/Users/christopherbratkovics/Desktop/fantasy-football-ai'

echo 'Starting Fantasy Football AI project cleanup...'

# SAFE TO DELETE - Cache files
echo 'Removing Python cache files...'
rm -rf 'src/fantasy_ai/__pycache__'
rm -rf 'src/fantasy_ai/cli/__pycache__'
rm -rf 'src/fantasy_ai/core/__pycache__'
rm -rf 'src/fantasy_ai/core/data/__pycache__'
rm -rf 'src/fantasy_ai/core/data/quality/__pycache__'
rm -rf 'src/fantasy_ai/core/data/sources/__pycache__'
rm -rf 'src/fantasy_ai/core/data/storage/__pycache__'

# SAFE TO DELETE - Log files
echo 'Removing empty log files...'
rm -f 'fantasy_ai.log'
rm -f 'src/fantasy_ai.log'

# ORGANIZE - Create directories
echo 'Creating organization directories...'
mkdir -p 'backups'
mkdir -p 'tests/dev'

# ORGANIZE - Move backup files
echo 'Moving backup files...'
mv 'src/fantasy_ai/cli/main.py.backup' 'backups/'
mv 'src/fantasy_ai/cli/main.py.bak' 'backups/'
mv 'src/fantasy_ai/cli/main.py.broken' 'backups/'
mv 'src/fantasy_ai/cli/main.py.working' 'backups/'

# ORGANIZE - Move test/debug files
echo 'Moving test and debug files...'
mv 'debug_nfl_api.py' 'tests/dev/'
mv 'debug_test.py' 'tests/dev/'
mv 'minimal_db_test.py' 'tests/dev/'
mv 'proper_api_flow_test.py' 'tests/dev/'
mv 'test_nfl_api.py' 'tests/dev/'
mv 'quick_fix_test.sh' 'tests/dev/'

# ORGANIZE - Move temporary files to appropriate locations
echo 'Moving temporary files...'
mv 'quick_cli_fix.py' 'scripts/' 2>/dev/null || mv 'quick_cli_fix.py' 'backups/'
mv 'analyze_project.py' 'scripts/'

# ORGANIZE - Clean up empty egg-info if desired (optional)
echo 'Cleaning up build artifacts...'
rm -rf 'src/fantasy_football_ai.egg-info' 2>/dev/null || true

# ORGANIZE - Remove .DS_Store files (macOS)
echo 'Removing macOS .DS_Store files...'
find . -name '.DS_Store' -type f -delete 2>/dev/null || true

# ORGANIZE - Clean up any remaining empty __pycache__ directories
echo 'Cleaning up remaining cache directories...'
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo ''
echo 'Cleanup completed!'
echo 'Project structure organized:'
echo '  - Cache files removed'
echo '  - Backup files moved to backups/'
echo '  - Test files moved to tests/dev/'
echo '  - Scripts organized in scripts/'
echo ''
echo 'Your production project is now clean and organized!'
echo ''
echo 'To verify the cleanup worked:'
echo '  python src/fantasy_ai/cli/main.py version'
echo '  python src/fantasy_ai/cli/main.py database --help'