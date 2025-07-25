#!/usr/bin/env python3
"""
Fantasy Football AI Project Cleanup Script
Organizes project structure and removes unnecessary files.
"""

import os
import shutil
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Set

class ProjectCleanup:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_dir = self.project_root / "src" / "fantasy_ai"
        self.backup_dir = self.project_root / ".cleanup_backup"
        
        # Define proper project structure
        self.required_dirs = {
            "src/fantasy_ai/core/data/sources",
            "src/fantasy_ai/core/data/storage", 
            "src/fantasy_ai/core/data/quality",
            "src/fantasy_ai/core/data/features",
            "src/fantasy_ai/core/models",
            "src/fantasy_ai/core/utils",
            "src/fantasy_ai/web",
            "src/fantasy_ai/cli",
            "tests/unit",
            "tests/integration", 
            "tests/fixtures",
            "docs/api",
            "docs/architecture",
            "data/raw",
            "data/processed",
            "data/external",
            "models/trained",
            "models/experiments",
            "config",
            "scripts",
            "notebooks/exploratory",
            "notebooks/analysis"
        }
        
        # Files that should be removed completely
        self.files_to_remove = {
            "__pycache__",
            "*.pyc",
            "*.pyo", 
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.temp",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "*.egg-info",
            "dist",
            "build",
            ".tox",
            ".mypy_cache"
        }
        
        # File patterns to organize by directory
        self.file_organization = {
            "config": ["*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.env*"],
            "scripts": ["*.sh", "*.bat", "cleanup_project.py"],
            "docs": ["*.md", "*.rst", "*.txt"],
            "notebooks": ["*.ipynb"],
            "data/raw": ["*.csv", "*.json", "*.xml"] 
        }

    def create_backup(self) -> None:
        """Create backup of current state."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        print(f"Creating backup in {self.backup_dir}")
        shutil.copytree(self.project_root, self.backup_dir, 
                       ignore=shutil.ignore_patterns(
                           '__pycache__', '*.pyc', '.git', '.cleanup_backup'
                       ))

    def create_directory_structure(self) -> None:
        """Create required directory structure."""
        print("Creating required directory structure...")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files for Python packages
            if "src/fantasy_ai" in str(dir_path):
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
        
        print(f"Created {len(self.required_dirs)} directories")

    def remove_unwanted_files(self) -> None:
        """Remove unwanted files and directories."""
        removed_count = 0
        
        print("Removing unwanted files...")
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Skip backup directory
            if '.cleanup_backup' in str(root_path):
                continue
                
            # Remove unwanted directories
            dirs_to_remove = []
            for d in dirs:
                if d in self.files_to_remove:
                    dir_path = root_path / d
                    print(f"Removing directory: {dir_path}")
                    shutil.rmtree(dir_path)
                    dirs_to_remove.append(d)
                    removed_count += 1
            
            # Remove from dirs list to prevent walking into them
            for d in dirs_to_remove:
                dirs.remove(d)
            
            # Remove unwanted files
            for f in files:
                file_path = root_path / f
                
                # Check if file matches removal patterns
                should_remove = False
                for pattern in self.files_to_remove:
                    if pattern.startswith('*'):
                        if f.endswith(pattern[1:]):
                            should_remove = True
                            break
                    elif f == pattern:
                        should_remove = True
                        break
                
                if should_remove:
                    print(f"Removing file: {file_path}")
                    file_path.unlink()
                    removed_count += 1
        
        print(f"Removed {removed_count} unwanted files/directories")

    def organize_files(self) -> None:
        """Organize files into proper directories."""
        moved_count = 0
        
        print("Organizing files...")
        
        # Walk through project root (not subdirectories)
        for item in self.project_root.iterdir():
            if item.is_file() and item.name not in ['.gitignore', 'README.md', 'pyproject.toml', 'requirements.txt']:
                
                # Determine where file should go
                target_dir = None
                for dir_name, patterns in self.file_organization.items():
                    for pattern in patterns:
                        if pattern.startswith('*'):
                            if item.name.endswith(pattern[1:]):
                                target_dir = dir_name
                                break
                        elif item.name == pattern:
                            target_dir = dir_name
                            break
                    if target_dir:
                        break
                
                # Move file if target found
                if target_dir:
                    target_path = self.project_root / target_dir
                    target_path.mkdir(parents=True, exist_ok=True)
                    
                    new_location = target_path / item.name
                    print(f"Moving {item.name} to {target_dir}/")
                    shutil.move(str(item), str(new_location))
                    moved_count += 1
        
        print(f"Organized {moved_count} files")

    def create_essential_files(self) -> None:
        """Create essential project files if they don't exist."""
        essential_files = {
            ".gitignore": self._get_gitignore_content(),
            "pyproject.toml": self._get_pyproject_content(),
            "src/fantasy_ai/__init__.py": '"""Fantasy Football AI Package."""\n__version__ = "0.1.0"\n',
            "src/fantasy_ai/core/__init__.py": '"""Core functionality."""\n',
            "src/fantasy_ai/core/data/__init__.py": '"""Data processing modules."""\n',
            "README.md": self._get_readme_content() if not (self.project_root / "README.md").exists() else None
        }
        
        created_count = 0
        for file_path, content in essential_files.items():
            if content is None:
                continue
                
            full_path = self.project_root / file_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                print(f"Created: {file_path}")
                created_count += 1
        
        print(f"Created {created_count} essential files")

    def _get_gitignore_content(self) -> str:
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Data
data/raw/*.csv
data/raw/*.json
data/processed/*.csv
data/processed/*.json

# Models
models/trained/*.pkl
models/trained/*.joblib
models/trained/*.h5

# Jupyter
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Backup
.cleanup_backup/
"""

    def _get_pyproject_content(self) -> str:
        return """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fantasy-football-ai"
version = "0.1.0"
description = "Advanced Fantasy Football AI Assistant with Neural Networks and Gaussian Mixture Models"
authors = [{name = "Christopher Bratkovics", email = "cbratkovics@gmail.com"}]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "tensorflow>=2.13.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "aiohttp>=3.8.0",
    "asyncio-throttle>=1.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
src_paths = ["src", "tests"]

[tool.mypy]
python_version = "3.10"
strict = true
"""

    def _get_readme_content(self) -> str:
        return """# Fantasy Football AI Assistant

Advanced Fantasy Football prediction system using Neural Networks and Gaussian Mixture Models.

## Project Structure

```
fantasy-football-ai/
├── src/fantasy_ai/          # Main package
├── tests/                   # Test suite
├── data/                    # Data storage
├── models/                  # Trained models
├── docs/                    # Documentation
├── config/                  # Configuration files
├── scripts/                 # Utility scripts
└── notebooks/               # Analysis notebooks
```

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m fantasy_ai.cli.main
```
"""

    def cleanup(self, create_backup: bool = True) -> None:
        """Run complete cleanup process."""
        print(f"Starting cleanup of project: {self.project_root}")
        
        if create_backup:
            self.create_backup()
        
        self.create_directory_structure()
        self.remove_unwanted_files() 
        self.organize_files()
        self.create_essential_files()
        
        print("\nCleanup completed successfully!")
        print(f"Project structure organized in: {self.project_root}")
        if create_backup:
            print(f"Backup available in: {self.backup_dir}")

def main():
    parser = argparse.ArgumentParser(description="Clean up Fantasy Football AI project structure")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run logic
        return
    
    try:
        cleanup = ProjectCleanup(args.project_root)
        cleanup.cleanup(create_backup=not args.no_backup)
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()