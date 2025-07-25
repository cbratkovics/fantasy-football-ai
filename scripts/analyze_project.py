#!/usr/bin/env python3
"""
Fantasy Football AI Project File Structure Analyzer and Organizer
Analyzes the project structure and identifies test files, backups, and production files.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json

def analyze_project_structure(project_root):
    """Analyze the entire project structure and categorize files."""
    
    project_path = Path(project_root)
    
    # File categories
    categories = {
        'production': [],
        'test_files': [],
        'backup_files': [],
        'jupyter_checkpoints': [],
        'cache_files': [],
        'log_files': [],
        'temporary_files': [],
        'configuration': [],
        'documentation': []
    }
    
    # Patterns for categorization
    test_patterns = [
        'test_', '_test', 'minimal_db_test', 'debug_', 'temp_'
    ]
    
    backup_patterns = [
        '.backup', '.bak', '.old', '.working', '.broken', '.corrupted', '.slow'
    ]
    
    temp_patterns = [
        '.tmp', '.temp', '__pycache__', '.pyc'
    ]
    
    config_patterns = [
        '.env', '.gitignore', 'requirements.txt', 'setup.py', 'pyproject.toml'
    ]
    
    doc_patterns = [
        '.md', '.txt', '.rst', 'README', 'LICENSE'
    ]
    
    # Walk through all files
    for root, dirs, files in os.walk(project_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_path)
            
            # Categorize file
            categorized = False
            
            # Check for Jupyter checkpoints
            if '.ipynb_checkpoints' in str(relative_path):
                categories['jupyter_checkpoints'].append(relative_path)
                categorized = True
            
            # Check for cache files
            elif '__pycache__' in str(relative_path) or file.endswith('.pyc'):
                categories['cache_files'].append(relative_path)
                categorized = True
            
            # Check for log files
            elif file.endswith('.log'):
                categories['log_files'].append(relative_path)
                categorized = True
            
            # Check for test files
            elif any(pattern in file.lower() for pattern in test_patterns):
                categories['test_files'].append(relative_path)
                categorized = True
            
            # Check for backup files
            elif any(pattern in file.lower() for pattern in backup_patterns):
                categories['backup_files'].append(relative_path)
                categorized = True
            
            # Check for temporary files
            elif any(pattern in file.lower() for pattern in temp_patterns):
                categories['temporary_files'].append(relative_path)
                categorized = True
            
            # Check for configuration files
            elif any(pattern in file.lower() for pattern in config_patterns):
                categories['configuration'].append(relative_path)
                categorized = True
            
            # Check for documentation
            elif any(file.lower().endswith(pattern) for pattern in doc_patterns):
                categories['documentation'].append(relative_path)
                categorized = True
            
            # Everything else is production
            if not categorized:
                categories['production'].append(relative_path)
    
    return categories

def print_file_analysis(categories, project_root):
    """Print detailed analysis of file categories."""
    
    print("=" * 80)
    print("FANTASY FOOTBALL AI PROJECT FILE ANALYSIS")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print()
    
    total_files = sum(len(files) for files in categories.values())
    print(f"Total Files Found: {total_files}")
    print()
    
    # Print each category
    for category, files in categories.items():
        if files:
            print(f"{category.upper().replace('_', ' ')} ({len(files)} files):")
            print("-" * 50)
            
            for file_path in sorted(files):
                file_size = get_file_size(Path(project_root) / file_path)
                print(f"  {file_path} ({file_size})")
            print()

def get_file_size(file_path):
    """Get human-readable file size."""
    try:
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    except OSError:
        return "Unknown"

def create_cleanup_commands(categories, project_root):
    """Generate commands to clean up the project."""
    
    cleanup_commands = []
    
    print("=" * 80)
    print("RECOMMENDED CLEANUP ACTIONS")
    print("=" * 80)
    
    # Safe to delete
    safe_to_delete = [
        'jupyter_checkpoints',
        'cache_files', 
        'temporary_files',
        'log_files'
    ]
    
    for category in safe_to_delete:
        files = categories.get(category, [])
        if files:
            print(f"\n{category.upper().replace('_', ' ')} - SAFE TO DELETE:")
            for file_path in files:
                full_path = Path(project_root) / file_path
                if full_path.is_dir():
                    print(f"  rm -rf '{file_path}'")
                    cleanup_commands.append(f"rm -rf '{full_path}'")
                else:
                    print(f"  rm '{file_path}'")
                    cleanup_commands.append(f"rm '{full_path}'")
    
    # Move to organized folders
    organize_categories = {
        'test_files': 'tests/',
        'backup_files': 'backups/',
        'documentation': 'docs/'
    }
    
    for category, target_dir in organize_categories.items():
        files = categories.get(category, [])
        if files:
            print(f"\n{category.upper().replace('_', ' ')} - MOVE TO {target_dir}:")
            target_path = Path(project_root) / target_dir
            print(f"  mkdir -p '{target_dir}'")
            cleanup_commands.append(f"mkdir -p '{target_path}'")
            
            for file_path in files:
                source = Path(project_root) / file_path
                dest = target_path / file_path.name
                print(f"  mv '{file_path}' '{target_dir}{file_path.name}'")
                cleanup_commands.append(f"mv '{source}' '{dest}'")
    
    return cleanup_commands

def generate_cleanup_script(cleanup_commands, project_root):
    """Generate a bash script to perform cleanup."""
    
    script_path = Path(project_root) / "cleanup_project.sh"
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Fantasy Football AI Project Cleanup Script\n")
        f.write("# Generated automatically - review before running\n\n")
        f.write("set -e  # Exit on error\n\n")
        f.write(f"cd '{project_root}'\n\n")
        
        f.write("echo 'Starting Fantasy Football AI project cleanup...'\n\n")
        
        for command in cleanup_commands:
            f.write(f"{command}\n")
        
        f.write("\necho 'Cleanup completed!'\n")
        f.write("echo 'Project structure organized.'\n")
    
    # Make script executable
    script_path.chmod(0o755)
    
    print(f"\n{'='*80}")
    print("CLEANUP SCRIPT GENERATED")
    print(f"{'='*80}")
    print(f"Script saved to: {script_path}")
    print(f"To run: ./cleanup_project.sh")
    print("IMPORTANT: Review the script before running!")

def show_production_structure(categories):
    """Show the clean production structure."""
    
    print(f"\n{'='*80}")
    print("PRODUCTION PROJECT STRUCTURE")
    print(f"{'='*80}")
    
    production_files = categories['production']
    config_files = categories['configuration']
    
    # Organize by directory
    structure = defaultdict(list)
    
    for file_path in production_files + config_files:
        if file_path.parent == Path('.'):
            structure['ROOT'].append(file_path.name)
        else:
            structure[str(file_path.parent)].append(file_path.name)
    
    # Print organized structure
    for directory in sorted(structure.keys()):
        if directory == 'ROOT':
            print("PROJECT ROOT:")
        else:
            print(f"{directory}/:")
        
        for file in sorted(structure[directory]):
            print(f"  {file}")
        print()

def main():
    """Main function to analyze and organize the project."""
    
    # Get project root (current directory)
    project_root = os.getcwd()
    
    print("Analyzing Fantasy Football AI project structure...")
    print(f"Project root: {project_root}")
    print()
    
    # Analyze the structure
    categories = analyze_project_structure(project_root)
    
    # Print analysis
    print_file_analysis(categories, project_root)
    
    # Show production structure
    show_production_structure(categories)
    
    # Create cleanup recommendations
    cleanup_commands = create_cleanup_commands(categories, project_root)
    
    # Generate cleanup script
    generate_cleanup_script(cleanup_commands, project_root)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_files = sum(len(files) for files in categories.values())
    production_files = len(categories['production']) + len(categories['configuration'])
    cleanup_files = total_files - production_files
    
    print(f"Total files: {total_files}")
    print(f"Production files: {production_files}")
    print(f"Files to clean up: {cleanup_files}")
    print(f"Space savings: {cleanup_files/total_files*100:.1f}% reduction")
    
    print(f"\nNext steps:")
    print(f"1. Review the analysis above")
    print(f"2. Check the generated cleanup_project.sh script")
    print(f"3. Run ./cleanup_project.sh to organize your project")
    print(f"4. Your production project will be clean and organized!")

if __name__ == "__main__":
    main()