#!/usr/bin/env python3
"""
Quick fix to remove the problematic line from the existing CLI.
This will patch the current CLI file to make it work.
"""

import re
from pathlib import Path

def fix_cli_file():
    """Fix the CLI file by removing problematic lines."""
    
    cli_file = Path("src/fantasy_ai/cli/main.py")
    
    if not cli_file.exists():
        print(f"❌ CLI file not found: {cli_file}")
        return False
    
    print(f"🔧 Fixing CLI file: {cli_file}")
    
    # Read the file
    content = cli_file.read_text()
    
    # Remove problematic lines
    # Remove any line that references async_commands outside of function definitions
    lines = content.split('\n')
    fixed_lines = []
    
    skip_next_lines = 0
    
    for i, line in enumerate(lines):
        # Skip lines that cause the error
        if 'for command in async_commands:' in line and 'def ' not in line:
            skip_next_lines = 3  # Skip this line and next few
            continue
        
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
            
        # Also remove any standalone async_commands references
        if line.strip().startswith('async_commands = [') and 'def ' not in line:
            # Skip until we find the closing bracket
            skip_next_lines = 10  # Generous skip
            continue
            
        fixed_lines.append(line)
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    # Create backup
    backup_file = cli_file.with_suffix('.py.broken')
    cli_file.rename(backup_file)
    print(f"📦 Created backup: {backup_file}")
    
    # Write fixed file
    cli_file.write_text(fixed_content)
    print(f"✅ Fixed CLI file written")
    
    return True

if __name__ == "__main__":
    print("🔧 Quick CLI Fix Tool")
    print("=" * 30)
    
    if fix_cli_file():
        print("\n✅ CLI fix completed!")
        print("Try running: python -m fantasy_ai.cli.main version")
    else:
        print("\n❌ CLI fix failed!")
        print("Use the minimal CLI instead: python minimal_cli.py version")