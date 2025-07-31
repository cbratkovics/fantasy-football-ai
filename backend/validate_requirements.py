#!/usr/bin/env python3
"""
Validate requirements.txt for deployment issues
"""

import re
import sys


def check_requirements():
    """Check requirements.txt for common issues"""
    issues = []
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("ERROR: requirements.txt not found!")
        return False
    
    # Check for local file paths
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Check for local file paths
        if '@' in line and ('file://' in line or '/' in line.split('@')[1]):
            issues.append(f"Line {i}: Local file path detected: {line}")
        
        # Check for git URLs (can cause issues in some environments)
        if 'git+' in line:
            issues.append(f"Line {i}: Git URL detected (may cause deployment issues): {line}")
        
        # Check for missing version pins
        if '==' not in line and '>=' not in line and '<=' not in line and '~=' not in line:
            if '[' in line:  # Handle extras like package[extra]
                pkg_name = line.split('[')[0]
            else:
                pkg_name = line
            if pkg_name and not pkg_name.startswith('#'):
                issues.append(f"Line {i}: No version specified for: {pkg_name}")
    
    # Check for Windows-specific packages
    windows_packages = ['pywin32', 'pypiwin32', 'windows-curses']
    for i, line in enumerate(lines, 1):
        for pkg in windows_packages:
            if pkg in line.lower():
                issues.append(f"Line {i}: Windows-specific package detected: {line.strip()}")
    
    # Check for duplicate packages
    packages = {}
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Extract package name
        if '==' in line:
            pkg_name = line.split('==')[0].strip()
        elif '>=' in line:
            pkg_name = line.split('>=')[0].strip()
        elif '<=' in line:
            pkg_name = line.split('<=')[0].strip()
        elif '~=' in line:
            pkg_name = line.split('~=')[0].strip()
        else:
            pkg_name = line.split('[')[0].strip() if '[' in line else line.strip()
        
        if pkg_name:
            if pkg_name in packages:
                issues.append(f"Line {i}: Duplicate package '{pkg_name}' (first seen on line {packages[pkg_name]})")
            else:
                packages[pkg_name] = i
    
    # Report results
    if issues:
        print("❌ Issues found in requirements.txt:\n")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\nTotal issues: {len(issues)}")
        return False
    else:
        print("✅ requirements.txt validation passed!")
        print(f"\nTotal packages: {len(packages)}")
        return True


def check_dependency_conflicts():
    """Check for known dependency conflicts"""
    print("\n🔍 Checking for dependency conflicts...")
    
    conflicts = []
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return False
    
    # Check langchain-anthropic and anthropic compatibility
    if 'langchain-anthropic' in content and 'anthropic' in content:
        # Extract versions
        import re
        langchain_match = re.search(r'langchain-anthropic==(\d+\.\d+\.\d+)', content)
        anthropic_match = re.search(r'anthropic==(\d+\.\d+\.\d+)', content)
        
        if langchain_match and anthropic_match:
            langchain_version = langchain_match.group(1)
            anthropic_version = anthropic_match.group(1)
            
            # Known compatibility issues
            if langchain_version == "0.1.1" and anthropic_version >= "1.0.0":
                conflicts.append(
                    f"langchain-anthropic {langchain_version} requires anthropic<1.0.0, "
                    f"but you have anthropic=={anthropic_version}"
                )
    
    if conflicts:
        print("\n❌ Dependency conflicts detected:")
        for conflict in conflicts:
            print(f"  • {conflict}")
        return False
    else:
        print("✅ No known dependency conflicts found")
        return True


if __name__ == "__main__":
    print("🚀 Railway Deployment Requirements Validator\n")
    print("=" * 50)
    
    valid = check_requirements()
    conflicts_ok = check_dependency_conflicts()
    
    print("\n" + "=" * 50)
    
    if valid and conflicts_ok:
        print("\n✅ All checks passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("\n❌ Issues detected. Please fix before deploying.")
        sys.exit(1)