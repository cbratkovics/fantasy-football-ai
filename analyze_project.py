#!/usr/bin/env python3
"""
Fantasy Football AI Project Analyzer
Analyzes project structure and identifies key files for API integration fixes.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ProjectAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.key_files = {}
        self.issues = []
        
    def analyze_project(self) -> Dict:
        """Analyze the entire project structure and identify key files."""
        print(f"Analyzing Fantasy Football AI project at: {self.project_root}")
        print("=" * 80)
        
        # Define critical files for API integration
        critical_files = {
            "api_integration": [
                "src/fantasy_ai/core/data/sources/nfl.py",
                "src/fantasy_ai/core/data/sources/nfl_comprehensive.py",
                "src/fantasy_ai/core/data/etl.py",
                "src/fantasy_ai/core/data/orchestrator.py"
            ],
            "database": [
                "src/fantasy_ai/core/data/storage/database.py",
                "src/fantasy_ai/core/data/storage/models.py",
                "src/fantasy_ai/core/data/storage/simple_database.py"
            ],
            "cli": [
                "src/fantasy_ai/cli/main.py",
                "src/fantasy_ai/cli/cli_commands.py"
            ],
            "config": [
                ".env",
                ".env.example",
                "pyproject.toml",
                "requirements.txt"
            ],
            "ml_models": [
                "src/fantasy_ai/models/ml_integration.py",
                "src/fantasy_ai/models/feature_engineering.py",
                "src/fantasy_ai/models/neural_network.py",
                "src/fantasy_ai/models/gmm_clustering.py"
            ]
        }
        
        # Analyze each category
        analysis = {}
        for category, file_list in critical_files.items():
            analysis[category] = self._analyze_file_category(category, file_list)
        
        # Check for API-Sports specific integration
        analysis["api_sports_integration"] = self._check_api_sports_integration()
        
        # Check environment and configuration
        analysis["environment"] = self._check_environment()
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations()
        
        return analysis
    
    def _analyze_file_category(self, category: str, file_list: List[str]) -> Dict:
        """Analyze a specific category of files."""
        category_analysis = {
            "category": category,
            "files": {},
            "missing_files": [],
            "issues": []
        }
        
        for file_path in file_list:
            full_path = self.project_root / file_path
            
            if full_path.exists():
                file_info = self._analyze_file(full_path)
                category_analysis["files"][file_path] = file_info
            else:
                category_analysis["missing_files"].append(file_path)
        
        return category_analysis
    
    def _analyze_file(self, file_path: Path) -> Dict:
        """Analyze a specific file."""
        try:
            file_info = {
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "exists": True,
                "readable": os.access(file_path, os.R_OK),
                "key_patterns": []
            }
            
            if file_path.suffix == '.py':
                file_info.update(self._analyze_python_file(file_path))
            elif file_path.name in ['.env', '.env.example']:
                file_info.update(self._analyze_env_file(file_path))
            elif file_path.suffix in ['.toml', '.txt']:
                file_info.update(self._analyze_config_file(file_path))
            
            return file_info
            
        except Exception as e:
            return {
                "path": str(file_path),
                "exists": True,
                "error": str(e),
                "readable": False
            }
    
    def _analyze_python_file(self, file_path: Path) -> Dict:
        """Analyze Python file for API integration patterns."""
        analysis = {
            "type": "python",
            "imports": [],
            "classes": [],
            "functions": [],
            "api_patterns": [],
            "db_patterns": [],
            "issues": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check for key patterns
            patterns_to_check = {
                "api_patterns": [
                    "api-sports", "nfl_api", "NFL_API_KEY", "x-apisports-key",
                    "requests.get", "aiohttp", "httpx", "api_key"
                ],
                "db_patterns": [
                    "sqlalchemy", "database", "session", "create_engine",
                    "Player", "WeeklyStats", "fantasy_points"
                ],
                "ml_patterns": [
                    "tensorflow", "sklearn", "pandas", "numpy",
                    "train_system", "predict", "FantasyFootballAI"
                ]
            }
            
            for pattern_type, patterns in patterns_to_check.items():
                found_patterns = []
                for pattern in patterns:
                    if pattern.lower() in content.lower():
                        # Find line numbers
                        line_nums = [i+1 for i, line in enumerate(lines) 
                                   if pattern.lower() in line.lower()]
                        if line_nums:
                            found_patterns.append({
                                "pattern": pattern,
                                "lines": line_nums[:5]  # First 5 occurrences
                            })
                analysis[pattern_type] = found_patterns
            
            # Extract imports, classes, functions
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    analysis["imports"].append({"line": i+1, "import": line})
                elif line.startswith('class '):
                    analysis["classes"].append({"line": i+1, "class": line})
                elif line.startswith('def ') or line.startswith('async def '):
                    analysis["functions"].append({"line": i+1, "function": line})
            
        except Exception as e:
            analysis["issues"].append(f"Error reading file: {e}")
        
        return analysis
    
    def _analyze_env_file(self, file_path: Path) -> Dict:
        """Analyze environment file."""
        analysis = {"type": "env", "variables": [], "issues": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    analysis["variables"].append({
                        "line": i+1,
                        "variable": var_name,
                        "has_value": len(line.split('=', 1)[1].strip()) > 0
                    })
                    
        except Exception as e:
            analysis["issues"].append(f"Error reading env file: {e}")
        
        return analysis
    
    def _analyze_config_file(self, file_path: Path) -> Dict:
        """Analyze configuration files."""
        analysis = {"type": "config", "content_preview": "", "issues": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Store first 500 characters as preview
            analysis["content_preview"] = content[:500]
            
            if file_path.suffix == '.toml':
                analysis["config_type"] = "toml"
            elif file_path.suffix == '.txt':
                analysis["config_type"] = "requirements"
                # Count dependencies
                lines = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.startswith('#')]
                analysis["dependency_count"] = len(lines)
                
        except Exception as e:
            analysis["issues"].append(f"Error reading config file: {e}")
        
        return analysis
    
    def _check_api_sports_integration(self) -> Dict:
        """Check for API-Sports specific integration."""
        integration_check = {
            "api_key_configured": False,
            "api_client_exists": False,
            "fantasy_points_calculation": False,
            "rate_limiting": False,
            "caching": False,
            "issues": []
        }
        
        # Check for API key in environment
        env_file = self.project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    env_content = f.read()
                if "NFL_API_KEY" in env_content:
                    integration_check["api_key_configured"] = True
            except:
                pass
        
        # Check API client implementation
        api_files = [
            "src/fantasy_ai/core/data/sources/nfl.py",
            "src/fantasy_ai/core/data/sources/nfl_comprehensive.py"
        ]
        
        for api_file in api_files:
            file_path = self.project_root / api_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                    
                    if "api-sports" in content or "x-apisports-key" in content:
                        integration_check["api_client_exists"] = True
                    
                    if "fantasy_points" in content or "calculate" in content:
                        integration_check["fantasy_points_calculation"] = True
                    
                    if "rate" in content or "limit" in content:
                        integration_check["rate_limiting"] = True
                    
                    if "cache" in content or "redis" in content:
                        integration_check["caching"] = True
                        
                except:
                    pass
        
        return integration_check
    
    def _check_environment(self) -> Dict:
        """Check Python environment and dependencies."""
        env_check = {
            "python_version": sys.version,
            "project_root_writable": os.access(self.project_root, os.W_OK),
            "virtual_env": "VIRTUAL_ENV" in os.environ,
            "required_packages": {}
        }
        
        # Check for required packages
        required_packages = [
            "requests", "pandas", "numpy", "sqlalchemy", 
            "tensorflow", "scikit-learn", "click", "aiohttp"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                env_check["required_packages"][package] = "installed"
            except ImportError:
                env_check["required_packages"][package] = "missing"
        
        return env_check
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate specific recommendations for fixes."""
        recommendations = []
        
        # Check if API integration files exist
        api_files = [
            "src/fantasy_ai/core/data/sources/nfl_comprehensive.py",
            "src/fantasy_ai/core/data/etl.py"
        ]
        
        missing_api_files = []
        for api_file in api_files:
            if not (self.project_root / api_file).exists():
                missing_api_files.append(api_file)
        
        if missing_api_files:
            recommendations.append({
                "priority": "HIGH",
                "category": "API Integration",
                "issue": "Missing API integration files",
                "files": missing_api_files,
                "action": "Need to share these files for API-Sports integration fixes"
            })
        
        # Check environment configuration
        env_file = self.project_root / ".env"
        if not env_file.exists():
            recommendations.append({
                "priority": "HIGH",
                "category": "Configuration",
                "issue": "Missing .env file",
                "action": "Need to create .env file with NFL_API_KEY"
            })
        
        # Check database models
        db_files = [
            "src/fantasy_ai/core/data/storage/models.py",
            "src/fantasy_ai/core/data/storage/simple_database.py"
        ]
        
        for db_file in db_files:
            if not (self.project_root / db_file).exists():
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Database",
                    "issue": f"Missing database file: {db_file}",
                    "action": "Need to share for database integration review"
                })
        
        return recommendations
    
    def print_analysis(self, analysis: Dict):
        """Print analysis results in a readable format."""
        print("\n🔍 PROJECT ANALYSIS RESULTS")
        print("=" * 80)
        
        # Print file analysis by category
        for category, data in analysis.items():
            if category in ["recommendations", "environment", "api_sports_integration"]:
                continue
                
            print(f"\n📁 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if data.get("files"):
                for file_path, file_info in data["files"].items():
                    status = "✅" if file_info.get("readable", False) else "❌"
                    size_kb = file_info.get("size", 0) / 1024
                    print(f"{status} {file_path} ({size_kb:.1f}KB)")
                    
                    # Show key patterns found
                    if file_info.get("api_patterns"):
                        print(f"    🔗 API patterns: {len(file_info['api_patterns'])} found")
                    if file_info.get("db_patterns"):
                        print(f"    🗄️  DB patterns: {len(file_info['db_patterns'])} found")
            
            if data.get("missing_files"):
                for missing_file in data["missing_files"]:
                    print(f"❌ {missing_file} (MISSING)")
        
        # Print API-Sports integration status
        print(f"\n🏈 API-SPORTS INTEGRATION STATUS")
        print("-" * 40)
        api_status = analysis.get("api_sports_integration", {})
        for key, value in api_status.items():
            if key != "issues":
                status = "✅" if value else "❌"
                print(f"{status} {key.replace('_', ' ').title()}: {value}")
        
        # Print environment status
        print(f"\n🐍 ENVIRONMENT STATUS")
        print("-" * 40)
        env = analysis.get("environment", {})
        print(f"Python: {env.get('python_version', 'Unknown')}")
        print(f"Virtual Environment: {'✅' if env.get('virtual_env') else '❌'}")
        print(f"Project Writable: {'✅' if env.get('project_root_writable') else '❌'}")
        
        if env.get("required_packages"):
            print("\nPackage Status:")
            for pkg, status in env["required_packages"].items():
                icon = "✅" if status == "installed" else "❌"
                print(f"  {icon} {pkg}: {status}")
        
        # Print recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n🎯 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                priority_icon = "🔴" if rec["priority"] == "HIGH" else "🟡"
                print(f"{priority_icon} {i}. {rec['category']}: {rec['issue']}")
                print(f"   Action: {rec['action']}")
                if rec.get("files"):
                    print(f"   Files: {', '.join(rec['files'])}")
                print()
    
    def generate_file_sharing_script(self, analysis: Dict) -> str:
        """Generate a script to share specific files with the AI."""
        files_to_share = []
        
        # Add files with API patterns
        for category, data in analysis.items():
            if category in ["recommendations", "environment", "api_sports_integration"]:
                continue
            
            if data.get("files"):
                for file_path, file_info in data["files"].items():
                    if (file_info.get("api_patterns") or 
                        file_info.get("db_patterns") or
                        "nfl" in file_path.lower() or
                        "etl" in file_path.lower()):
                        files_to_share.append(file_path)
        
        # Add missing critical files to the list
        for rec in analysis.get("recommendations", []):
            if rec.get("files"):
                files_to_share.extend(rec["files"])
        
        # Generate script
        script = f'''# Files to share with AI for Fantasy Football API integration fixes
# Copy and paste the contents of these files:

FILES_TO_SHARE = [
'''
        
        for file_path in sorted(set(files_to_share)):
            script += f'    "{file_path}",\n'
        
        script += ''']

# Run this to see file contents:
for file_path in FILES_TO_SHARE:
    print(f"\\n{'='*60}")
    print(f"FILE: {file_path}")
    print('='*60)
    try:
        with open(file_path, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("❌ FILE NOT FOUND")
    except Exception as e:
        print(f"❌ ERROR: {e}")
'''
        
        return script

def main():
    """Main function to run the project analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Fantasy Football AI project")
    parser.add_argument("--project-root", default=".", 
                       help="Path to project root directory")
    parser.add_argument("--output-script", default="share_files.py",
                       help="Output file for sharing script")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ProjectAnalyzer(args.project_root)
    analysis = analyzer.analyze_project()
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Generate file sharing script
    sharing_script = analyzer.generate_file_sharing_script(analysis)
    
    with open(args.output_script, 'w') as f:
        f.write(sharing_script)
    
    print(f"\n📝 File sharing script generated: {args.output_script}")
    print(f"Run: python {args.output_script}")
    print("\nThen copy the output and share with the AI for specific fixes!")

if __name__ == "__main__":
    main()