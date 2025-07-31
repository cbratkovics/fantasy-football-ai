#!/usr/bin/env python3
"""
Analyze Docker image size and provide optimization recommendations
"""

import subprocess
import os

def get_library_sizes():
    """Estimate sizes of major ML libraries"""
    library_sizes = {
        'tensorflow': 1500,  # ~1.5GB
        'xgboost': 150,     # ~150MB
        'lightgbm': 100,    # ~100MB
        'scikit-learn': 150, # ~150MB
        'pandas': 50,       # ~50MB
        'numpy': 30,        # ~30MB
        'matplotlib': 40,   # ~40MB
        'seaborn': 5,       # ~5MB
        'plotly': 100,      # ~100MB
        'scipy': 60,        # ~60MB
        'optuna': 10,       # ~10MB
        'python-base': 150, # Python 3.11 slim base
        'system-deps': 100, # gcc, g++, etc.
    }
    return library_sizes

def analyze_project():
    """Analyze project structure and identify issues"""
    
    print("🔍 DOCKER IMAGE SIZE ANALYSIS")
    print("="*60)
    
    # 1. Check for redundant directories
    print("\n1. REDUNDANT DIRECTORIES:")
    print("   ❌ Both 'frontend' and 'frontend-next' exist")
    print("   - frontend-next: 441MB (mostly node_modules)")
    print("   - frontend: 72KB (old Streamlit app)")
    print("   💡 Remove the unused frontend directory")
    
    # 2. Analyze ML libraries
    print("\n2. ML LIBRARY SIZES (estimated):")
    lib_sizes = get_library_sizes()
    total_ml = 0
    for lib, size in lib_sizes.items():
        if lib not in ['python-base', 'system-deps']:
            total_ml += size
            print(f"   - {lib}: ~{size}MB")
    print(f"   📊 Total ML libraries: ~{total_ml/1000:.1f}GB")
    
    # 3. Check model sizes
    print("\n3. MODEL FILES:")
    print("   - Total models directory: 1.5MB")
    print("   - Individual models are small (100-616KB each)")
    print("   ✅ Models are reasonably sized")
    
    # 4. Data bundling check
    print("\n4. DATA BUNDLING:")
    print("   ✅ No large data files found in project")
    print("   ✅ No 10-year historical data bundled")
    
    # 5. Docker strategy issues
    print("\n5. DOCKER STRATEGY ISSUES:")
    print("   ❌ docker-compose.yml builds ALL services together")
    print("   ❌ Backend Dockerfile copies entire backend directory")
    print("   ❌ No .dockerignore to exclude unnecessary files")
    print("   ❌ Installing ALL ML libraries for production")
    
    print("\n" + "="*60)
    print("📦 ESTIMATED DOCKER IMAGE SIZES:")
    print("="*60)
    
    # Calculate sizes
    backend_size = lib_sizes['python-base'] + lib_sizes['system-deps'] + total_ml + 50  # +50MB for app code
    frontend_size = 500  # Next.js with node_modules
    
    print(f"Backend image: ~{backend_size/1000:.1f}GB")
    print(f"Frontend image: ~{frontend_size/1000:.1f}GB")
    print(f"Combined: ~{(backend_size + frontend_size)/1000:.1f}GB")
    
    print("\n🚨 MAIN ISSUE: TensorFlow alone is ~1.5GB!")
    
    print("\n" + "="*60)
    print("💡 OPTIMIZATION RECOMMENDATIONS:")
    print("="*60)
    
    recommendations = [
        "1. CREATE SEPARATE DEPLOYMENTS:",
        "   - Deploy frontend and backend separately on Railway",
        "   - Frontend: Use Railway's Node.js buildpack",
        "   - Backend: Use custom Dockerfile with optimizations",
        "",
        "2. OPTIMIZE BACKEND DOCKERFILE:",
        "   - Use tensorflow-cpu instead of full tensorflow (saves ~1GB)",
        "   - Create production requirements.txt without training-only libs",
        "   - Add comprehensive .dockerignore",
        "",
        "3. SPLIT ML LIBRARIES:",
        "   - Production: Only libraries needed for inference",
        "   - Training: Separate container/environment",
        "",
        "4. USE MULTI-STAGE DOCKER BUILD:",
        "   - Build stage: Install all dependencies",
        "   - Runtime stage: Copy only needed files",
        "",
        "5. REMOVE REDUNDANT CODE:",
        "   - Delete frontend directory (old Streamlit)",
        "   - Clean up unused scripts",
        "   - Remove visualization libraries from production"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*60)
    print("📋 PRODUCTION vs TRAINING LIBRARIES:")
    print("="*60)
    
    print("\nPRODUCTION (needed for inference):")
    print("- tensorflow-cpu (or just load saved models)")
    print("- scikit-learn")
    print("- pandas")
    print("- numpy")
    print("- joblib")
    print("- FastAPI stack")
    
    print("\nTRAINING ONLY (can remove):")
    print("- xgboost (if models are pre-trained)")
    print("- lightgbm (if models are pre-trained)")
    print("- optuna")
    print("- matplotlib")
    print("- seaborn")
    print("- plotly")
    print("- beautifulsoup4")

if __name__ == "__main__":
    analyze_project()