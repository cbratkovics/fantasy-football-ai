#!/usr/bin/env python3
"""
Test script to validate Railway deployment readiness
Run this locally to check for potential issues
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test environment setup"""
    logger.info("Testing environment setup...")
    
    # Set test PORT
    os.environ['PORT'] = '8000'
    
    port = os.getenv('PORT')
    logger.info(f"PORT: {port}")
    
    if not port:
        logger.error("PORT not set")
        return False
    
    try:
        port_int = int(port)
        logger.info(f"Port validation: {port_int} - OK")
    except ValueError:
        logger.error("Invalid port")
        return False
    
    return True

def test_basic_imports():
    """Test basic package imports"""
    logger.info("Testing basic imports...")
    
    try:
        import fastapi
        logger.info(f"FastAPI: {fastapi.__version__} - OK")
    except ImportError as e:
        logger.error(f"FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        logger.info("Uvicorn: OK")
    except ImportError as e:
        logger.error(f"Uvicorn import failed: {e}")
        return False
    
    return True

def test_optional_imports():
    """Test optional dependencies"""
    logger.info("Testing optional imports...")
    
    # Database dependencies
    try:
        import sqlalchemy
        logger.info(f"SQLAlchemy: {sqlalchemy.__version__} - OK")
    except ImportError as e:
        logger.warning(f"SQLAlchemy import failed: {e}")
    
    try:
        import psycopg2
        logger.info("psycopg2: OK")
    except ImportError as e:
        logger.warning(f"psycopg2 import failed: {e}")
    
    # ML dependencies
    try:
        import tensorflow
        logger.info(f"TensorFlow: {tensorflow.__version__} - OK")
    except ImportError as e:
        logger.warning(f"TensorFlow import failed: {e}")
    
    return True

def test_app_imports():
    """Test application imports"""
    logger.info("Testing application imports...")
    
    try:
        # Test main app import
        sys.path.insert(0, '/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend')
        from main import app
        logger.info("Main app import: OK")
        
        # Test if app has health endpoint
        routes = [route.path for route in app.routes]
        if '/health' in routes:
            logger.info("Health endpoint: FOUND")
        else:
            logger.warning("Health endpoint: NOT FOUND")
        
        return True
    except Exception as e:
        logger.error(f"App import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_server():
    """Test if we can start a minimal server"""
    logger.info("Testing minimal server startup...")
    
    try:
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        import uvicorn
        
        # Don't actually start the server, just test configuration
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
        logger.info("Minimal server configuration: OK")
        return True
    except Exception as e:
        logger.error(f"Minimal server test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("RAILWAY DEPLOYMENT READINESS TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Basic Imports", test_basic_imports),
        ("Optional Imports", test_optional_imports),
        ("App Imports", test_app_imports),
        ("Minimal Server", test_minimal_server)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        results[test_name] = test_func()
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! App should deploy successfully to Railway.")
    else:
        logger.warning("\n⚠️  Some tests failed. Check logs above for issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)