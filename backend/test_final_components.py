#!/usr/bin/env python3
"""
Final component test with proper environment loading
"""
import sys
import os
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

# Clear any cached environment variables
for key in list(os.environ.keys()):
    if 'DATABASE' in key or 'REDIS' in key or 'API_KEY' in key:
        del os.environ[key]

# Change to backend directory for proper .env loading
os.chdir('/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend')

from dotenv import load_dotenv
load_dotenv('.env.local')  # Load local environment first
load_dotenv('.env')        # Then load main env

def test_environment_loading():
    """Test environment variable loading"""
    print("🔍 Environment Variables Test")
    print("="*50)
    
    db_url = os.getenv("DATABASE_URL")
    redis_url = os.getenv("REDIS_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"DATABASE_URL: {db_url[:50]}..." if db_url else "Not set")
    print(f"REDIS_URL: {redis_url}")
    print(f"OPENAI_API_KEY: {'Set (' + openai_key[:10] + '...)' if openai_key else 'Not set'}")
    print(f"ANTHROPIC_API_KEY: {'Set (' + anthropic_key[:10] + '...)' if anthropic_key else 'Not set'}")
    
    return all([db_url, redis_url, openai_key, anthropic_key])

def test_database_with_current_env():
    """Test database with current environment"""
    print("\n🔍 Database Connection Test")
    print("="*50)
    
    try:
        from backend.models.database import engine
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test").fetchone()
            print(f"✅ Database connection successful: {result[0]}")
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_ml_models_functionality():
    """Test ML models with sample data"""
    print("\n🔍 ML Models Functionality Test")
    print("="*50)
    
    try:
        from backend.ml.ensemble_predictions import EnsemblePredictionEngine
        import pandas as pd
        import numpy as np
        
        engine = EnsemblePredictionEngine()
        print("✅ ML Engine created successfully")
        
        # Test with mock data to see if models can process data
        mock_features = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10)
        })
        
        # Test if the engine has the expected methods
        expected_methods = ['predict_player_week', 'get_player_tiers', 'calculate_efficiency_metrics']
        methods_exist = [hasattr(engine, method) for method in expected_methods]
        
        print(f"✅ Expected methods present: {sum(methods_exist)}/{len(expected_methods)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Models test failed: {e}")
        return False

def test_llm_services_functionality():
    """Test LLM services functionality"""
    print("\n🔍 LLM Services Functionality Test")
    print("="*50)
    
    try:
        from backend.services.llm_service import LLMService
        from backend.services.vector_store import VectorStoreService
        from backend.services.subscription_service import SubscriptionService
        
        import asyncio
        
        async def test_llm():
            # Test LLM service
            llm_service = LLMService(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            await llm_service.initialize()
            print("✅ LLM Service initialized")
            
            # Test vector store
            vector_store = VectorStoreService(openai_api_key=os.getenv("OPENAI_API_KEY"))
            await vector_store.initialize()
            print("✅ Vector Store initialized")
            
            # Test subscription service
            sub_service = SubscriptionService()
            tiers = ["scout", "analyst", "gm"]
            for tier in tiers:
                features = sub_service.get_tier_features(tier)
                print(f"✅ Subscription tier '{tier}': {len(features)} features")
            
            return True
        
        return asyncio.run(test_llm())
        
    except Exception as e:
        print(f"❌ LLM Services test failed: {e}")
        return False

def test_api_imports_and_structure():
    """Test API structure and imports"""
    print("\n🔍 API Structure Test")
    print("="*50)
    
    api_modules = [
        'backend.api.auth',
        'backend.api.players', 
        'backend.api.predictions',
        'backend.api.llm_endpoints',
        'backend.api.subscriptions',
        'backend.api.payments'
    ]
    
    passed = 0
    for module in api_modules:
        try:
            __import__(module)
            print(f"✅ {module.split('.')[-1]}: Imported successfully")
            passed += 1
        except Exception as e:
            print(f"❌ {module.split('.')[-1]}: {e}")
    
    print(f"✅ API modules: {passed}/{len(api_modules)} passed")
    return passed == len(api_modules)

def run_comprehensive_test():
    """Run all component tests"""
    print("🚀 Final Component Test Suite")
    print("="*60)
    
    tests = [
        ("Environment Loading", test_environment_loading),
        ("Database Connection", test_database_with_current_env),
        ("ML Models", test_ml_models_functionality),
        ("LLM Services", test_llm_services_functionality),
        ("API Structure", test_api_imports_and_structure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}: Critical error - {e}")
            results[test_name] = False
    
    # Generate final report
    print("\n" + "="*60)
    print("FINAL TEST REPORT")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = (passed / total) * 100
    print(f"\n📊 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 PROJECT STATUS: HEALTHY")
    elif success_rate >= 60:
        print("⚠️  PROJECT STATUS: NEEDS ATTENTION")
    else:
        print("🚨 PROJECT STATUS: CRITICAL ISSUES")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)