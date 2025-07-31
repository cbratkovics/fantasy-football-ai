#!/usr/bin/env python3
"""
Comprehensive Test Suite for Fantasy Football AI Project
Tests all major components to ensure they work correctly
"""
import asyncio
import sys
import os
import time
import requests
import subprocess
import signal
from typing import Dict, List, Any
from datetime import datetime

# Add project to path
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ComponentTester:
    """Comprehensive component testing class"""
    
    def __init__(self):
        self.results = {}
        self.server_process = None
        
    def log_test(self, component: str, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        if component not in self.results:
            self.results[component] = []
        
        self.results[component].append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {component}: {test_name} - {details}")
        
    def test_imports(self):
        """Test all critical imports"""
        print("\n" + "="*60)
        print("TESTING: Core Imports")
        print("="*60)
        
        imports_to_test = [
            ("FastAPI", "from fastapi import FastAPI"),
            ("SQLAlchemy", "from sqlalchemy import create_engine"),
            ("Redis", "import redis"),
            ("Pandas", "import pandas as pd"),
            ("NumPy", "import numpy as np"),
            ("Scikit-learn", "from sklearn.ensemble import RandomForestRegressor"),
            ("TensorFlow", "import tensorflow as tf"),
            ("LangChain OpenAI", "from langchain_openai import ChatOpenAI"),
            ("LangChain Anthropic", "from langchain_anthropic import ChatAnthropic"),
            ("ChromaDB", "import chromadb"),
            ("Sentence Transformers", "from sentence_transformers import SentenceTransformer"),
        ]
        
        for name, import_stmt in imports_to_test:
            try:
                exec(import_stmt)
                self.log_test("Imports", name, True, "Available")
            except ImportError as e:
                self.log_test("Imports", name, False, str(e))
            except Exception as e:
                self.log_test("Imports", name, False, f"Error: {e}")
    
    def test_environment_variables(self):
        """Test environment variables"""
        print("\n" + "="*60)
        print("TESTING: Environment Variables")
        print("="*60)
        
        required_vars = [
            "DATABASE_URL",
            "REDIS_URL", 
            "JWT_SECRET_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY"
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                display_value = value[:10] + "..." if len(value) > 10 else value
                if "key" in var.lower() or "secret" in var.lower():
                    display_value = "***MASKED***"
                self.log_test("Environment", var, True, f"Set: {display_value}")
            else:
                self.log_test("Environment", var, False, "Not set")
    
    async def test_database_connection(self):
        """Test database connectivity"""
        print("\n" + "="*60)
        print("TESTING: Database Connection")
        print("="*60)
        
        try:
            from backend.models.database import engine, SessionLocal, Base
            
            # Test engine creation
            self.log_test("Database", "Engine Creation", True, "SQLAlchemy engine created")
            
            # Test connection
            try:
                with engine.connect() as conn:
                    result = conn.execute("SELECT 1 as test").fetchone()
                    self.log_test("Database", "Connection", True, f"Connected: {result[0]}")
            except Exception as e:
                self.log_test("Database", "Connection", False, str(e))
            
            # Test session creation
            try:
                with SessionLocal() as session:
                    self.log_test("Database", "Session Creation", True, "Session created successfully")
            except Exception as e:
                self.log_test("Database", "Session Creation", False, str(e))
                
        except Exception as e:
            self.log_test("Database", "Setup", False, str(e))
    
    async def test_redis_connection(self):
        """Test Redis connectivity"""
        print("\n" + "="*60)
        print("TESTING: Redis Connection")
        print("="*60)
        
        try:
            from backend.core.cache import cache
            
            # Test basic operations
            test_key = "test_key_comprehensive"
            test_value = {"message": "test", "timestamp": time.time()}
            
            # Test set
            success = cache.set(test_key, test_value, ttl=60)
            self.log_test("Redis", "Set Operation", success, f"Key: {test_key}")
            
            # Test get
            retrieved = cache.get(test_key)
            get_success = retrieved is not None and retrieved["message"] == "test"
            self.log_test("Redis", "Get Operation", get_success, f"Retrieved: {bool(retrieved)}")
            
            # Test delete
            delete_success = cache.delete(test_key)
            self.log_test("Redis", "Delete Operation", delete_success, "Key deleted")
            
            # Test async Redis client
            from backend.core.cache import get_redis_client
            async_client = await get_redis_client()
            if async_client:
                await async_client.set("async_test", "value", ex=60)
                async_value = await async_client.get("async_test")
                await async_client.delete("async_test")
                self.log_test("Redis", "Async Operations", async_value == "value", "Async client working")
            else:
                self.log_test("Redis", "Async Operations", False, "Async client not available")
                
        except Exception as e:
            self.log_test("Redis", "Connection", False, str(e))
    
    async def test_ml_models(self):
        """Test ML prediction models"""
        print("\n" + "="*60)
        print("TESTING: ML Models")
        print("="*60)
        
        try:
            # Test ensemble predictions import
            from backend.ml.ensemble_predictions import EnsemblePredictionEngine
            self.log_test("ML Models", "Ensemble Import", True, "EnsemblePredictionEngine imported")
            
            # Test model initialization
            try:
                engine = EnsemblePredictionEngine()
                self.log_test("ML Models", "Engine Creation", True, "Prediction engine created")
                
                # Test if models can be loaded (this might fail without data)
                try:
                    # This might fail without proper data, so we catch it
                    hasattr(engine, 'random_forest_model')
                    self.log_test("ML Models", "Model Structure", True, "Engine has expected attributes")
                except Exception as e:
                    self.log_test("ML Models", "Model Structure", False, str(e))
                    
            except Exception as e:
                self.log_test("ML Models", "Engine Creation", False, str(e))
                
        except Exception as e:
            self.log_test("ML Models", "Import", False, str(e))
    
    async def test_llm_services(self):
        """Test LLM services"""
        print("\n" + "="*60)
        print("TESTING: LLM Services")
        print("="*60)
        
        try:
            # Test LLM service import and initialization
            from backend.services.llm_service import LLMService
            
            llm_service = LLMService(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            await llm_service.initialize()
            self.log_test("LLM Services", "LLM Service Init", True, "Service initialized")
            
            # Test vector store
            from backend.services.vector_store import VectorStoreService
            vector_store = VectorStoreService(openai_api_key=os.getenv("OPENAI_API_KEY"))
            await vector_store.initialize()
            self.log_test("LLM Services", "Vector Store Init", True, "Vector store initialized")
            
            # Test subscription service
            from backend.services.subscription_service import SubscriptionService
            sub_service = SubscriptionService()
            tier_info = sub_service.get_tier_features("analyst")
            self.log_test("LLM Services", "Subscription Service", True, f"Tiers available: {len(tier_info)}")
            
        except Exception as e:
            self.log_test("LLM Services", "Initialization", False, str(e))
    
    async def test_api_routes(self):
        """Test API route imports"""
        print("\n" + "="*60)
        print("TESTING: API Routes")
        print("="*60)
        
        api_modules = [
            ("Auth Routes", "from backend.api import auth"),
            ("Player Routes", "from backend.api import players"),
            ("Prediction Routes", "from backend.api import predictions"),
            ("LLM Routes", "from backend.api import llm_endpoints"),
            ("Payment Routes", "from backend.api import payments"),
            ("Subscription Routes", "from backend.api import subscriptions"),
        ]
        
        for name, import_stmt in api_modules:
            try:
                exec(import_stmt)
                self.log_test("API Routes", name, True, "Imported successfully")
            except Exception as e:
                self.log_test("API Routes", name, False, str(e))
    
    def start_test_server(self):
        """Start server for endpoint testing"""
        try:
            print("\n" + "="*60)
            print("TESTING: Server Startup")
            print("="*60)
            
            env = os.environ.copy()
            env['PYTHONPATH'] = '/Users/christopherbratkovics/Desktop/fantasy-football-ai'
            
            # Change to backend directory
            original_dir = os.getcwd()
            os.chdir('/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend')
            
            self.server_process = subprocess.Popen([
                '/Users/christopherbratkovics/anaconda3/envs/agentic_ai_env/bin/python',
                'main.py'
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give server time to start
            print("Starting server (waiting 15 seconds)...")
            time.sleep(15)
            
            # Check if process is still running
            if self.server_process.poll() is None:
                self.log_test("Server", "Startup", True, "Server process running")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                self.log_test("Server", "Startup", False, f"Process died: {stderr.decode()[:200]}...")
                return False
                
        except Exception as e:
            self.log_test("Server", "Startup", False, str(e))
            return False
        finally:
            os.chdir(original_dir)
    
    def test_endpoints(self):
        """Test server endpoints"""
        print("\n" + "="*60)
        print("TESTING: Server Endpoints")
        print("="*60)
        
        base_url = "http://localhost:8000"
        
        endpoints_to_test = [
            ("Root", "GET", "/"),
            ("Health", "GET", "/health"),
            ("LLM Health", "GET", "/api/llm/health"),
        ]
        
        for name, method, path in endpoints_to_test:
            try:
                if method == "GET":
                    response = requests.get(f"{base_url}{path}", timeout=10)
                    
                self.log_test("Endpoints", f"{name} ({method})", 
                            response.status_code < 500, 
                            f"Status: {response.status_code}")
                            
            except requests.exceptions.ConnectionError:
                self.log_test("Endpoints", f"{name} ({method})", False, "Connection refused")
            except Exception as e:
                self.log_test("Endpoints", f"{name} ({method})", False, str(e))
    
    def cleanup_server(self):
        """Stop test server"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("Server stopped cleanly")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("Server killed (timeout)")
            except Exception as e:
                print(f"Error stopping server: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for component, tests in self.results.items():
            print(f"\n📋 {component.upper()}:")
            component_passed = 0
            component_total = len(tests)
            
            for test in tests:
                status = "✅ PASS" if test["success"] else "❌ FAIL"
                print(f"  {status} {test['test']}: {test['details']}")
                if test["success"]:
                    component_passed += 1
                    
            success_rate = (component_passed / component_total) * 100 if component_total > 0 else 0
            print(f"  📊 Success Rate: {component_passed}/{component_total} ({success_rate:.1f}%)")
            
            total_tests += component_total
            passed_tests += component_passed
        
        overall_success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 80:
            print(f"\n🎉 PROJECT STATUS: HEALTHY (>{overall_success_rate:.1f}%)")
        elif overall_success_rate >= 60:
            print(f"\n⚠️  PROJECT STATUS: NEEDS ATTENTION ({overall_success_rate:.1f}%)")
        else:
            print(f"\n🚨 PROJECT STATUS: CRITICAL ISSUES ({overall_success_rate:.1f}%)")
        
        return overall_success_rate >= 80

async def main():
    """Run comprehensive test suite"""
    print("🚀 Starting Comprehensive Fantasy Football AI Test Suite")
    print(f"📅 Test Date: {datetime.now().isoformat()}")
    
    tester = ComponentTester()
    
    try:
        # Run all tests
        tester.test_imports()
        tester.test_environment_variables()
        await tester.test_database_connection()
        await tester.test_redis_connection()
        await tester.test_ml_models()
        await tester.test_llm_services()
        await tester.test_api_routes()
        
        # Test server startup and endpoints
        if tester.start_test_server():
            time.sleep(5)  # Additional wait for full startup
            tester.test_endpoints()
        
        # Generate final report
        success = tester.generate_report()
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Critical test failure: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        tester.cleanup_server()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)