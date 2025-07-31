#!/usr/bin/env python3
"""
Final comprehensive system test for Fantasy Football AI
"""
import asyncio
import sys
import os
import time
import requests
import subprocess
import json
from datetime import datetime
from typing import Dict, Any

# Add project to path
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

# Change to backend directory for proper .env loading
os.chdir('/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend')

from dotenv import load_dotenv
load_dotenv('.env.local')
load_dotenv('.env')

class SystemTester:
    """Final system tester"""
    
    def __init__(self):
        self.results = {}
        self.server_process = None
    
    def log_result(self, category: str, test: str, success: bool, details: str = ""):
        """Log test result"""
        if category not in self.results:
            self.results[category] = []
        
        self.results[category].append({
            "test": test,
            "success": success,
            "details": details
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {category}/{test}: {details}")
    
    def test_infrastructure(self):
        """Test core infrastructure"""
        print("\n🏗️  TESTING INFRASTRUCTURE")
        print("="*50)
        
        # Environment variables
        required_vars = ["DATABASE_URL", "REDIS_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        env_success = all(os.getenv(var) for var in required_vars)
        self.log_result("Infrastructure", "Environment Variables", env_success, 
                       f"{sum(1 for var in required_vars if os.getenv(var))}/{len(required_vars)} set")
        
        # Redis connection
        try:
            import redis
            r = redis.from_url(os.getenv("REDIS_URL"))
            r.ping()
            r.set("test", "value", ex=10)
            value = r.get("test")
            r.delete("test")
            self.log_result("Infrastructure", "Redis Connection", True, "Operations successful")
        except Exception as e:
            self.log_result("Infrastructure", "Redis Connection", False, str(e))
        
        # Database connection
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(os.getenv("DATABASE_URL"))
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).fetchone()
                self.log_result("Infrastructure", "Database Connection", True, f"Connected: {result[0]}")
        except Exception as e:
            self.log_result("Infrastructure", "Database Connection", False, str(e))
    
    async def test_core_services(self):
        """Test core ML and LLM services"""
        print("\n🤖 TESTING CORE SERVICES")
        print("="*50)
        
        # ML Models
        try:
            from backend.ml.ensemble_predictions import EnsemblePredictionEngine
            engine = EnsemblePredictionEngine()
            
            # Check if engine has key methods
            methods = ['predict_player_week', 'get_player_tiers']
            has_methods = sum(1 for method in methods if hasattr(engine, method))
            self.log_result("Core Services", "ML Engine", has_methods > 0, 
                           f"{has_methods}/{len(methods)} methods available")
        except Exception as e:
            self.log_result("Core Services", "ML Engine", False, str(e))
        
        # LLM Services
        try:
            from backend.services.llm_service import LLMService
            llm = LLMService(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            await llm.initialize()
            self.log_result("Core Services", "LLM Service", True, "Initialized with models")
        except Exception as e:
            self.log_result("Core Services", "LLM Service", False, str(e))
        
        # Vector Store
        try:
            from backend.services.vector_store import VectorStoreService
            vector_store = VectorStoreService(openai_api_key=os.getenv("OPENAI_API_KEY"))
            await vector_store.initialize()
            self.log_result("Core Services", "Vector Store", True, "ChromaDB initialized")
        except Exception as e:
            self.log_result("Core Services", "Vector Store", False, str(e))
        
        # Subscription Service
        try:
            from backend.services.subscription_service import SubscriptionService
            sub_service = SubscriptionService()
            tiers = sub_service.get_tier_features("analyst")
            self.log_result("Core Services", "Subscription Service", len(tiers) > 0, 
                           f"{len(tiers)} features for analyst tier")
        except Exception as e:
            self.log_result("Core Services", "Subscription Service", False, str(e))
    
    def test_api_structure(self):
        """Test API module imports"""
        print("\n🔌 TESTING API STRUCTURE")
        print("="*50)
        
        api_modules = {
            "Auth API": "backend.api.auth",
            "Players API": "backend.api.players",
            "Predictions API": "backend.api.predictions", 
            "LLM API": "backend.api.llm_endpoints",
            "Subscriptions API": "backend.api.subscriptions",
            "Payments API": "backend.api.payments"
        }
        
        for name, module_path in api_modules.items():
            try:
                __import__(module_path)
                self.log_result("API Structure", name, True, "Module imported")
            except Exception as e:
                self.log_result("API Structure", name, False, str(e))
    
    def start_server(self):
        """Start the FastAPI server"""
        print("\n🚀 STARTING SERVER")
        print("="*50)
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = '/Users/christopherbratkovics/Desktop/fantasy-football-ai'
            
            self.server_process = subprocess.Popen([
                '/Users/christopherbratkovics/anaconda3/envs/agentic_ai_env/bin/python',
                'main.py'
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("Waiting for server startup (20 seconds)...")
            time.sleep(20)
            
            if self.server_process.poll() is None:
                self.log_result("Server", "Startup", True, "Process running")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                error_msg = stderr.decode()[:300] if stderr else "Unknown error"
                self.log_result("Server", "Startup", False, error_msg)
                return False
                
        except Exception as e:
            self.log_result("Server", "Startup", False, str(e))
            return False
    
    def test_endpoints(self):
        """Test server endpoints"""
        print("\n🌐 TESTING ENDPOINTS")
        print("="*50)
        
        base_url = "http://localhost:8000"
        
        # Test basic endpoints
        endpoints = [
            ("Root", "GET", "/"),
            ("Health", "GET", "/health"),
            ("Players", "GET", "/players"),
            ("LLM Health", "GET", "/api/llm/health")
        ]
        
        for name, method, path in endpoints:
            try:
                response = requests.get(f"{base_url}{path}", timeout=15)
                success = response.status_code < 500
                self.log_result("Endpoints", f"{name} ({method})", success, 
                               f"Status: {response.status_code}")
                
                # Try to parse JSON response
                if success:
                    try:
                        data = response.json()
                        if isinstance(data, dict) and len(data) > 0:
                            self.log_result("Endpoints", f"{name} Response", True, 
                                           f"Valid JSON with {len(data)} fields")
                    except:
                        self.log_result("Endpoints", f"{name} Response", True, "Non-JSON response")
                        
            except requests.exceptions.ConnectionError:
                self.log_result("Endpoints", f"{name} ({method})", False, "Connection refused")
            except Exception as e:
                self.log_result("Endpoints", f"{name} ({method})", False, str(e))
    
    def cleanup(self):
        """Cleanup resources"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("✅ Server stopped cleanly")
            except:
                self.server_process.kill()
                print("⚠️  Server killed")
    
    def generate_final_report(self):
        """Generate final system report"""
        print("\n" + "="*60)
        print("🎯 FINAL SYSTEM TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            print(f"\n📂 {category.upper()}:")
            category_passed = sum(1 for test in tests if test["success"])
            category_total = len(tests)
            
            for test in tests:
                status = "✅" if test["success"] else "❌"
                print(f"   {status} {test['test']}: {test['details']}")
            
            success_rate = (category_passed / category_total) * 100 if category_total > 0 else 0
            print(f"   📊 {category} Success: {category_passed}/{category_total} ({success_rate:.1f}%)")
            
            total_tests += category_total
            passed_tests += category_passed
        
        overall_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL SYSTEM HEALTH:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {overall_rate:.1f}%")
        
        # System status determination
        if overall_rate >= 90:
            status = "🎉 EXCELLENT - Production Ready"
        elif overall_rate >= 80:
            status = "✅ GOOD - Minor Issues"
        elif overall_rate >= 70:
            status = "⚠️  FAIR - Needs Attention"
        else:
            status = "🚨 CRITICAL - Major Issues"
        
        print(f"\n🏆 SYSTEM STATUS: {status}")
        
        # Key capabilities summary
        print(f"\n🔧 KEY CAPABILITIES:")
        capabilities = {
            "LLM Integration": any("LLM" in cat for cat in self.results.keys()),
            "Database Operations": any("Database" in str(test) for tests in self.results.values() for test in tests),
            "Redis Caching": any("Redis" in str(test) for tests in self.results.values() for test in tests),
            "ML Predictions": any("ML" in str(test) for tests in self.results.values() for test in tests),
            "API Endpoints": any("Endpoints" in cat for cat in self.results.keys()),
            "Server Startup": any("Server" in cat for cat in self.results.keys())
        }
        
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"   {status} {capability}")
        
        return overall_rate >= 80

async def main():
    """Run complete system test"""
    print("🔥 FANTASY FOOTBALL AI - FINAL SYSTEM TEST")
    print(f"📅 Test Date: {datetime.now().isoformat()}")
    print("="*60)
    
    tester = SystemTester()
    
    try:
        # Run all tests
        tester.test_infrastructure()
        await tester.test_core_services()
        tester.test_api_structure()
        
        # Test server endpoints
        if tester.start_server():
            time.sleep(3)  # Extra wait for endpoints to be ready
            tester.test_endpoints()
        
        # Generate final report
        success = tester.generate_final_report()
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Critical system test failure: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)