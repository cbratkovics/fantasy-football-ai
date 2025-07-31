#!/usr/bin/env python3
import requests
import time

def quick_test():
    """Quick test of running server"""
    base_url = "http://localhost:8000"
    
    for i in range(3):
        try:
            print(f"Attempt {i+1}: Testing health endpoint...")
            response = requests.get(f"{base_url}/health", timeout=10)
            print(f"✓ Health: {response.status_code}")
            
            print(f"Attempt {i+1}: Testing LLM health endpoint...")
            response = requests.get(f"{base_url}/api/llm/health", timeout=30)
            print(f"✓ LLM Health: {response.status_code} - {response.json()}")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {i+1} failed: {e}")
            time.sleep(5)
    
    return False

if __name__ == "__main__":
    success = quick_test()
    print("✅ SUCCESS" if success else "❌ FAILED")