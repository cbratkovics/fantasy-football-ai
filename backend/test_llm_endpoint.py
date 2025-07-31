#!/usr/bin/env python3
"""
Test script for LLM endpoints
"""
import requests
import json
import time
import subprocess
import signal
import os
import sys
from threading import Thread

def start_server():
    """Start the FastAPI server in the background"""
    print("Starting FastAPI server...")
    
    # Start server process
    env = os.environ.copy()
    env['PYTHONPATH'] = '/Users/christopherbratkovics/Desktop/fantasy-football-ai'
    
    process = subprocess.Popen([
        '/Users/christopherbratkovics/anaconda3/envs/agentic_ai_env/bin/python',
        'main.py'
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Give server time to start
    time.sleep(10)
    
    return process

def test_endpoints():
    """Test LLM endpoints"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test LLM health endpoint  
    try:
        response = requests.get(f"{base_url}/api/llm/health", timeout=5)
        print(f"✓ LLM health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ LLM health check failed: {e}")
        return False
    
    print("🎉 Basic endpoint tests passed!")
    return True

if __name__ == "__main__":
    # Change to backend directory
    os.chdir('/Users/christopherbratkovics/Desktop/fantasy-football-ai/backend')
    
    # Start server
    server_process = None
    try:
        server_process = start_server()
        
        # Test endpoints
        success = test_endpoints()
        
        print("\n" + "="*50)
        if success:
            print("✅ All LLM endpoint tests PASSED!")
        else:
            print("❌ Some LLM endpoint tests FAILED!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Cleanup
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait()
    
    sys.exit(0 if success else 1)