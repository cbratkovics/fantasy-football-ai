#!/usr/bin/env python3
"""
Simple test script to verify LLM integration status
"""
import sys
import os
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_server_imports():
    """Test what the server can import"""
    try:
        print("Testing server imports...")
        
        # Configure logging first
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Try to import LLM components like the server does
        try:
            from backend.api import llm_endpoints
            from backend.services.llm_service import LLMService
            from backend.services.vector_store import VectorStoreService
            LLM_AVAILABLE = True
            print("✓ LLM services available for server")
        except ImportError as e:
            print(f"❌ LLM services not available: {e}")
            LLM_AVAILABLE = False
        
        print(f"LLM_AVAILABLE = {LLM_AVAILABLE}")
        
        if LLM_AVAILABLE:
            print("✓ Server should include LLM endpoints at /api/llm/")
        else:
            print("❌ Server will NOT include LLM endpoints")
            
        return LLM_AVAILABLE
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_server_imports()
    print(f"\nLLM Integration Status: {'✅ READY' if success else '❌ NOT READY'}")