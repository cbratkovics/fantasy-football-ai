#!/usr/bin/env python3
"""
Test script for LLM integration
"""
import asyncio
import sys
import os
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

async def test_llm_integration():
    """Test basic LLM service functionality"""
    try:
        # Test imports
        print("Testing imports...")
        from backend.services.llm_service import LLMService
        from backend.services.vector_store import VectorStoreService
        from backend.services.subscription_service import SubscriptionService
        print("✓ All LLM services imported successfully")
        
        # Test LLM service initialization
        print("\nTesting LLM service initialization...")
        llm_service = LLMService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        await llm_service.initialize()
        print("✓ LLM service initialized successfully")
        
        # Test vector store initialization  
        print("\nTesting vector store initialization...")
        vector_store = VectorStoreService(openai_api_key=os.getenv("OPENAI_API_KEY"))
        await vector_store.initialize()
        print("✓ Vector store initialized successfully")
        
        # Test subscription service
        print("\nTesting subscription service...")
        sub_service = SubscriptionService()
        tier_features = sub_service.get_tier_features("analyst")
        print(f"✓ Subscription service working - Analyst tier features: {tier_features}")
        
        print("\n🎉 All LLM integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = asyncio.run(test_llm_integration())
    sys.exit(0 if success else 1)