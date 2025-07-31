"""
LLM Service for Fantasy Football AI
Provides production-ready LLM capabilities with caching, rate limiting, and cost optimization
"""

import asyncio
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from openai import AsyncOpenAI
import tiktoken

from backend.core.cache import get_redis_client
from backend.models.database import get_db, User
from backend.core.rate_limiter import RateLimiter
from backend.services.subscription_service import SubscriptionService

logger = logging.getLogger(__name__)

class LLMConfig:
    """Configuration class for LLM service"""
    
    # Model configurations
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    
    # Cost per 1K tokens (approximate)
    TOKEN_COSTS = {
        GPT_3_5_TURBO: {"input": 0.0005, "output": 0.0015},
        GPT_4: {"input": 0.03, "output": 0.06},
        CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125},
        CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015}
    }
    
    # Response time targets
    MAX_RESPONSE_TIME = 10.0  # seconds
    TARGET_FIRST_TOKEN_TIME = 0.5  # seconds
    
    # Rate limits by subscription tier
    RATE_LIMITS = {
        "scout": {"requests_per_hour": 20, "tokens_per_day": 50000},
        "analyst": {"requests_per_hour": 1000, "tokens_per_day": 1000000},
        "gm": {"requests_per_hour": 5000, "tokens_per_day": 5000000}
    }


class TokenCounter:
    """Utility class for counting tokens"""
    
    def __init__(self):
        self.encoders = {}
    
    def get_encoder(self, model: str):
        """Get or create tiktoken encoder for model"""
        if model not in self.encoders:
            try:
                if "gpt" in model.lower():
                    self.encoders[model] = tiktoken.encoding_for_model(model)
                else:
                    # Fallback encoder for non-OpenAI models
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
            except KeyError:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using model-specific encoder"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))


class StreamingCallback:
    """Callback for streaming LLM responses"""
    
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.tokens = []
        self.start_time = None
        self.first_token_time = None
    
    async def on_llm_start(self, *args, **kwargs):
        """Called when LLM starts generating"""
        self.start_time = datetime.now()
    
    async def on_llm_new_token(self, token: str, **kwargs):
        """Called when a new token is generated"""
        if self.first_token_time is None:
            self.first_token_time = datetime.now()
        
        self.tokens.append(token)
        
        if self.websocket:
            await self.websocket.send_text(json.dumps({
                "type": "token",
                "data": token,
                "timestamp": datetime.now().isoformat()
            }))


class LLMService:
    """
    Production LLM service with caching, streaming, and cost optimization
    """
    
    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.token_counter = TokenCounter()
        self.redis_client = None
        self.subscription_service = SubscriptionService()
        
        # Initialize models
        self.openai_client = ChatOpenAI(
            model=LLMConfig.GPT_3_5_TURBO,
            openai_api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1000,
            streaming=True
        )
        
        self.anthropic_client = ChatAnthropic(
            model=LLMConfig.CLAUDE_3_HAIKU,
            anthropic_api_key=anthropic_api_key,
            temperature=0.1,
            max_tokens=1000,
            streaming=True
        )
        
        # Embeddings for vector search
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        logger.info("LLM Service initialized")
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await get_redis_client()
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any], model: str) -> str:
        """Generate cache key for query"""
        cache_input = f"{query}:{json.dumps(context, sort_keys=True)}:{model}"
        return f"llm_cache:{hashlib.md5(cache_input.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        try:
            if self.redis_client:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any], ttl: int = 3600):
        """Cache response with TTL"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(response)
                )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _check_rate_limits(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        try:
            user_tier = await self.subscription_service.get_user_tier(user_id)
            limits = LLMConfig.RATE_LIMITS.get(user_tier, LLMConfig.RATE_LIMITS["scout"])
            
            # Check hourly request limit
            hour_key = f"rate_limit:{user_id}:{datetime.now().strftime('%Y%m%d%H')}"
            current_requests = await self.redis_client.get(hour_key) or 0
            
            if int(current_requests) >= limits["requests_per_hour"]:
                return False
            
            # Check daily token limit
            day_key = f"token_limit:{user_id}:{datetime.now().strftime('%Y%m%d')}"
            current_tokens = await self.redis_client.get(day_key) or 0
            
            if int(current_tokens) >= limits["tokens_per_day"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    async def _update_usage_metrics(self, user_id: str, tokens_used: int):
        """Update usage metrics for user"""
        try:
            # Update hourly requests
            hour_key = f"rate_limit:{user_id}:{datetime.now().strftime('%Y%m%d%H')}"
            await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)
            
            # Update daily tokens
            day_key = f"token_limit:{user_id}:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_client.incrby(day_key, tokens_used)
            await self.redis_client.expire(day_key, 86400)
            
        except Exception as e:
            logger.error(f"Usage metrics update failed: {e}")
    
    def _select_model(self, query_type: str, complexity: str = "medium") -> tuple:
        """Select optimal model based on query type and complexity"""
        if query_type == "draft_assistant":
            if complexity == "high":
                return self.anthropic_client, LLMConfig.CLAUDE_3_SONNET
            else:
                return self.openai_client, LLMConfig.GPT_3_5_TURBO
        
        elif query_type == "analysis":
            return self.anthropic_client, LLMConfig.CLAUDE_3_SONNET
        
        else:
            return self.openai_client, LLMConfig.GPT_3_5_TURBO
    
    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        user_id: str,
        query_type: str = "general",
        complexity: str = "medium",
        stream: bool = False,
        websocket=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate LLM response with caching, rate limiting, and streaming
        """
        
        # Check rate limits
        if not await self._check_rate_limits(user_id):
            yield {
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED"
            }
            return
        
        # Select model
        model_client, model_name = self._select_model(query_type, complexity)
        
        # Check cache
        cache_key = self._generate_cache_key(query, context, model_name)
        cached_response = await self._get_cached_response(cache_key)
        
        if cached_response and not stream:
            yield cached_response
            return
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, context, query_type)
            
            # Count input tokens
            input_text = " ".join([msg.content for msg in messages])
            input_tokens = self.token_counter.count_tokens(input_text, model_name)
            
            # Setup streaming callback
            callback = StreamingCallback(websocket)
            
            start_time = datetime.now()
            
            if stream:
                # Streaming response
                async for chunk in model_client.astream(messages):
                    if hasattr(chunk, 'content'):
                        token = chunk.content
                        await callback.on_llm_new_token(token)
                        
                        yield {
                            "type": "token",
                            "content": token,
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Final response
                full_response = "".join(callback.tokens)
                output_tokens = self.token_counter.count_tokens(full_response, model_name)
                
                response_data = {
                    "type": "complete",
                    "content": full_response,
                    "model": model_name,
                    "tokens_used": input_tokens + output_tokens,
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "first_token_time": (callback.first_token_time - start_time).total_seconds() if callback.first_token_time else None
                }
                
                yield response_data
                
            else:
                # Non-streaming response
                response = await model_client.agenerate([messages])
                full_response = response.generations[0][0].text
                
                output_tokens = self.token_counter.count_tokens(full_response, model_name)
                response_time = (datetime.now() - start_time).total_seconds()
                
                response_data = {
                    "content": full_response,
                    "model": model_name,
                    "tokens_used": input_tokens + output_tokens,
                    "response_time": response_time,
                    "cached": False
                }
                
                # Cache response
                await self._cache_response(cache_key, response_data)
                
                yield response_data
            
            # Update usage metrics
            await self._update_usage_metrics(user_id, input_tokens + output_tokens)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            yield {
                "error": f"Generation failed: {str(e)}",
                "code": "GENERATION_ERROR"
            }
    
    def _prepare_messages(self, query: str, context: Dict[str, Any], query_type: str) -> List[BaseMessage]:
        """Prepare messages for LLM based on query type"""
        
        if query_type == "draft_assistant":
            system_prompt = self._get_draft_assistant_prompt(context)
        elif query_type == "analysis":
            system_prompt = self._get_analysis_prompt(context)
        else:
            system_prompt = "You are a helpful fantasy football assistant."
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
    
    def _get_draft_assistant_prompt(self, context: Dict[str, Any]) -> str:
        """Get draft assistant system prompt"""
        
        prompt = """You are an expert fantasy football draft assistant with access to advanced analytics.

Current Draft Context:
- Round: {round}
- Pick: {pick}
- Available Players: {available_players}
- Team Needs: {team_needs}
- League Settings: {league_settings}

Your expertise includes:
- Player tier classifications from Gaussian Mixture Models
- Neural network predictions with 93.1% accuracy
- Advanced metrics like efficiency ratios and momentum detection
- Draft strategy optimization

Provide concise, actionable advice. Always explain your reasoning and consider:
1. Player tiers and values
2. Positional scarcity
3. Team construction strategy
4. Risk/reward analysis

Keep responses under 150 words unless asked for detailed analysis."""

        return prompt.format(
            round=context.get("round", "N/A"),
            pick=context.get("pick", "N/A"), 
            available_players=context.get("available_players", "N/A"),
            team_needs=context.get("team_needs", "N/A"),
            league_settings=context.get("league_settings", "Standard")
        )
    
    def _get_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Get analysis system prompt"""
        
        return """You are an expert fantasy football analyst specializing in:

- Injury impact analysis
- Trade evaluation with multi-factor scoring
- Lineup optimization strategies
- News interpretation and fantasy relevance
- Advanced statistical analysis

Use data-driven insights and explain your methodology. Consider both immediate and long-term fantasy implications."""

    async def analyze_semantic_similarity(self, query: str, cached_queries: List[str]) -> Optional[str]:
        """Find semantically similar cached queries"""
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            
            # In a production system, you'd use a vector database here
            # For now, we'll implement a simple similarity check
            # This would be replaced with Weaviate or ChromaDB integration
            
            return None  # Placeholder for vector similarity search
            
        except Exception as e:
            logger.error(f"Semantic similarity check failed: {e}")
            return None