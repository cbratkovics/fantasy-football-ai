"""
LLM-powered API endpoints for Fantasy Football AI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.models.database import get_db, User, Player, DraftTier, Prediction
from backend.services.llm_service import LLMService, LLMConfig
from backend.services.subscription_service import SubscriptionService
from backend.api.auth import get_current_user
from backend.core.cache import get_redis_client

logger = logging.getLogger(__name__)

# Initialize services
llm_service = None
subscription_service = SubscriptionService()

# Pydantic models for API
class DraftAssistantRequest(BaseModel):
    query: str = Field(..., description="User's draft question")
    draft_context: Dict[str, Any] = Field(default_factory=dict, description="Current draft state")
    complexity: str = Field(default="medium", description="Query complexity: low, medium, high")
    stream: bool = Field(default=False, description="Enable streaming response")

class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Analysis question")
    player_ids: Optional[List[str]] = Field(default=None, description="Relevant player IDs")
    analysis_type: str = Field(default="general", description="Type of analysis")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class TradeAnalysisRequest(BaseModel):
    giving_players: List[str] = Field(..., description="Player IDs being traded away")  
    receiving_players: List[str] = Field(..., description="Player IDs being received")
    league_context: Dict[str, Any] = Field(default_factory=dict, description="League settings and context")
    team_needs: List[str] = Field(default_factory=list, description="Current team needs")

class LineupOptimizationRequest(BaseModel):
    available_players: List[str] = Field(..., description="Available player IDs")
    lineup_slots: List[str] = Field(..., description="Required lineup positions")
    week: int = Field(..., description="Week number")
    scoring_format: str = Field(default="ppr", description="Scoring format")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Additional constraints")

class LLMResponse(BaseModel):
    content: str
    model: str
    tokens_used: int
    response_time: float
    cached: bool = False
    confidence: Optional[float] = None

class StreamingLLMResponse(BaseModel):
    type: str  # "token", "complete", "error"
    content: Optional[str] = None
    timestamp: str
    model: Optional[str] = None
    tokens_used: Optional[int] = None

# Router
router = APIRouter(prefix="/api/llm", tags=["LLM"])

# Health check endpoint
@router.get("/health")
async def llm_health_check():
    """Health check for LLM services"""
    global llm_service
    try:
        if llm_service is None:
            import os
            llm_service = LLMService(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            await llm_service.initialize()
        
        return {
            "status": "healthy",
            "service": "llm-service", 
            "models_available": {
                "openai": llm_service.openai_client is not None,
                "anthropic": llm_service.anthropic_client is not None
            },
            "vector_store": "available",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def get_llm_service() -> LLMService:
    """Dependency to get initialized LLM service"""
    global llm_service
    if llm_service is None:
        import os
        llm_service = LLMService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        await llm_service.initialize()
    return llm_service

@router.post("/draft/assistant", response_model=LLMResponse)
async def draft_assistant(
    request: DraftAssistantRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Real-time draft assistant that provides contextual advice
    """
    try:
        # Check usage limits
        usage_check = await subscription_service.check_usage_limits(current_user.id, 1)
        if not usage_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Usage limit exceeded",
                    "tier": usage_check["tier"],
                    "remaining": usage_check.get("remaining", 0)
                }
            )
        
        # Enhance context with existing ML data
        enhanced_context = await _enhance_draft_context(request.draft_context, db)
        
        # Generate response
        response_generator = llm.generate_response(
            query=request.query,
            context=enhanced_context,
            user_id=current_user.id,
            query_type="draft_assistant",
            complexity=request.complexity,
            stream=False
        )
        
        # Get first (and only) response for non-streaming
        response_data = None
        async for response in response_generator:
            response_data = response
            break
        
        if not response_data or "error" in response_data:
            raise HTTPException(
                status_code=500,
                detail=response_data.get("error", "Generation failed")
            )
        
        # Record usage
        await subscription_service.record_usage(
            current_user.id, 
            "draft_assistant", 
            response_data.get("tokens_used", 0)
        )
        
        return LLMResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Draft assistant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/draft/assistant/stream")
async def draft_assistant_stream(
    request: DraftAssistantRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Streaming draft assistant for real-time responses
    """
    
    async def generate_stream():
        try:
            # Check usage limits
            usage_check = await subscription_service.check_usage_limits(current_user.id, 1)
            if not usage_check["allowed"]:
                yield f"data: {{'error': 'Usage limit exceeded', 'code': 'RATE_LIMIT'}}\n\n"
                return
            
            # Enhance context
            enhanced_context = await _enhance_draft_context(request.draft_context, db)
            
            # Generate streaming response
            response_generator = llm.generate_response(
                query=request.query,
                context=enhanced_context,
                user_id=current_user.id,
                query_type="draft_assistant", 
                complexity=request.complexity,
                stream=True
            )
            
            tokens_used = 0
            async for response in response_generator:
                if response.get("type") == "complete":
                    tokens_used = response.get("tokens_used", 0)
                    # Record final usage
                    await subscription_service.record_usage(
                        current_user.id,
                        "draft_assistant",
                        tokens_used
                    )
                
                yield f"data: {response}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming draft assistant error: {e}")
            yield f"data: {{'error': '{str(e)}', 'code': 'STREAM_ERROR'}}\n\n"
    
    return EventSourceResponse(generate_stream())

@router.post("/analysis/injury", response_model=LLMResponse) 
async def injury_analysis(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Analyze injury reports and convert to fantasy-relevant insights
    """
    try:
        # Check usage limits
        usage_check = await subscription_service.check_usage_limits(current_user.id, 1)
        if not usage_check["allowed"]:
            raise HTTPException(status_code=429, detail="Usage limit exceeded")
        
        # Enhance context with player data
        enhanced_context = await _enhance_analysis_context(request, db)
        
        # Generate response
        response_generator = llm.generate_response(
            query=request.query,
            context=enhanced_context,
            user_id=current_user.id,
            query_type="analysis",
            complexity="high"
        )
        
        response_data = None
        async for response in response_generator:
            response_data = response
            break
        
        if not response_data or "error" in response_data:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Record usage
        await subscription_service.record_usage(
            current_user.id,
            "injury_analysis", 
            response_data.get("tokens_used", 0)
        )
        
        return LLMResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Injury analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trades/analyze", response_model=LLMResponse)
async def analyze_trade(
    request: TradeAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Multi-factor trade evaluation with natural language insights
    """
    try:
        # Check usage limits 
        usage_check = await subscription_service.check_usage_limits(current_user.id, 1)
        if not usage_check["allowed"]:
            raise HTTPException(status_code=429, detail="Usage limit exceeded")
        
        # Get player data for trade analysis
        all_player_ids = request.giving_players + request.receiving_players
        players_data = await _get_players_data(all_player_ids, db)
        
        # Build trade context
        trade_context = {
            "giving_players": [players_data.get(pid, {}) for pid in request.giving_players],
            "receiving_players": [players_data.get(pid, {}) for pid in request.receiving_players],
            "league_context": request.league_context,
            "team_needs": request.team_needs,
            "trade_type": "player_for_player"
        }
        
        # Generate analysis
        query = f"Analyze this fantasy football trade: giving {request.giving_players} for {request.receiving_players}"
        
        response_generator = llm.generate_response(
            query=query,
            context=trade_context,
            user_id=current_user.id,
            query_type="analysis",
            complexity="high"
        )
        
        response_data = None
        async for response in response_generator:
            response_data = response
            break
        
        if not response_data or "error" in response_data:
            raise HTTPException(status_code=500, detail="Trade analysis failed")
        
        # Record usage
        await subscription_service.record_usage(
            current_user.id,
            "trade_analysis",
            response_data.get("tokens_used", 0)
        )
        
        return LLMResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lineup/optimize", response_model=LLMResponse)
async def optimize_lineup(
    request: LineupOptimizationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Lineup optimization with explanations
    """
    try:
        # Check usage limits
        usage_check = await subscription_service.check_usage_limits(current_user.id, 1)
        if not usage_check["allowed"]:
            raise HTTPException(status_code=429, detail="Usage limit exceeded")
        
        # Get player predictions and data
        players_data = await _get_players_data(request.available_players, db)
        predictions = await _get_predictions_for_week(request.available_players, request.week, db)
        
        # Build lineup context
        lineup_context = {
            "available_players": players_data,
            "predictions": predictions,  
            "lineup_slots": request.lineup_slots,
            "week": request.week,
            "scoring_format": request.scoring_format,
            "constraints": request.constraints
        }
        
        query = f"Optimize my fantasy football lineup for week {request.week} with these players and explain your reasoning"
        
        response_generator = llm.generate_response(
            query=query,
            context=lineup_context,
            user_id=current_user.id,
            query_type="analysis",
            complexity="high"
        )
        
        response_data = None
        async for response in response_generator:
            response_data = response
            break
        
        if not response_data or "error" in response_data:
            raise HTTPException(status_code=500, detail="Lineup optimization failed")
        
        # Record usage
        await subscription_service.record_usage(
            current_user.id,
            "lineup_optimization",
            response_data.get("tokens_used", 0)
        )
        
        return LLMResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lineup optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subscription/usage")
async def get_usage_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's LLM usage statistics
    """
    try:
        tier = await subscription_service.get_user_tier(current_user.id)
        usage = await subscription_service.get_user_usage(current_user.id, "week")
        limits = await subscription_service.check_usage_limits(current_user.id, 0)
        
        return {
            "user_id": current_user.id,
            "tier": tier,
            "current_usage": usage,
            "limits": limits,
            "tier_features": subscription_service.get_tier_features(tier)
        }
        
    except Exception as e:
        logger.error(f"Usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time draft assistance
@router.websocket("/draft/live")
async def draft_live_websocket(
    websocket: WebSocket,
    user_id: str,  # Would normally extract from JWT token
    llm: LLMService = Depends(get_llm_service)
):
    """
    WebSocket endpoint for live draft assistance
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive draft context and query
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            draft_context = data.get("draft_context", {})
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Query is required"
                })
                continue
            
            # Check usage limits
            usage_check = await subscription_service.check_usage_limits(user_id, 1)
            if not usage_check["allowed"]:
                await websocket.send_json({
                    "type": "error", 
                    "message": "Usage limit exceeded"
                })
                continue
            
            # Stream response through websocket
            response_generator = llm.generate_response(
                query=query,
                context=draft_context,
                user_id=user_id,
                query_type="draft_assistant",
                complexity="medium",
                stream=True,
                websocket=websocket
            )
            
            async for response in response_generator:
                if response.get("type") == "complete":
                    # Record usage on completion
                    await subscription_service.record_usage(
                        user_id,
                        "draft_live",
                        response.get("tokens_used", 0)
                    )
                
                # Response is already sent via websocket in the streaming callback
                pass
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

# Helper functions
async def _enhance_draft_context(draft_context: Dict, db: Session) -> Dict:
    """Enhance draft context with ML predictions and tier data"""
    try:
        enhanced = draft_context.copy()
        
        # Get available players with tiers
        if "available_players" in draft_context:
            player_ids = draft_context["available_players"]
            
            # Get tier data
            tiers = db.query(DraftTier).filter(
                DraftTier.player_id.in_(player_ids)
            ).all()
            
            tier_map = {tier.player_id: {
                "tier": tier.tier,
                "tier_label": tier.tier_label,
                "probability": float(tier.probability) if tier.probability else None
            } for tier in tiers}
            
            enhanced["player_tiers"] = tier_map
        
        # Add current week predictions if available
        current_week = draft_context.get("current_week")
        if current_week and "available_players" in draft_context:
            predictions = db.query(Prediction).filter(
                Prediction.player_id.in_(draft_context["available_players"]),
                Prediction.week == current_week
            ).all()
            
            prediction_map = {pred.player_id: {
                "predicted_points": float(pred.predicted_points),
                "confidence": float(pred.prediction_std) if pred.prediction_std else None
            } for pred in predictions}
            
            enhanced["predictions"] = prediction_map
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Context enhancement error: {e}")
        return draft_context

async def _enhance_analysis_context(request: AnalysisRequest, db: Session) -> Dict:
    """Enhance analysis context with player data"""
    context = request.context.copy()
    
    if request.player_ids:
        players_data = await _get_players_data(request.player_ids, db)
        context["players"] = players_data
    
    context["analysis_type"] = request.analysis_type
    return context

async def _get_players_data(player_ids: List[str], db: Session) -> Dict[str, Dict]:
    """Get comprehensive player data"""
    players = db.query(Player).filter(Player.player_id.in_(player_ids)).all()
    
    return {
        player.player_id: {
            "name": player.full_name,
            "position": player.position,
            "team": player.team,
            "age": player.age,
            "years_exp": player.years_exp,
            "status": player.status
        } for player in players
    }

async def _get_predictions_for_week(player_ids: List[str], week: int, db: Session) -> Dict[str, Dict]:
    """Get predictions for specific week"""
    predictions = db.query(Prediction).filter(
        Prediction.player_id.in_(player_ids),
        Prediction.week == week
    ).all()
    
    return {
        pred.player_id: {
            "predicted_points": float(pred.predicted_points),
            "confidence_interval": pred.confidence_interval,
            "model_version": pred.model_version
        } for pred in predictions
    }