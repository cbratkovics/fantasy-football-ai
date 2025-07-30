"""
WebSocket Routes for Real-time Updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import logging
import asyncio

from backend.core.websocket import (
    manager, 
    websocket_endpoint,
    send_prediction_update,
    send_injury_update,
    send_momentum_alert
)
from backend.core.auth import get_current_user_ws

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_main(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint for real-time updates
    
    Connection URL: ws://localhost:8000/ws?token=YOUR_TOKEN
    
    Message format:
    {
        "type": "subscribe|unsubscribe|ping",
        "room": "player:123|team:KC|game:456|alerts:momentum"
    }
    """
    await websocket_endpoint(websocket, token)


@router.websocket("/ws/predictions/{player_id}")
async def websocket_player_predictions(
    websocket: WebSocket,
    player_id: str,
    token: Optional[str] = Query(None)
):
    """
    Dedicated WebSocket for player prediction updates
    Auto-subscribes to player room
    """
    # Validate token
    user_id = None
    if token:
        # Validate token in production
        user_id = "demo_user"
    
    # Connect and auto-subscribe
    connection_id = await manager.connect(websocket, user_id, token)
    await manager.subscribe_to_room(connection_id, f"player:{player_id}")
    
    try:
        # Send initial prediction
        from backend.ml.ensemble_predictions import EnsemblePredictionEngine
        engine = EnsemblePredictionEngine()
        
        # Get current prediction
        prediction = engine.predict_player_week(
            player_id=player_id,
            season=2024,
            week=10,
            include_explanations=False
        )
        
        if "error" not in prediction:
            await manager.send_personal_message(connection_id, {
                "type": "initial_prediction",
                "data": prediction
            })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "refresh":
                # Refresh prediction
                prediction = engine.predict_player_week(
                    player_id=player_id,
                    season=data.get("season", 2024),
                    week=data.get("week", 10),
                    include_explanations=False
                )
                
                await manager.send_personal_message(connection_id, {
                    "type": "prediction_refresh",
                    "data": prediction
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(connection_id)


@router.websocket("/ws/live-scores")
async def websocket_live_scores(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Live score updates WebSocket
    Streams score updates for all active games
    """
    # Connect
    user_id = None
    if token:
        user_id = "demo_user"
    
    connection_id = await manager.connect(websocket, user_id, token)
    
    # Subscribe to all active games
    # In production, would get active games from database
    active_games = ["game:2024_10_KC_LV", "game:2024_10_BUF_MIA"]
    
    for game in active_games:
        await manager.subscribe_to_room(connection_id, game)
    
    try:
        # Simulate live score updates
        while True:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # In production, fetch real scores
            await manager.send_personal_message(connection_id, {
                "type": "heartbeat",
                "message": "Connection active"
            })
            
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(connection_id)


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    alert_types: str = Query("all"),  # comma-separated: momentum,injury,news
    token: Optional[str] = Query(None)
):
    """
    Alert subscription WebSocket
    Get real-time alerts for momentum changes, injuries, and news
    """
    # Parse alert types
    types = alert_types.split(",") if alert_types != "all" else ["momentum", "injury", "news"]
    
    # Connect
    user_id = None
    if token:
        user_id = "demo_user"
    
    connection_id = await manager.connect(websocket, user_id, token)
    
    # Subscribe to requested alert types
    for alert_type in types:
        await manager.subscribe_to_room(connection_id, f"alerts:{alert_type}")
    
    try:
        # Send welcome message
        await manager.send_personal_message(connection_id, {
            "type": "alerts_connected",
            "subscribed_types": types,
            "message": "You will receive real-time alerts"
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "update_subscriptions":
                # Update alert subscriptions
                new_types = data.get("alert_types", [])
                
                # Unsubscribe from old
                for alert_type in types:
                    if alert_type not in new_types:
                        await manager.unsubscribe_from_room(connection_id, f"alerts:{alert_type}")
                
                # Subscribe to new
                for alert_type in new_types:
                    if alert_type not in types:
                        await manager.subscribe_to_room(connection_id, f"alerts:{alert_type}")
                
                types = new_types
                
                await manager.send_personal_message(connection_id, {
                    "type": "subscriptions_updated",
                    "subscribed_types": types
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(connection_id)


# Demo endpoint to trigger updates
@router.post("/api/v1/demo/trigger-update")
async def trigger_demo_update(
    update_type: str,
    player_id: Optional[str] = None
):
    """
    Demo endpoint to trigger WebSocket updates
    For testing WebSocket functionality
    """
    if update_type == "prediction":
        if player_id:
            await send_prediction_update(player_id, {
                "predicted_points": 18.5,
                "confidence": 0.85,
                "trend": "up"
            })
            return {"status": "sent", "type": "prediction", "player_id": player_id}
    
    elif update_type == "injury":
        if player_id:
            await send_injury_update(player_id, {
                "status": "Questionable",
                "injury": "Hamstring",
                "impact": "May be limited",
                "return_date": "Week 11"
            })
            return {"status": "sent", "type": "injury", "player_id": player_id}
    
    elif update_type == "momentum":
        if player_id:
            await send_momentum_alert(player_id, {
                "alert_type": "breakout_candidate",
                "momentum_score": 0.25,
                "streak": 3,
                "recommendation": "Buy"
            })
            return {"status": "sent", "type": "momentum", "player_id": player_id}
    
    return {"error": "Invalid update type or missing player_id"}