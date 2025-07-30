"""
WebSocket Manager for Real-time Updates
Supports live predictions, injury updates, and score notifications
"""
import json
import logging
import asyncio
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# WebSocket event types
class EventType:
    PREDICTION_UPDATE = "prediction_update"
    SCORE_UPDATE = "score_update"
    INJURY_UPDATE = "injury_update"
    NEWS_UPDATE = "news_update"
    TIER_CHANGE = "tier_change"
    MOMENTUM_ALERT = "momentum_alert"
    GAME_START = "game_start"
    GAME_END = "game_end"
    SYSTEM_ALERT = "system_alert"


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting
    Supports room-based subscriptions and user authentication
    """
    
    def __init__(self):
        # Active connections: {connection_id: {"websocket": ws, "user_id": str, "subscriptions": set}}
        self.active_connections: Dict[str, Dict] = {}
        
        # Room subscriptions: {room_name: set(connection_ids)}
        self.rooms: Dict[str, Set[str]] = {}
        
        # User to connections mapping: {user_id: set(connection_ids)}
        self.user_connections: Dict[str, Set[str]] = {}
        
        # Redis for pub/sub across multiple servers
        self.redis_client = None
        self.pubsub = None
        self.redis_task = None
    
    async def init_redis(self, redis_url: str):
        """Initialize Redis for distributed WebSocket support"""
        try:
            self.redis_client = await redis.from_url(redis_url)
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to broadcast channel
            await self.pubsub.subscribe("websocket_broadcast")
            
            # Start listening for Redis messages
            self.redis_task = asyncio.create_task(self._redis_listener())
            
            logger.info("WebSocket Redis pub/sub initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for WebSocket: {str(e)}")
    
    async def _redis_listener(self):
        """Listen for Redis pub/sub messages"""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await self._handle_redis_message(data)
                    except Exception as e:
                        logger.error(f"Error handling Redis message: {str(e)}")
        except Exception as e:
            logger.error(f"Redis listener error: {str(e)}")
    
    async def _handle_redis_message(self, data: dict):
        """Handle message from Redis pub/sub"""
        event_type = data.get("type")
        room = data.get("room")
        payload = data.get("payload")
        
        if room:
            await self._broadcast_to_room_local(room, {
                "type": event_type,
                "data": payload,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "subscriptions": set(),
            "auth_token": auth_token,
            "connected_at": datetime.utcnow()
        }
        
        # Map user to connection
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        # Send welcome message
        await self.send_personal_message(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        # Remove from user connections
        if connection["user_id"]:
            user_connections = self.user_connections.get(connection["user_id"], set())
            user_connections.discard(connection_id)
            if not user_connections:
                del self.user_connections[connection["user_id"]]
        
        # Remove from all rooms
        for room in connection["subscriptions"]:
            if room in self.rooms:
                self.rooms[room].discard(connection_id)
                if not self.rooms[room]:
                    del self.rooms[room]
        
        # Remove connection
        del self.active_connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def subscribe_to_room(self, connection_id: str, room: str):
        """Subscribe connection to a room"""
        if connection_id not in self.active_connections:
            return False
        
        # Add to room
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(connection_id)
        
        # Track subscription
        self.active_connections[connection_id]["subscriptions"].add(room)
        
        # Notify subscription
        await self.send_personal_message(connection_id, {
            "type": "subscribed",
            "room": room,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Connection {connection_id} subscribed to room {room}")
        return True
    
    async def unsubscribe_from_room(self, connection_id: str, room: str):
        """Unsubscribe connection from a room"""
        if connection_id not in self.active_connections:
            return False
        
        # Remove from room
        if room in self.rooms:
            self.rooms[room].discard(connection_id)
            if not self.rooms[room]:
                del self.rooms[room]
        
        # Remove subscription
        self.active_connections[connection_id]["subscriptions"].discard(room)
        
        # Notify unsubscription
        await self.send_personal_message(connection_id, {
            "type": "unsubscribed",
            "room": room,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    async def send_personal_message(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]["websocket"]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {str(e)}")
                await self.disconnect(connection_id)
    
    async def send_user_message(self, user_id: str, message: dict):
        """Send message to all connections of a user"""
        connection_ids = self.user_connections.get(user_id, set()).copy()
        
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
    
    async def broadcast_to_room(self, room: str, message: dict):
        """Broadcast message to all connections in a room"""
        # Broadcast locally
        await self._broadcast_to_room_local(room, message)
        
        # Broadcast to other servers via Redis
        if self.redis_client:
            try:
                await self.redis_client.publish("websocket_broadcast", json.dumps({
                    "type": message.get("type"),
                    "room": room,
                    "payload": message
                }))
            except Exception as e:
                logger.error(f"Redis broadcast error: {str(e)}")
    
    async def _broadcast_to_room_local(self, room: str, message: dict):
        """Broadcast message to local connections in a room"""
        connection_ids = self.rooms.get(room, set()).copy()
        
        # Send to all connections in room
        disconnected = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]["websocket"]
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {str(e)}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        connection_ids = list(self.active_connections.keys())
        
        disconnected = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]["websocket"]
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {str(e)}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)


# Global connection manager
manager = ConnectionManager()


# WebSocket endpoint handlers
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """Main WebSocket endpoint"""
    # Validate token and get user info
    user_id = None
    if token:
        # In production, validate JWT token here
        # user_id = validate_token(token)
        user_id = "demo_user"  # Placeholder
    
    # Accept connection
    connection_id = await manager.connect(websocket, user_id, token)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "subscribe":
                room = data.get("room")
                if room:
                    await manager.subscribe_to_room(connection_id, room)
            
            elif message_type == "unsubscribe":
                room = data.get("room")
                if room:
                    await manager.unsubscribe_from_room(connection_id, room)
            
            elif message_type == "ping":
                await manager.send_personal_message(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                # Echo unknown messages
                await manager.send_personal_message(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(connection_id)


# Helper functions for sending updates
async def send_prediction_update(player_id: str, prediction_data: dict):
    """Send prediction update to subscribers"""
    room = f"player:{player_id}"
    
    await manager.broadcast_to_room(room, {
        "type": EventType.PREDICTION_UPDATE,
        "player_id": player_id,
        "data": prediction_data,
        "timestamp": datetime.utcnow().isoformat()
    })


async def send_score_update(game_id: str, score_data: dict):
    """Send live score update"""
    room = f"game:{game_id}"
    
    await manager.broadcast_to_room(room, {
        "type": EventType.SCORE_UPDATE,
        "game_id": game_id,
        "data": score_data,
        "timestamp": datetime.utcnow().isoformat()
    })


async def send_injury_update(player_id: str, injury_data: dict):
    """Send injury update notification"""
    # Send to player subscribers
    await manager.broadcast_to_room(f"player:{player_id}", {
        "type": EventType.INJURY_UPDATE,
        "player_id": player_id,
        "data": injury_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Send to team subscribers
    team = injury_data.get("team")
    if team:
        await manager.broadcast_to_room(f"team:{team}", {
            "type": EventType.INJURY_UPDATE,
            "player_id": player_id,
            "data": injury_data,
            "timestamp": datetime.utcnow().isoformat()
        })


async def send_momentum_alert(player_id: str, momentum_data: dict):
    """Send momentum/breakout alerts"""
    alert_type = momentum_data.get("alert_type", "momentum_change")
    
    await manager.broadcast_to_room(f"player:{player_id}", {
        "type": EventType.MOMENTUM_ALERT,
        "player_id": player_id,
        "alert_type": alert_type,
        "data": momentum_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Also send to alerts room for users tracking all momentum changes
    await manager.broadcast_to_room("alerts:momentum", {
        "type": EventType.MOMENTUM_ALERT,
        "player_id": player_id,
        "alert_type": alert_type,
        "data": momentum_data,
        "timestamp": datetime.utcnow().isoformat()
    })


async def send_system_alert(message: str, severity: str = "info"):
    """Send system-wide alerts"""
    await manager.broadcast_to_all({
        "type": EventType.SYSTEM_ALERT,
        "severity": severity,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })


# WebSocket rooms structure:
# - player:{player_id} - Updates for specific player
# - team:{team} - Updates for all players on team
# - game:{game_id} - Live updates for specific game
# - user:{user_id} - Personal notifications
# - alerts:momentum - All momentum/breakout alerts
# - alerts:injuries - All injury updates
# - alerts:news - All news updates
# - system - System-wide announcements