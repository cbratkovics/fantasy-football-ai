"""
Vector Store Service for Fantasy Football AI
Handles embeddings and semantic search for enhanced LLM context
"""

import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

from backend.core.cache import get_redis_client
from backend.models.database import get_db, Player, PlayerStats, Prediction

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Vector database service for semantic search and context retrieval
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.chroma_client = None
        self.collections = {}
        self.sentence_model = None
        self.openai_embeddings = None
        self.redis_client = None
        
        # Collection names
        self.PLAYER_COLLECTION = "fantasy_players"
        self.NEWS_COLLECTION = "player_news"
        self.DRAFT_SCENARIOS = "draft_scenarios"
        self.TRADE_HISTORY = "trade_history"
        
    async def initialize(self):
        """Initialize vector store components"""
        try:
            # Initialize ChromaDB with new client format
            self.chroma_client = chromadb.PersistentClient(
                path="./vector_db"
            )
            
            # Initialize embedding models
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if self.openai_api_key:
                self.openai_embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key
                )
            
            # Initialize Redis
            self.redis_client = await get_redis_client()
            
            # Create or get collections
            await self._initialize_collections()
            
            logger.info("Vector store service initialized")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise
    
    async def _initialize_collections(self):
        """Create or retrieve ChromaDB collections"""
        try:
            # Player profiles collection
            self.collections[self.PLAYER_COLLECTION] = self.chroma_client.get_or_create_collection(
                name=self.PLAYER_COLLECTION,
                metadata={"description": "Player profiles and statistics"}
            )
            
            # News collection
            self.collections[self.NEWS_COLLECTION] = self.chroma_client.get_or_create_collection(
                name=self.NEWS_COLLECTION,
                metadata={"description": "Player news and injury reports"}
            )
            
            # Draft scenarios collection
            self.collections[self.DRAFT_SCENARIOS] = self.chroma_client.get_or_create_collection(
                name=self.DRAFT_SCENARIOS,
                metadata={"description": "Common draft scenarios and advice"}
            )
            
            # Trade history collection
            self.collections[self.TRADE_HISTORY] = self.chroma_client.get_or_create_collection(
                name=self.TRADE_HISTORY,
                metadata={"description": "Historical trade analysis"}
            )
            
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise
    
    def _generate_embedding(self, text: str, model: str = "sentence_transformer") -> List[float]:
        """Generate embedding for text"""
        try:
            if model == "sentence_transformer":
                return self.sentence_model.encode(text).tolist()
            elif model == "openai" and self.openai_embeddings:
                return self.openai_embeddings.embed_query(text)
            else:
                return self.sentence_model.encode(text).tolist()
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def index_player_profiles(self, force_reindex: bool = False):
        """Index all player profiles for semantic search"""
        try:
            # Check if already indexed recently
            cache_key = "vector_store:players_indexed"
            if not force_reindex and self.redis_client:
                last_indexed = await self.redis_client.get(cache_key)
                if last_indexed:
                    logger.info("Player profiles already indexed recently")
                    return
            
            with next(get_db()) as db:
                players = db.query(Player).all()
                
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                for player in players:
                    # Create searchable text
                    player_text = f"{player.full_name} {player.position} {player.team}"
                    if player.age:
                        player_text += f" age {player.age}"
                    if player.years_exp:
                        player_text += f" {player.years_exp} years experience"
                    
                    # Generate embedding
                    embedding = self._generate_embedding(player_text)
                    if not embedding:
                        continue
                    
                    documents.append(player_text)
                    embeddings.append(embedding)
                    metadatas.append({
                        "player_id": player.player_id,
                        "name": player.full_name,
                        "position": player.position,
                        "team": player.team,
                        "age": player.age or 0,
                        "years_exp": player.years_exp or 0,
                        "indexed_at": datetime.now().isoformat()
                    })
                    ids.append(f"player_{player.player_id}")
                
                # Add to collection
                if documents:
                    self.collections[self.PLAYER_COLLECTION].add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                # Cache indexing timestamp
                if self.redis_client:
                    await self.redis_client.setex(cache_key, 3600, datetime.now().isoformat())
                
                logger.info(f"Indexed {len(documents)} player profiles")
                
        except Exception as e:
            logger.error(f"Player indexing failed: {e}")
    
    async def index_draft_scenarios(self):
        """Index common draft scenarios and advice"""
        try:
            # Predefined draft scenarios with advice
            scenarios = [
                {
                    "scenario": "RB dead zone rounds 3-5",
                    "advice": "Avoid RBs in rounds 3-5 as value drops significantly. Focus on WRs and target RBs in rounds 6+ or handcuffs.",
                    "positions": ["RB"],
                    "rounds": [3, 4, 5],
                    "strategy": "positional_value"
                },
                {
                    "scenario": "Late round QB strategy",
                    "advice": "Wait until rounds 10+ for QB. Target streaming options and focus on volume-based QBs in good offenses.",
                    "positions": ["QB"],
                    "rounds": [10, 11, 12, 13, 14, 15],
                    "strategy": "positional_scarcity"
                },
                {
                    "scenario": "TE premium after top tier",
                    "advice": "After Kelce/Andrews tier, TE becomes dart throws. Consider waiting for volume-based TEs or rookie breakouts.",
                    "positions": ["TE"],
                    "rounds": [4, 5, 6, 7, 8],
                    "strategy": "tier_based"
                },
                {
                    "scenario": "Zero RB strategy",
                    "advice": "Load up on WRs early, find RB value in middle rounds. Target handcuffs and snap count climbers.",
                    "positions": ["RB", "WR"],
                    "rounds": [1, 2, 3],
                    "strategy": "contrarian"
                },
                {
                    "scenario": "Stacking QB with pass catchers",
                    "advice": "In superflex/2QB, stack QB with WR or TE from same team to maximize ceiling games.",
                    "positions": ["QB", "WR", "TE"],
                    "rounds": [1, 2, 3, 4, 5, 6],
                    "strategy": "correlation"
                }
            ]
            
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, scenario in enumerate(scenarios):
                # Create searchable text
                scenario_text = f"{scenario['scenario']} {scenario['advice']} {' '.join(scenario['positions'])}"
                
                embedding = self._generate_embedding(scenario_text)
                if not embedding:
                    continue
                
                documents.append(scenario_text)
                embeddings.append(embedding)
                metadatas.append({
                    "scenario_id": f"draft_scenario_{i}",
                    "scenario": scenario["scenario"],
                    "advice": scenario["advice"],
                    "positions": scenario["positions"],
                    "rounds": scenario["rounds"],
                    "strategy": scenario["strategy"],
                    "indexed_at": datetime.now().isoformat()
                })
                ids.append(f"scenario_{i}")
            
            if documents:
                self.collections[self.DRAFT_SCENARIOS].add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Indexed {len(documents)} draft scenarios")
            
        except Exception as e:
            logger.error(f"Draft scenario indexing failed: {e}")
    
    async def search_similar_players(
        self, 
        query: str, 
        n_results: int = 5,
        position_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar players based on query"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Build where clause for filtering
            where_clause = {}
            if position_filter:
                where_clause["position"] = position_filter
            
            # Search collection
            results = self.collections[self.PLAYER_COLLECTION].query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            similar_players = []
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas'][0]):
                    similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    similar_players.append({
                        "player_id": metadata["player_id"],
                        "name": metadata["name"],
                        "position": metadata["position"],
                        "team": metadata["team"],
                        "similarity_score": similarity_score,
                        "matched_text": results['documents'][0][i] if results['documents'] else ""
                    })
            
            return similar_players
            
        except Exception as e:
            logger.error(f"Player search failed: {e}")
            return []
    
    async def search_draft_scenarios(
        self, 
        query: str, 
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for relevant draft scenarios"""
        try:
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            results = self.collections[self.DRAFT_SCENARIOS].query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            scenarios = []
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas'][0]):
                    similarity_score = 1 - results['distances'][0][i]
                    
                    scenarios.append({
                        "scenario": metadata["scenario"],
                        "advice": metadata["advice"],
                        "positions": metadata["positions"],
                        "strategy": metadata["strategy"],
                        "similarity_score": similarity_score
                    })
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Draft scenario search failed: {e}")
            return []
    
    async def get_enhanced_context(
        self, 
        query: str, 
        context_type: str = "general"
    ) -> Dict[str, Any]:
        """Get enhanced context using vector search"""
        try:
            enhanced_context = {
                "query": query,
                "context_type": context_type,
                "retrieved_at": datetime.now().isoformat()
            }
            
            if context_type == "draft":
                # Search for relevant draft scenarios
                scenarios = await self.search_draft_scenarios(query, n_results=3)
                enhanced_context["draft_scenarios"] = scenarios
                
                # Search for similar players if names mentioned
                players = await self.search_similar_players(query, n_results=5)
                enhanced_context["similar_players"] = players
                
            elif context_type == "player_analysis":
                # Focus on player similarities
                players = await self.search_similar_players(query, n_results=8)
                enhanced_context["similar_players"] = players
                
            elif context_type == "trade":
                # Get similar players for trade context
                players = await self.search_similar_players(query, n_results=10)
                enhanced_context["similar_players"] = players
            
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Context enhancement failed: {e}")
            return {"error": str(e)}
    
    async def add_news_item(
        self, 
        player_id: str, 
        headline: str, 
        content: str, 
        source: str,
        impact_score: float = 0.5
    ):
        """Add news item to vector store"""
        try:
            news_text = f"{headline} {content}"
            embedding = self._generate_embedding(news_text)
            
            if embedding:
                news_id = f"news_{player_id}_{datetime.now().timestamp()}"
                
                self.collections[self.NEWS_COLLECTION].add(
                    documents=[news_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "player_id": player_id,
                        "headline": headline,
                        "source": source,
                        "impact_score": impact_score,
                        "created_at": datetime.now().isoformat()
                    }],
                    ids=[news_id]
                )
                
                logger.info(f"Added news item for player {player_id}")
                
        except Exception as e:
            logger.error(f"News indexing failed: {e}")
    
    async def search_player_news(
        self, 
        player_id: str, 
        days_back: int = 7,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for recent news about a player"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            results = self.collections[self.NEWS_COLLECTION].query(
                query_texts=[f"player {player_id}"],
                n_results=n_results,
                where={"player_id": player_id}
            )
            
            news_items = []
            if results and results['metadatas']:
                for metadata in results['metadatas'][0]:
                    created_at = datetime.fromisoformat(metadata["created_at"])
                    if created_at >= cutoff_date:
                        news_items.append({
                            "headline": metadata["headline"],
                            "source": metadata["source"],
                            "impact_score": metadata["impact_score"],
                            "created_at": metadata["created_at"]
                        })
            
            return news_items
            
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about vector store collections"""
        try:
            stats = {}
            
            for name, collection in self.collections.items():
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "last_updated": datetime.now().isoformat()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}