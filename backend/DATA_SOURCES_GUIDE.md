# Fantasy Football AI - Available Data Sources Guide

## Overview
This document describes all data sources and APIs currently configured in this project, including what data can be accessed with existing API keys.

## 1. Primary Data Sources (Currently Active)

### Sleeper API (FREE - No API Key Required)
**Status**: ✅ **FULLY FUNCTIONAL**
- **Base URL**: `https://api.sleeper.app/v1`
- **Authentication**: None required (public API)
- **Rate Limit**: 1000 requests/minute

**Available Data**:
- **Players**: `/players/nfl` - Complete NFL player database
  - Player info: name, position, team, age, status
  - Fantasy positions and metadata
  - Updated regularly during season
  
- **Stats**: `/stats/nfl/{season_type}/{season}/{week}`
  - Historical stats from 2009-present
  - Real-time stats during games
  - Fantasy points (STD, PPR, Half-PPR)
  - Detailed stats: passing, rushing, receiving, defensive
  
- **Projections**: `/projections/nfl/{season_type}/{season}/{week}`
  - Weekly projections for all players
  - Updated before each week
  
- **League Data**: `/league/{league_id}`
  - User leagues and rosters
  - Matchups and scores
  - Draft results

### PostgreSQL Database (Supabase)
**Status**: ✅ **CONFIGURED & WORKING**
- **Connection**: Already configured in DATABASE_URL
- **Current Data**:
  - 11,386 NFL players
  - 30,406 player stats records
  - Seasons: 2019-2023 (partial)

### Redis Cache
**Status**: ✅ **CONFIGURED**
- **URL**: `redis://localhost:6379`
- **Purpose**: Caching API responses
- **TTL**: 24 hours for player data, 1 hour for stats

## 2. LLM APIs (Configured with Keys)

### OpenAI API
**Status**: ✅ **API KEY PRESENT**
- **Key**: Available in `.env.local`
- **Usage**: 
  - Player analysis and insights
  - Natural language queries
  - Fantasy advice generation
  - Chat functionality

### Anthropic API (Claude)
**Status**: ✅ **API KEY PRESENT**
- **Key**: Available in `.env.local`
- **Usage**:
  - Advanced analysis
  - Long-form content generation
  - Complex reasoning about matchups

## 3. Additional APIs (Keys Not Configured)

### ESPN Fantasy API
**Status**: ❌ **NO API KEY**
- **Placeholder**: `ESPN_API_KEY=` (empty)
- **Potential Data**:
  - ESPN fantasy leagues
  - Player news and updates
  - Expert rankings
  - Injury reports

### NFL Official API
**Status**: ❌ **NO API KEY**
- **Placeholder**: `NFL_API_KEY=` (empty)
- **Potential Data**:
  - Official NFL statistics
  - Game schedules
  - Team information
  - Play-by-play data

### Weather API
**Status**: ❌ **NO API KEY**
- **Placeholder**: `WEATHER_API_KEY=` (empty)
- **Potential Data**:
  - Game day weather conditions
  - Wind speed, precipitation
  - Temperature effects on scoring

## 4. Other Configured Services

### Stripe (Payment Processing)
**Status**: 🧪 **TEST MODE**
- **Keys**: Test keys configured
- **Usage**: Subscription management

### ChromaDB (Vector Store)
**Status**: ✅ **FUNCTIONAL**
- **Location**: Local storage
- **Usage**: 
  - Semantic search for players
  - Similar player recommendations
  - Historical performance matching

## 5. Data Collection Implementation

### Current Implementation
```python
# Sleeper API Client (backend/data/sleeper_client.py)
- Async/await for performance
- Redis caching
- Rate limiting (900 req/min)
- Automatic retries with backoff

# Available Methods:
- get_all_players()
- get_player(player_id)
- get_stats(season_type, season, week)
- get_week_stats(season_type, season, week)
- get_projections(season_type, season, week)
- get_trending_players(type, add_drop, hours)
```

### Data Pipeline Features
1. **Automatic Caching**: All API responses cached in Redis
2. **Rate Limiting**: Prevents API throttling
3. **Error Handling**: Exponential backoff on failures
4. **Type Safety**: Dataclass models for all entities

## 6. Data Availability Summary

### What You CAN Access Now:
✅ **Complete NFL player database** (11,000+ players)
✅ **Historical stats** (2009-present via Sleeper)
✅ **Weekly projections** (current season)
✅ **Fantasy scoring** (all formats)
✅ **Player trends** and hot pickups
✅ **LLM-powered analysis** (OpenAI + Anthropic)
✅ **Vector similarity search** (ChromaDB)

### What You CANNOT Access (Need API Keys):
❌ ESPN fantasy leagues and expert content
❌ Official NFL play-by-play data
❌ Weather conditions for games
❌ Premium stats providers (PFF, etc.)

## 7. Usage Examples

### Fetch Current Players
```python
from backend.data.sleeper_client import SleeperAPIClient

async with SleeperAPIClient() as client:
    players = await client.get_all_players()
    # Returns dict of 11,000+ NFL players
```

### Get Player Stats
```python
# Get Week 1 2023 stats
stats = await client.get_week_stats("regular", 2023, 1)
# Returns stats for all players who played
```

### LLM Analysis
```python
from backend.services.llm_service import LLMService

llm = LLMService()
analysis = await llm.analyze_player(
    player_id="4981",  # Josh Allen
    context="Week 10 matchup vs Jets"
)
```

## 8. Recommendations

1. **Immediate Use**: The Sleeper API provides comprehensive data for a fully functional fantasy football platform
2. **No Additional Keys Needed**: Current setup supports all core features
3. **Optional Enhancements**: 
   - Add weather API for game conditions
   - ESPN API for expert rankings
   - NFL API for deeper statistics

The project is fully capable of providing fantasy football predictions and analysis with the currently configured data sources.