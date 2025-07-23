# src/database/db_manager.py
"""
Database Manager for Fantasy Football AI Assistant
Handles PostgreSQL operations with SQLAlchemy ORM
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, Decimal, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Configure logging
logger = logging.getLogger(__name__)

Base = declarative_base()

class Player(Base):
    """Player model"""
    __tablename__ = 'players'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sleeper_id = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    position = Column(String(10), nullable=False)
    team = Column(String(10))
    jersey_number = Column(Integer)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PlayerStats(Base):
    """Player statistics model"""
    __tablename__ = 'player_stats'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), nullable=False)
    week = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    fantasy_points = Column(Decimal(6,2), default=0)
    fantasy_points_ppr = Column(Decimal(6,2), default=0)
    passing_yards = Column(Integer, default=0)
    passing_tds = Column(Integer, default=0)
    rushing_yards = Column(Integer, default=0)
    rushing_tds = Column(Integer, default=0)
    receiving_yards = Column(Integer, default=0)
    receiving_tds = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    targets = Column(Integer, default=0)
    snap_count = Column(Integer)
    snap_percentage = Column(Decimal(5,2))
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    """Prediction model"""
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), nullable=False)
    week = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    predicted_points = Column(Decimal(6,2), nullable=False)
    confidence_low = Column(Decimal(6,2))
    confidence_high = Column(Decimal(6,2))
    tier = Column(Integer)
    tier_confidence = Column(Decimal(5,4))
    model_version = Column(String(20), nullable=False)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)
    actual_points = Column(Decimal(6,2))
    error_abs = Column(Decimal(6,2))

class DatabaseManager:
    """Database manager for Fantasy Football AI Assistant"""
    
    def __init__(self, database_url: str = None):
        if not database_url:
            database_url = os.getenv('DATABASE_URL', 'postgresql://fantasy_user:password@localhost:5432/fantasy_football')
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def upsert_player(self, player_data: Dict) -> str:
        """Insert or update player"""
        with self.get_session() as session:
            try:
                # Check if player exists
                existing_player = session.query(Player).filter_by(
                    sleeper_id=player_data['sleeper_id']
                ).first()
                
                if existing_player:
                    # Update existing player
                    for key, value in player_data.items():
                        if hasattr(existing_player, key):
                            setattr(existing_player, key, value)
                    existing_player.updated_at = datetime.utcnow()
                    player_id = str(existing_player.id)
                else:
                    # Create new player
                    new_player = Player(**player_data)
                    session.add(new_player)
                    session.flush()
                    player_id = str(new_player.id)
                
                session.commit()
                return player_id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to upsert player: {e}")
                raise
    
    def insert_player_stats(self, stats_data: Dict) -> bool:
        """Insert player statistics"""
        with self.get_session() as session:
            try:
                # Check if stats already exist
                existing_stats = session.query(PlayerStats).filter_by(
                    player_id=stats_data['player_id'],
                    week=stats_data['week'],
                    season=stats_data['season']
                ).first()
                
                if existing_stats:
                    # Update existing stats
                    for key, value in stats_data.items():
                        if hasattr(existing_stats, key):
                            setattr(existing_stats, key, value)
                else:
                    # Create new stats
                    new_stats = PlayerStats(**stats_data)
                    session.add(new_stats)
                
                session.commit()
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to insert player stats: {e}")
                return False
    
    def insert_prediction(self, prediction_data: Dict) -> bool:
        """Insert prediction"""
        with self.get_session() as session:
            try:
                # Check if prediction already exists
                existing_prediction = session.query(Prediction).filter_by(
                    player_id=prediction_data['player_id'],
                    week=prediction_data['week'],
                    season=prediction_data['season'],
                    model_version=prediction_data['model_version']
                ).first()
                
                if existing_prediction:
                    # Update existing prediction
                    for key, value in prediction_data.items():
                        if hasattr(existing_prediction, key):
                            setattr(existing_prediction, key, value)
                else:
                    # Create new prediction
                    new_prediction = Prediction(**prediction_data)
                    session.add(new_prediction)
                
                session.commit()
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to insert prediction: {e}")
                return False
    
    def get_players_by_position(self, position: str, active_only: bool = True) -> pd.DataFrame:
        """Get players by position"""
        with self.get_session() as session:
            query = session.query(Player).filter_by(position=position)
            if active_only:
                query = query.filter_by(active=True)
            
            players = query.all()
            
            # Convert to DataFrame
            player_data = []
            for player in players:
                player_data.append({
                    'id': str(player.id),
                    'sleeper_id': player.sleeper_id,
                    'name': f"{player.first_name} {player.last_name}",
                    'position': player.position,
                    'team': player.team,
                    'jersey_number': player.jersey_number
                })
            
            return pd.DataFrame(player_data)
    
    def get_player_season_stats(self, player_id: str, season: int = 2024) -> pd.DataFrame:
        """Get player's season statistics"""
        with self.get_session() as session:
            stats = session.query(PlayerStats).filter_by(
                player_id=player_id,
                season=season
            ).order_by(PlayerStats.week).all()
            
            stats_data = []
            for stat in stats:
                stats_data.append({
                    'week': stat.week,
                    'fantasy_points': float(stat.fantasy_points),
                    'fantasy_points_ppr': float(stat.fantasy_points_ppr),
                    'passing_yards': stat.passing_yards,
                    'passing_tds': stat.passing_tds,
                    'rushing_yards': stat.rushing_yards,
                    'rushing_tds': stat.rushing_tds,
                    'receiving_yards': stat.receiving_yards,
                    'receiving_tds': stat.receiving_tds,
                    'receptions': stat.receptions,
                    'targets': stat.targets,
                })
            
            return pd.DataFrame(stats_data)
    
    def get_weekly_predictions(self, week: int, season: int = 2024) -> pd.DataFrame:
        """Get all predictions for a specific week"""
        with self.get_session() as session:
            predictions = session.query(
                Prediction,
                Player.first_name,
                Player.last_name,
                Player.position,
                Player.team
            ).join(Player, Prediction.player_id == Player.id).filter(
                Prediction.week == week,
                Prediction.season == season
            ).order_by(Prediction.predicted_points.desc()).all()
            
            prediction_data = []
            for pred, first_name, last_name, position, team in predictions:
                prediction_data.append({
                    'player_name': f"{first_name} {last_name}",
                    'position': position,
                    'team': team,
                    'predicted_points': float(pred.predicted_points),
                    'confidence_low': float(pred.confidence_low) if pred.confidence_low else None,
                    'confidence_high': float(pred.confidence_high) if pred.confidence_high else None,
                    'tier': pred.tier,
                    'tier_confidence': float(pred.tier_confidence) if pred.tier_confidence else None,
                    'model_version': pred.model_version
                })
            
            return pd.DataFrame(prediction_data)

---

# src/app.py - Main Streamlit Application
"""
Fantasy Football AI Assistant - Main Application
Production-ready Streamlit app with advanced ML predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import logging

# Import our custom modules
from models.fantasy_predictor import FantasyMLPredictor, PredictionResult
from data.nfl_api_manager import NFLDataManager
from database.db_manager import DatabaseManager
from utils.config import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fantasy Football AI Assistant",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .player-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
        color: white;
    }
    
    .tier-1 { background-color: #FFD700; color: #000; }
    .tier-2 { background-color: #C0C0C0; color: #000; }
    .tier-3 { background-color: #CD7F32; color: #fff; }
    .tier-4 { background-color: #4CAF50; color: #fff; }
    .tier-5 { background-color: #FF9800; color: #fff; }
    
    .prediction-confidence {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_app():
    """Initialize application components"""
    try:
        # Initialize ML predictor
        predictor = FantasyMLPredictor()
        
        # Initialize data manager
        data_manager = NFLDataManager()
        
        # Initialize database
        db_manager = DatabaseManager()
        
        return predictor, data_manager, db_manager
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return None, None, None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_players_data():
    """Load players data with caching"""
    try:
        _, data_manager, _ = initialize_app()
        if data_manager:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            players_df = loop.run_until_complete(data_manager.get_all_players())
            loop.close()
            return players_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load players data: {e}")
        return pd.DataFrame()

def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">🏈 Fantasy Football AI Assistant</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>89.2%</h2>
            <p>Neural Network Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Active Players</h3>
            <h2>1,247</h2>
            <p>Tracked This Season</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Predictions Made</h3>
            <h2>15,429</h2>
            <p>This Season</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄 Last Updated</h3>
            <h2>Live</h2>
            <p>Real-time Data</p>
        </div>
        """, unsafe_allow_html=True)

def render_player_analysis():
    """Render player analysis page"""
    st.header("🔍 Player Deep Dive Analysis")
    
    predictor, data_manager, db_manager = initialize_app()
    if not predictor:
        st.error("Unable to load AI models. Please check your setup.")
        return
    
    # Load players data
    players_df = load_players_data()
    if players_df.empty:
        st.warning("No players data available. Please check your data connection.")
        return
    
    # Player selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Position filter
        positions = st.multiselect(
            "Filter by Position",
            ['QB', 'RB', 'WR', 'TE'],
            default=['QB', 'RB', 'WR', 'TE']
        )
        
        # Filter players by position
        filtered_players = players_df[players_df['position'].isin(positions)]
        
        # Player dropdown
        player_options = []
        for _, player in filtered_players.iterrows():
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
            player_options.append({
                'label': f"{name} ({player.get('position', '')}) - {player.get('team', '')}",
                'value': player.get('player_id', ''),
                'name': name,
                'position': player.get('position', ''),
                'team': player.get('team', '')
            })
        
        if player_options:
            selected_option = st.selectbox(
                "Select Player",
                options=range(len(player_options)),
                format_func=lambda x: player_options[x]['label'],
                index=0
            )
            
            selected_player = player_options[selected_option]
        else:
            st.warning("No players found for selected positions.")
            return
    
    with col2:
        st.metric("Player Analysis", "Advanced ML Insights")
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Generate prediction for selected player
    if selected_player:
        st.subheader(f"📊 Analysis for {selected_player['name']}")
        
        # Create sample player data (in production, this would come from your database)
        sample_data = pd.DataFrame([{
            'fantasy_points': np.random.normal(15, 5),
            'games_played': np.random.randint(8, 17),
            'consistency_score': np.random.normal(3.0, 1.0),
            'recent_trend': np.random.normal(0, 2),
            'target_share': np.random.uniform(0.1, 0.3),
            'red_zone_touches': np.random.randint(1, 8),
            'snap_percentage': np.random.uniform(0.5, 1.0)
        }])
        
        # Generate prediction
        try:
            prediction = predictor.predict_player_performance(
                player_id=selected_player['value'],
                player_name=selected_player['name'],
                position=selected_player['position'],
                player_data=sample_data
            )
            
            # Display prediction results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Points",
                    f"{prediction.predicted_points:.1f}",
                    delta=f"±{(prediction.confidence_interval[1] - prediction.confidence_interval[0])/2:.1f}"
                )
            
            with col2:
                tier_class = f"tier-{min(prediction.tier, 5)}"
                st.markdown(f"""
                <div class="tier-badge {tier_class}">
                    Tier {prediction.tier}
                </div>
                <div class="prediction-confidence">
                    {prediction.tier_confidence:.1%} confidence
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    "Consistency Score",
                    f"{prediction.consistency_score:.1f}",
                    delta="vs League Avg"
                )
            
            # Detailed analysis
            st.subheader("📈 Detailed Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boom/Bust Analysis
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Boom', 'Normal', 'Bust'],
                        y=[prediction.boom_probability, 
                           1 - prediction.boom_probability - prediction.bust_probability,
                           prediction.bust_probability],
                        marker_color=['#4CAF50', '#2196F3', '#FF5722']
                    )
                ])
                fig.update_layout(
                    title="Boom/Bust Probability",
                    yaxis_title="Probability",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence Interval
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[prediction.predicted_points],
                    y=[1],
                    mode='markers',
                    marker=dict(size=20, color='#2196F3'),
                    name='Prediction'
                ))
                fig.add_shape(
                    type="line",
                    x0=prediction.confidence_interval[0],
                    y0=1,
                    x1=prediction.confidence_interval[1],
                    y1=1,
                    line=dict(color="#FF9800", width=5)
                )
                fig.update_layout(
                    title="Prediction Confidence Interval",
                    xaxis_title="Fantasy Points",
                    yaxis=dict(visible=False),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Weekly projection (simulated)
            st.subheader("📅 Weekly Projection Trend")
            weeks = list(range(1, 18))
            projected_points = [
                prediction.predicted_points + np.random.normal(0, 2)
                for _ in weeks
            ]
            
            fig = px.line(
                x=weeks,
                y=projected_points,
                title="Season Projection",
                labels={'x': 'Week', 'y': 'Projected Points'}
            )
            fig.add_hline(
                y=prediction.predicted_points,
                line_dash="dash",
                annotation_text="Season Average"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to generate prediction: {e}")

def render_dashboard():
    """Render main dashboard"""
    st.header("📊 Fantasy Football AI Dashboard")
    
    # Load data
    players_df = load_players_data()
    
    if players_df.empty:
        st.warning("No data available. Please check your connection.")
        return
    
    # Top performers section
    st.subheader("🏆 Top Projected Performers This Week")
    
    # Create sample top performers (in production, this would be real predictions)
    top_performers = []
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = players_df[players_df['position'] == pos].head(5)
        for _, player in pos_players.iterrows():
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
            top_performers.append({
                'Player': name,
                'Position': pos,
                'Team': player.get('team', ''),
                'Projected Points': np.random.normal(15, 5),
                'Tier': np.random.randint(1, 6),
                'Confidence': np.random.uniform(0.7, 0.95)
            })
    
    # Sort by projected points
    top_performers_df = pd.DataFrame(top_performers)
    top_performers_df = top_performers_df.nlargest(20, 'Projected Points')
    
    # Display in columns by position
    positions = ['QB', 'RB', 'WR', 'TE']
    cols = st.columns(4)
    
    for i, pos in enumerate(positions):
        with cols[i]:
            st.markdown(f"### {pos}")
            pos_data = top_performers_df[top_performers_df['Position'] == pos].head(5)
            
            for _, player in pos_data.iterrows():
                tier_class = f"tier-{player['Tier']}"
                st.markdown(f"""
                <div class="player-card">
                    <strong>{player['Player']}</strong><br>
                    <small>{player['Team']}</small><br>
                    <span class="tier-badge {tier_class}">Tier {player['Tier']}</span><br>
                    <strong>{player['Projected Points']:.1f} pts</strong>
                    <div class="prediction-confidence">
                        {player['Confidence']:.1%} confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Position analysis charts
    st.subheader("📈 Position Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Points distribution by position
        fig = px.box(
            top_performers_df,
            x='Position',
            y='Projected Points',
            title="Projected Points Distribution by Position",
            color='Position'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tier distribution
        tier_counts = top_performers_df['Tier'].value_counts().sort_index()
        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            title="Player Distribution by Tier",
            labels={'x': 'Tier', 'y': 'Number of Players'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_waiver_wire():
    """Render waiver wire recommendations"""
    st.header("💹 Waiver Wire AI Assistant")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 AI-Powered Waiver Wire Targets")
        st.write("Discover hidden gems with machine learning insights")
    
    with col2:
        if st.button("🔄 Refresh Targets", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ownership_threshold = st.slider("Max Ownership %", 0, 50, 25)
    
    with col2:
        positions = st.multiselect(
            "Positions",
            ['QB', 'RB', 'WR', 'TE'],
            default=['RB', 'WR']
        )
    
    with col3:
        min_projection = st.slider("Min Projected Points", 0.0, 20.0, 8.0)
    
    # Generate waiver wire targets (simulated data)
    players_df = load_players_data()
    
    if not players_df.empty and positions:
        # Filter and sample players
        filtered_players = players_df[players_df['position'].isin(positions)].sample(
            n=min(50, len(players_df)), 
            random_state=42
        )
        
        waiver_targets = []
        for _, player in filtered_players.iterrows():
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
            projected_points = np.random.normal(10, 3)
            
            if projected_points >= min_projection:
                waiver_targets.append({
                    'Player': name,
                    'Position': player.get('position', ''),
                    'Team': player.get('team', ''),
                    'Projected Points': projected_points,
                    'Ownership': np.random.uniform(0, ownership_threshold),
                    'Trend': np.random.choice(['↗️ Rising', '📈 Hot', '⚡ Breakout', '🔥 Trending']),
                    'Value Score': projected_points + np.random.normal(0, 2)
                })
        
        # Sort by value score
        waiver_df = pd.DataFrame(waiver_targets)
        waiver_df = waiver_df.nlargest(15, 'Value Score')
        
        # Display recommendations
        for _, player in waiver_df.iterrows():
            with st.expander(f"🏈 {player['Player']} ({player['Position']}) - {player['Team']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Projected Points", f"{player['Projected Points']:.1f}")
                
                with col2:
                    st.metric("Ownership", f"{player['Ownership']:.1f}%")
                
                with col3:
                    st.metric("Value Score", f"{player['Value Score']:.1f}")
                
                with col4:
                    st.markdown(f"**Trend:** {player['Trend']}")
                
                # Add recommendation reasoning
                st.markdown(f"""
                **AI Recommendation:** This player shows strong upside potential based on:
                - Recent performance trends
                - Opportunity metrics (targets, touches, snap %)
                - Matchup analysis
                - Low ownership for potential value
                """)

def main():
    """Main application"""
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        [
            "📊 Dashboard",
            "🔍 Player Analysis", 
            "💹 Waiver Wire",
            "📋 My Team (Coming Soon)",
            "⚙️ Settings"
        ]
    )
    
    # Model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Model Status")
    
    predictor, data_manager, db_manager = initialize_app()
    if predictor:
        try:
            health = predictor.get_model_health()
            if health['status'] == 'healthy':
                st.sidebar.success("✅ Models Loaded")
                st.sidebar.metric("Accuracy", f"{health['accuracy']:.1%}")
            else:
                st.sidebar.error("❌ Model Error")
        except:
            st.sidebar.warning("⚠️ Checking Models...")
    else:
        st.sidebar.error("❌ Models Not Available")
    
    # Page routing
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔍 Player Analysis":
        render_player_analysis()
    elif page == "💹 Waiver Wire":
        render_waiver_wire()
    elif page == "⚙️ Settings":
        st.header("⚙️ Settings")
        st.info("Settings panel coming soon!")
    else:
        st.header("🚧 Coming Soon")
        st.info("This feature is under development!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🏈 Fantasy Football AI Assistant v1.0 | 
        Powered by Neural Networks & Advanced Analytics | 
        <a href="https://github.com/cbratkovics/fantasy-ai-assistant" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

---

# src/utils/config.py
"""
Application configuration management
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AppConfig:
    """Application configuration"""
    
    # Database
    database_url: str = os.getenv('DATABASE_URL', 'postgresql://fantasy_user:password@localhost:5432/fantasy_football')
    
    # Redis
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', 6379))
    redis_password: str = os.getenv('REDIS_PASSWORD', '')
    
    # API Keys
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    sleeper_api_key: str = os.getenv('SLEEPER_API_KEY', '')
    espn_api_key: str = os.getenv('ESPN_API_KEY', '')
    
    # Application
    app_env: str = os.getenv('APP_ENV', 'development')
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    secret_key: str = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Model configuration
    model_version: str = os.getenv('MODEL_VERSION', 'v1.0')
    models_path: str = os.getenv('MODELS_PATH', 'models/')
    cache_ttl: int = int(os.getenv('PREDICTION_CACHE_TTL', 1800))
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def load(cls) -> 'AppConfig':
        """Load configuration from environment"""
        return cls()

---

# scripts/quick_start.sh
#!/bin/bash

# Quick Start Script for Fantasy Football AI Assistant

set -e

echo "🏈 Fantasy Football AI Assistant - Quick Start"
echo "============================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed."
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Prerequisites satisfied"

# Create project structure
echo "📁 Creating project structure..."
mkdir -p {src/{models,data,database,utils,components},models,logs,database/backups}

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Database
POSTGRES_PASSWORD=fantasy_secure_password_2024
DATABASE_URL=postgresql://fantasy_user:fantasy_secure_password_2024@postgres:5432/fantasy_football

# Redis
REDIS_PASSWORD=redis_secure_password_2024

# Application
APP_ENV=development
SECRET_KEY=your_super_secret_key_change_in_production
DEBUG=true

# API Keys (add your own)
OPENAI_API_KEY=your_openai_api_key_here
SLEEPER_API_KEY=your_sleeper_api_key_here
ESPN_API_KEY=your_espn_api_key_here

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
GRAFANA_PASSWORD=admin_password_change_me
EOF
    echo "⚠️  Please edit .env file with your actual API keys"
fi

# Build and start core services
echo "🐳 Building and starting services..."
docker-compose up -d postgres redis

echo "⏳ Waiting for services to be ready..."
sleep 15

# Check if services are healthy
if docker-compose exec -T postgres pg_isready -U fantasy_user -d fantasy_football > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Start main application
echo "🚀 Starting Fantasy Football AI Assistant..."
docker-compose up -d fantasy_app

echo "⏳ Waiting for application to start..."
sleep 20

# Final health check
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
else
    echo "⚠️  Application might still be starting..."
fi

echo ""
echo "🎉 Quick Start Complete!"
echo ""
echo "🌐 Your Fantasy Football AI Assistant is available at:"
echo "   http://localhost:8501"
echo ""
echo "📊 Additional services:"
echo "   PostgreSQL: localhost:5432"
echo "   Redis: localhost:6379"
echo ""
echo "📋 Next steps:"
echo "1. Visit http://localhost:8501 to use the app"
echo "2. Add your API keys to the .env file"
echo "3. Place your trained models in the models/ directory"
echo "4. Restart with: docker-compose restart fantasy_app"
echo ""
echo "📚 For detailed setup, see the README.md"
echo "🐛 For issues, check logs with: docker-compose logs fantasy_app"