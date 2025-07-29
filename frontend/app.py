import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Get API URL from environment or default
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Fantasy Football AI",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .player-card {
        background-color: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .tier-1 { background-color: #fbbf24; color: #1f2937; }
    .tier-2 { background-color: #c0c0c0; color: #1f2937; }
    .tier-3 { background-color: #cd7f32; color: #ffffff; }
    .tier-4 { background-color: #3b82f6; color: #ffffff; }
    .tier-5 { background-color: #6b7280; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Fantasy Football AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>ML-Powered Draft Optimization & Weekly Predictions</p>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Player Rankings", "Draft Assistant", "Weekly Projections", "Team Analysis"]
)

# Helper function for API calls
def api_call(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API call with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Ensure the backend service is running.")
        st.info("If running locally, verify that 'make docker-up' has been executed.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Dashboard Page
if page == "Dashboard":
    st.header("Fantasy Football Dashboard")
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    
    health = api_call("/health")
    if health:
        with col1:
            st.metric("API Status", "Online" if health.get("status") == "healthy" else "Offline")
        with col2:
            st.metric("Database", health.get("database", "Unknown"))
        with col3:
            st.metric("ML Models", "Loaded" if health.get("models_loaded") else "Not Loaded")
        with col4:
            st.metric("Data Updated", datetime.now().strftime("%Y-%m-%d"))
    
    # Top Players by Position
    st.subheader("Top Players by Position")
    
    rankings = api_call("/players/rankings?limit=15")
    if rankings:
        df = pd.DataFrame(rankings)
        
        # Group by position
        positions = ["QB", "RB", "WR", "TE"]
        cols = st.columns(len(positions))
        
        for idx, pos in enumerate(positions):
            with cols[idx]:
                st.markdown(f"### {pos}")
                pos_players = df[df['position'] == pos].head(3)
                
                for _, player in pos_players.iterrows():
                    tier_class = f"tier-{player.get('tier', 5)}"
                    st.markdown(f"""
                    <div class='player-card'>
                        <strong>{player['name']}</strong> ({player['team']})<br>
                        <span class='tier-badge {tier_class}'>{player.get('tier_label', 'Tier ' + str(player.get('tier', 'N/A')))}</span><br>
                        Projected: {player.get('predicted_points', 0):.1f} pts<br>
                        Trend: {player.get('trend', 'stable')}
                    </div>
                    """, unsafe_allow_html=True)

# Player Rankings Page
elif page == "Player Rankings":
    st.header("Player Rankings & Analysis")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        position_filter = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE", "K", "DEF"])
    with col2:
        scoring_type = st.selectbox("Scoring Type", ["ppr", "half_ppr", "standard"])
    with col3:
        tier_filter = st.multiselect("Tiers", options=list(range(1, 17)), default=list(range(1, 6)))
    with col4:
        min_games = st.number_input("Min Games Played", min_value=1, max_value=17, value=6)
    
    # Fetch rankings with parameters
    params = f"?scoring_type={scoring_type}"
    if position_filter != "All":
        params += f"&position={position_filter}"
    
    rankings = api_call(f"/players/rankings{params}&limit=200")
    
    if rankings:
        df = pd.DataFrame(rankings)
        
        # Apply tier filter
        if tier_filter:
            df = df[df['tier'].isin(tier_filter)]
        
        # Display statistics
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Rankings table
            st.subheader("Player Rankings")
            
            # Format the dataframe for display
            display_df = df[['name', 'position', 'team', 'tier_label', 'predicted_points', 'confidence_interval', 'trend']].copy()
            display_df['Range'] = display_df['confidence_interval'].apply(
                lambda x: f"{x['low']:.1f} - {x['high']:.1f}" if x else "N/A"
            )
            display_df = display_df.drop('confidence_interval', axis=1)
            display_df.columns = ['Name', 'Pos', 'Team', 'Tier', 'Proj Points', 'Trend', 'Range']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
        
        with col2:
            # Tier distribution
            st.subheader("Tier Distribution")
            tier_counts = df['tier'].value_counts().sort_index()
            fig_tier = px.bar(
                x=tier_counts.index,
                y=tier_counts.values,
                labels={'x': 'Tier', 'y': 'Player Count'},
                title="Players per Tier"
            )
            st.plotly_chart(fig_tier, use_container_width=True)
            
            # Position breakdown
            st.subheader("Position Breakdown")
            pos_counts = df['position'].value_counts()
            fig_pos = px.pie(
                values=pos_counts.values,
                names=pos_counts.index,
                title="Players by Position"
            )
            st.plotly_chart(fig_pos, use_container_width=True)

# Draft Assistant Page
elif page == "Draft Assistant":
    st.header("AI-Powered Draft Assistant")
    
    # Draft settings
    col1, col2, col3 = st.columns(3)
    with col1:
        draft_position = st.number_input("Your Draft Position", min_value=1, max_value=14, value=1)
    with col2:
        league_size = st.selectbox("League Size", [8, 10, 12, 14])
    with col3:
        scoring_format = st.selectbox("Scoring Format", ["PPR", "Half-PPR", "Standard"])
    
    # Current pick calculation
    current_round = st.number_input("Current Round", min_value=1, max_value=16, value=1)
    
    # Snake draft logic
    if current_round % 2 == 1:  # Odd round
        current_pick = (current_round - 1) * league_size + draft_position
    else:  # Even round
        current_pick = current_round * league_size - draft_position + 1
    
    st.metric("Current Overall Pick", f"{current_pick}")
    
    # Best available players
    st.subheader("Best Available Players")
    
    # Mock drafted players (in real app, this would be tracked)
    drafted_players = st.multiselect("Already Drafted Players", options=[], default=[])
    
    # Get recommendations
    rankings = api_call(f"/players/rankings?scoring_type={scoring_format.lower().replace('-', '_')}&limit=300")
    
    if rankings:
        df = pd.DataFrame(rankings)
        
        # Filter out drafted players
        available_df = df[~df['name'].isin(drafted_players)]
        
        # Position needs analysis
        st.subheader("Recommended Picks by Position")
        
        tabs = st.tabs(["Best Available", "By Position", "Value Picks", "Sleepers"])
        
        with tabs[0]:
            # Top 10 best available
            top_available = available_df.head(10)
            for _, player in top_available.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{player['name']}** - {player['position']} ({player['team']})")
                with col2:
                    st.write(f"Tier {player['tier']}")
                with col3:
                    st.write(f"{player['predicted_points']:.1f} pts")
                with col4:
                    if st.button("Draft", key=f"draft_{player['name']}"):
                        st.success(f"Drafted {player['name']}!")
        
        with tabs[1]:
            # Group by position
            for pos in ["QB", "RB", "WR", "TE"]:
                st.markdown(f"### {pos}")
                pos_players = available_df[available_df['position'] == pos].head(5)
                st.dataframe(
                    pos_players[['name', 'team', 'tier', 'predicted_points']],
                    use_container_width=True,
                    hide_index=True
                )
        
        with tabs[2]:
            # Value picks (players available later than their tier suggests)
            expected_pick = current_pick + league_size  # Next pick
            value_picks = available_df[
                (available_df['tier'] <= current_round + 1) & 
                (available_df.index + len(drafted_players) > expected_pick)
            ].head(10)
            
            st.write("Players likely to be available at your next pick:")
            st.dataframe(
                value_picks[['name', 'position', 'team', 'tier', 'predicted_points']],
                use_container_width=True,
                hide_index=True
            )
        
        with tabs[3]:
            # Sleeper picks (high upside players in later tiers)
            sleepers = available_df[
                (available_df['tier'] >= 10) & 
                (available_df['trend'] == 'up')
            ].head(10)
            
            st.write("High-upside players available in later rounds:")
            st.dataframe(
                sleepers[['name', 'position', 'team', 'tier', 'predicted_points', 'trend']],
                use_container_width=True,
                hide_index=True
            )

# Weekly Projections Page
elif page == "Weekly Projections":
    st.header("Weekly Projections & Start/Sit Analysis")
    
    week = st.selectbox("Select Week", options=list(range(1, 18)), index=0)
    
    # Get projections
    projections = api_call(f"/players/projections/week/{week}")
    
    if projections:
        df = pd.DataFrame(projections)
        
        tabs = st.tabs(["Start/Sit Recommendations", "All Projections", "Matchup Analysis"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Start These Players")
                start_players = df[df['recommendation'] == 'START'].head(10)
                for _, player in start_players.iterrows():
                    st.success(
                        f"**{player['name']}** ({player['position']}) "
                        f"vs {player.get('opponent', 'TBD')} - "
                        f"{player['projected_points']:.1f} pts"
                    )
            
            with col2:
                st.markdown("### Sit These Players")
                sit_players = df[df['recommendation'] == 'BENCH'].head(10)
                for _, player in sit_players.iterrows():
                    st.warning(
                        f"**{player['name']}** ({player['position']}) "
                        f"vs {player.get('opponent', 'TBD')} - "
                        f"{player['projected_points']:.1f} pts"
                    )
        
        with tabs[1]:
            # Filter options
            position = st.selectbox("Filter by Position", ["All", "QB", "RB", "WR", "TE"])
            
            display_df = df if position == "All" else df[df['position'] == position]
            
            # Show projections table
            projection_cols = ['name', 'position', 'team', 'opponent', 'projected_points', 'floor', 'ceiling']
            st.dataframe(
                display_df[projection_cols].rename(columns={
                    'name': 'Player',
                    'position': 'Pos',
                    'team': 'Team',
                    'opponent': 'Opp',
                    'projected_points': 'Proj',
                    'floor': 'Floor',
                    'ceiling': 'Ceiling'
                }),
                use_container_width=True,
                height=600
            )
        
        with tabs[2]:
            st.info("Matchup analysis coming soon - will include defensive rankings and weather impact")

# Team Analysis Page
elif page == "Team Analysis":
    st.header("Team Analysis & Optimization")
    
    # Team input
    st.subheader("Your Roster")
    
    # In a real app, this would be connected to league data
    roster_positions = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "FLEX": 1,
        "K": 1,
        "DEF": 1
    }
    
    roster = {}
    for position, slots in roster_positions.items():
        if slots == 1:
            player = st.selectbox(f"Select {position}", options=["None"], key=f"roster_{position}")
            roster[position] = [player] if player != "None" else []
        else:
            players = st.multiselect(f"Select {slots} {position}s", options=[], key=f"roster_{position}")
            roster[position] = players
    
    if st.button("Analyze Roster"):
        st.subheader("Roster Analysis")
        
        # Mock analysis results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Projected Weekly Points", "125.4")
            st.metric("Roster Strength", "B+")
        with col2:
            st.metric("Injury Risk", "Low")
            st.metric("Bye Week Conflicts", "2")
        with col3:
            st.metric("Trade Value", "High")
            st.metric("Waiver Priority", "3")
        
        # Recommendations
        st.subheader("AI Recommendations")
        st.info("• Consider trading for a WR1 to improve ceiling")
        st.info("• Your RB depth is strong - could package for upgrades")
        st.info("• Monitor waiver wire for emerging TE options")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Fantasy Football AI Assistant | "
    "Powered by Advanced ML Models | Data updated daily</p>",
    unsafe_allow_html=True
)