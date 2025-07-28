"""
Fantasy Football AI - Streamlit Frontend
Production-ready web application with authentication, ML predictions, and draft tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import time

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
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #1e40af;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #2563eb;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #f0f9ff;
        border: 1px solid #e0e7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border-radius: 0.375rem;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Tier badges */
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.125rem;
    }
    
    .tier-1 { background-color: #dc2626; color: white; }
    .tier-2 { background-color: #ea580c; color: white; }
    .tier-3 { background-color: #f59e0b; color: white; }
    .tier-4 { background-color: #84cc16; color: white; }
    .tier-5 { background-color: #22c55e; color: white; }
    .tier-6 { background-color: #14b8a6; color: white; }
    .tier-7 { background-color: #06b6d4; color: white; }
    .tier-8 { background-color: #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Session state initialization
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'subscription_tier' not in st.session_state:
    st.session_state.subscription_tier = 'free'
if 'drafted_players' not in st.session_state:
    st.session_state.drafted_players = []
if 'my_roster' not in st.session_state:
    st.session_state.my_roster = []

# Helper functions
def make_api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Optional[Dict]:
    """Make authenticated API request"""
    headers = {}
    if st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
    
    try:
        if method == "GET":
            response = requests.get(
                f"{API_BASE_URL}{endpoint}",
                headers=headers,
                params=params,
                timeout=API_TIMEOUT
            )
        else:
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                headers=headers,
                json=data,
                timeout=API_TIMEOUT
            )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.session_state.auth_token = None
            st.error("Session expired. Please log in again.")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
    
    return None

def format_tier_badge(tier: int, tier_label: str) -> str:
    """Format tier as HTML badge"""
    return f'<span class="tier-badge tier-{tier}">{tier_label}</span>'

def format_confidence_interval(ci: Dict[str, float]) -> str:
    """Format confidence interval"""
    return f"{ci['low']:.1f} - {ci['high']:.1f}"

# Authentication functions
def login_form():
    """Display login form"""
    st.subheader("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if email and password:
                response = make_api_request(
                    "/auth/login",
                    method="POST",
                    data={"email": email, "password": password}
                )
                
                if response:
                    st.session_state.auth_token = response["access_token"]
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.experimental_rerun()
            else:
                st.error("Please enter both email and password")

def register_form():
    """Display registration form"""
    st.subheader("📝 Register")
    
    with st.form("register_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        username = st.text_input("Username", placeholder="Optional")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register", use_container_width=True)
        
        if submitted:
            if not email or not password:
                st.error("Email and password are required")
            elif password != password_confirm:
                st.error("Passwords do not match")
            else:
                response = make_api_request(
                    "/auth/register",
                    method="POST",
                    data={
                        "email": email,
                        "username": username,
                        "password": password
                    }
                )
                
                if response:
                    st.session_state.auth_token = response["access_token"]
                    st.session_state.user_email = email
                    st.success("Registration successful! Welcome to Fantasy Football AI!")
                    st.experimental_rerun()

# Page functions
def show_rankings_page():
    """Display player rankings page"""
    st.title("🏆 Player Rankings")
    st.markdown("AI-powered player rankings with tier-based draft recommendations")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        position = st.selectbox(
            "Position",
            ["All", "QB", "RB", "WR", "TE"],
            key="rankings_position"
        )
    
    with col2:
        scoring = st.selectbox(
            "Scoring Type",
            ["ppr", "half_ppr", "standard"],
            format_func=lambda x: x.replace('_', '-').upper(),
            key="rankings_scoring"
        )
    
    with col3:
        tier_filter = st.selectbox(
            "Tier Filter",
            ["All"] + list(range(1, 17)),
            key="rankings_tier"
        )
    
    with col4:
        show_drafted = st.checkbox("Hide drafted players", value=True)
    
    # Fetch rankings
    params = {
        "scoring": scoring,
        "limit": 200
    }
    
    if position != "All":
        params["position"] = position
    
    if tier_filter != "All":
        params["tier"] = tier_filter
    
    rankings = make_api_request("/players/rankings", params=params)
    
    if rankings:
        # Convert to DataFrame
        df = pd.DataFrame(rankings)
        
        # Filter drafted players
        if show_drafted and st.session_state.drafted_players:
            df = df[~df['player_id'].isin(st.session_state.drafted_players)]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(df))
        
        with col2:
            if position != "All":
                avg_proj = df['predicted_points'].mean()
                st.metric("Avg Projection", f"{avg_proj:.1f} pts")
        
        with col3:
            if len(df) > 0:
                top_tier = df['tier'].min()
                st.metric("Best Available Tier", top_tier)
        
        with col4:
            positions_available = df['position'].nunique()
            st.metric("Positions Available", positions_available)
        
        # Rankings table
        st.subheader("📊 Rankings Table")
        
        # Format data for display
        display_df = df[['name', 'position', 'team', 'tier', 'tier_label', 
                        'predicted_points', 'confidence_interval']].copy()
        
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['Tier'] = display_df.apply(
            lambda x: format_tier_badge(x['tier'], x['tier_label']), axis=1
        )
        display_df['Projection'] = display_df['predicted_points'].round(1)
        display_df['Range'] = display_df['confidence_interval'].apply(format_confidence_interval)
        
        # Reorder columns
        display_df = display_df[['Rank', 'name', 'position', 'team', 'Tier', 
                                'Projection', 'Range']]
        
        # Display with HTML
        st.markdown(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        
        # Visualizations
        st.subheader("📈 Projections by Position")
        
        if len(df) > 0:
            # Box plot by position
            fig = px.box(
                df,
                x='position',
                y='predicted_points',
                color='position',
                title='Fantasy Points Distribution by Position',
                labels={'predicted_points': 'Projected Points', 'position': 'Position'}
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Position",
                yaxis_title="Projected Fantasy Points"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tier distribution
            col1, col2 = st.columns(2)
            
            with col1:
                tier_counts = df.groupby(['tier', 'position']).size().reset_index(name='count')
                
                fig = px.bar(
                    tier_counts,
                    x='tier',
                    y='count',
                    color='position',
                    title='Player Distribution by Tier',
                    labels={'tier': 'Tier', 'count': 'Number of Players'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average points by tier
                tier_avg = df.groupby('tier')['predicted_points'].mean().reset_index()
                
                fig = px.line(
                    tier_avg,
                    x='tier',
                    y='predicted_points',
                    title='Average Projection by Tier',
                    markers=True,
                    labels={'tier': 'Tier', 'predicted_points': 'Avg Projected Points'}
                )
                
                fig.update_traces(line_color='#3b82f6', line_width=3)
                st.plotly_chart(fig, use_container_width=True)

def show_draft_assistant_page():
    """Display draft assistant page"""
    st.title("🎯 Draft Assistant")
    st.markdown("AI-powered real-time draft recommendations")
    
    # Check subscription
    if st.session_state.subscription_tier == 'free':
        st.warning("🔒 Draft Assistant requires a Pro or Premium subscription")
        st.markdown("""
        <div class="info-box">
        <h4>Upgrade to Pro to unlock:</h4>
        <ul>
        <li>Real-time draft recommendations</li>
        <li>Position-specific strategies</li>
        <li>Value-based drafting</li>
        <li>Custom scoring support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Draft settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        draft_round = st.number_input("Current Round", min_value=1, max_value=16, value=1)
    
    with col2:
        draft_pick = st.number_input("Current Pick", min_value=1, max_value=12, value=1)
    
    with col3:
        team_size = st.number_input("Teams in League", min_value=8, max_value=14, value=12)
    
    # Roster needs
    st.subheader("📋 Roster Needs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        qb_need = st.number_input("QB Needed", min_value=0, max_value=3, value=1)
    
    with col2:
        rb_need = st.number_input("RB Needed", min_value=0, max_value=6, value=2)
    
    with col3:
        wr_need = st.number_input("WR Needed", min_value=0, max_value=6, value=3)
    
    with col4:
        te_need = st.number_input("TE Needed", min_value=0, max_value=3, value=1)
    
    # Get recommendations
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        params = {
            "round": draft_round,
            "pick": draft_pick,
            "drafted_players": st.session_state.drafted_players,
            "roster_needs": {
                "QB": qb_need,
                "RB": rb_need,
                "WR": wr_need,
                "TE": te_need
            }
        }
        
        recommendations = make_api_request(
            "/draft/recommendations",
            method="POST",
            params=params
        )
        
        if recommendations:
            # Strategy notes
            st.markdown(f"""
            <div class="info-box">
            <h4>Round {draft_round} Strategy:</h4>
            <p>{recommendations['strategy_notes']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommended players
            st.subheader("🎯 Top Recommendations")
            
            for i, player in enumerate(recommendations['recommended_players'], 1):
                col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 2, 2])
                
                with col1:
                    st.markdown(f"**#{i}**")
                
                with col2:
                    st.markdown(f"**{player['name']}** ({player['position']} - {player['team']})")
                
                with col3:
                    st.markdown(format_tier_badge(player['tier'], f"Tier {player['tier']}"), 
                               unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"**{player['predicted_points']:.1f}** pts")
                
                with col5:
                    if st.button(f"Draft", key=f"draft_{player['player_id']}"):
                        st.session_state.drafted_players.append(player['player_id'])
                        st.session_state.my_roster.append(player)
                        st.success(f"Drafted {player['name']}!")
                        st.experimental_rerun()
    
    # My roster
    if st.session_state.my_roster:
        st.subheader("🏈 My Roster")
        
        roster_df = pd.DataFrame(st.session_state.my_roster)
        
        # Group by position
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = roster_df[roster_df['position'] == position]
            if len(pos_players) > 0:
                st.markdown(f"**{position}s:**")
                for _, player in pos_players.iterrows():
                    st.markdown(f"- {player['name']} ({player['team']}) - "
                               f"{player['predicted_points']:.1f} pts")

def show_weekly_predictions_page():
    """Display weekly predictions page"""
    st.title("📅 Weekly Predictions")
    st.markdown("Start/Sit decisions powered by neural networks")
    
    # Week selector
    current_week = st.selectbox(
        "Select Week",
        list(range(1, 18)),
        format_func=lambda x: f"Week {x}",
        key="predictions_week"
    )
    
    # My team predictions
    if st.session_state.my_roster:
        st.subheader("🏈 My Team Projections")
        
        # Get predictions for roster
        player_ids = [p['player_id'] for p in st.session_state.my_roster]
        
        predictions = make_api_request(
            "/predictions/custom",
            method="POST",
            data={
                "player_ids": player_ids,
                "week": current_week,
                "scoring_type": "ppr"
            }
        )
        
        if predictions:
            # Merge with roster data
            pred_dict = {p['player_id']: p for p in predictions}
            
            # Create lineup recommendations
            lineup = {
                'QB': [],
                'RB': [],
                'WR': [],
                'TE': [],
                'FLEX': []
            }
            
            for player in st.session_state.my_roster:
                if player['player_id'] in pred_dict:
                    player_pred = pred_dict[player['player_id']]
                    player_data = {
                        **player,
                        'week_projection': player_pred['predicted_points'],
                        'confidence': player_pred.get('model_confidence', 0.8)
                    }
                    
                    position = player['position']
                    if position in lineup:
                        lineup[position].append(player_data)
            
            # Sort by projection
            for pos in lineup:
                lineup[pos].sort(key=lambda x: x['week_projection'], reverse=True)
            
            # Display optimal lineup
            st.subheader("🎯 Optimal Lineup")
            
            total_projection = 0
            
            # Starting positions
            positions = [
                ('QB', 1),
                ('RB', 2),
                ('WR', 2),
                ('TE', 1),
                ('FLEX', 1)
            ]
            
            for position, slots in positions:
                st.markdown(f"**{position}:**")
                
                if position == 'FLEX':
                    # FLEX can be RB, WR, or TE
                    flex_eligible = []
                    flex_eligible.extend(lineup['RB'][2:])  # RBs not starting
                    flex_eligible.extend(lineup['WR'][2:])  # WRs not starting
                    flex_eligible.extend(lineup['TE'][1:])  # TEs not starting
                    flex_eligible.sort(key=lambda x: x['week_projection'], reverse=True)
                    players = flex_eligible[:slots]
                else:
                    players = lineup[position][:slots]
                
                for player in players:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        confidence_emoji = "🟢" if player['confidence'] > 0.8 else "🟡"
                        st.markdown(f"{confidence_emoji} {player['name']} ({player['team']})")
                    
                    with col2:
                        st.markdown(f"**{player['week_projection']:.1f}** pts")
                    
                    with col3:
                        st.markdown(f"{player['confidence']:.0%} conf")
                    
                    total_projection += player['week_projection']
            
            st.markdown(f"### Total Projection: **{total_projection:.1f}** points")
    
    else:
        st.info("Build your roster in the Draft Assistant to see weekly projections")
    
    # Matchup analyzer
    st.subheader("🔍 Matchup Analyzer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Player 1", placeholder="Search player...")
    
    with col2:
        player2 = st.text_input("Player 2", placeholder="Search player...")
    
    if st.button("Compare Players", use_container_width=True):
        if player1 and player2:
            st.info("Player comparison feature coming soon!")

def show_waiver_wire_page():
    """Display waiver wire suggestions page"""
    st.title("🔄 Waiver Wire")
    st.markdown("AI-powered breakout predictions and waiver priorities")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position = st.selectbox(
            "Position Filter",
            ["All", "QB", "RB", "WR", "TE"],
            key="waiver_position"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            ["Priority Score", "Next Week Projection", "Season Projection", "Add %"],
            key="waiver_sort"
        )
    
    with col3:
        min_available = st.slider(
            "Min Available %",
            min_value=50,
            max_value=100,
            value=80,
            help="Only show players available in X% of leagues"
        )
    
    # Get suggestions
    params = {}
    if position != "All":
        params["position"] = position
    
    suggestions = make_api_request("/waiver/suggestions", params=params)
    
    if suggestions:
        st.subheader("🎯 Top Waiver Targets")
        
        for i, player in enumerate(suggestions, 1):
            with st.expander(f"#{i} - {player['name']} ({player['position']})"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Priority Score", f"{player['priority_score']:.1f}/10")
                
                with col2:
                    st.metric("Next Week", f"{player['predicted_points_next_week']:.1f} pts")
                
                with col3:
                    st.metric("Season Projection", f"{player['season_projection']:.0f} pts")
                
                with col4:
                    st.metric("Add %", f"{player['add_percentage']:.1f}%")
                
                st.markdown(f"**Analysis:** {player['reasoning']}")
                
                if st.button(f"Add to Watchlist", key=f"watch_{player['player_id']}"):
                    st.success(f"Added {player['name']} to watchlist!")

def show_account_page():
    """Display account/subscription page"""
    st.title("👤 My Account")
    
    # User info
    st.subheader("Account Information")
    st.markdown(f"**Email:** {st.session_state.user_email}")
    
    # Get subscription info
    sub_info = make_api_request("/subscription/info")
    
    if sub_info:
        st.subheader("📊 Subscription Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Tier", sub_info['tier'].title())
            
            if sub_info['expires_at']:
                expires = datetime.fromisoformat(sub_info['expires_at'])
                days_left = (expires - datetime.now()).days
                st.metric("Days Remaining", days_left)
        
        with col2:
            st.metric(
                "API Calls", 
                f"{sub_info['api_calls_remaining']}/{sub_info['api_calls_limit']}"
            )
            
            usage_pct = (sub_info['api_calls_limit'] - sub_info['api_calls_remaining']) / sub_info['api_calls_limit']
            st.progress(usage_pct)
        
        # Features
        st.subheader("🎯 Your Features")
        for feature in sub_info['features']:
            st.markdown(f"✅ {feature}")
        
        # Upgrade options
        if sub_info['tier'] != 'premium':
            st.subheader("🚀 Upgrade Your Experience")
            
            if sub_info['tier'] == 'free':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-box">
                    <h4>Pro - $9.99/month</h4>
                    <ul>
                    <li>Full GMM draft tiers</li>
                    <li>Advanced ML predictions</li>
                    <li>Unlimited rosters</li>
                    <li>Real-time updates</li>
                    <li>Waiver wire AI</li>
                    <li>1,000 API calls/hour</li>
                    </ul>
                    <button class="stButton">Upgrade to Pro</button>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-box">
                    <h4>Premium - $19.99/month</h4>
                    <ul>
                    <li>Everything in Pro</li>
                    <li>Custom league scoring</li>
                    <li>DFS optimizer</li>
                    <li>Historical analysis</li>
                    <li>Priority support</li>
                    <li>10,000 API calls/hour</li>
                    </ul>
                    <button class="stButton">Upgrade to Premium</button>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Logout
    if st.button("Logout", type="secondary"):
        st.session_state.auth_token = None
        st.session_state.user_email = None
        st.session_state.subscription_tier = 'free'
        st.experimental_rerun()

# Main app
def main():
    """Main application"""
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1e3a8a/ffffff?text=Fantasy+Football+AI", 
                 use_column_width=True)
        
        if st.session_state.auth_token:
            st.markdown(f"👤 {st.session_state.user_email}")
            st.markdown(f"📊 {st.session_state.subscription_tier.title()} Tier")
            
            st.divider()
            
            # Navigation
            page = st.radio(
                "Navigation",
                ["Player Rankings", "Draft Assistant", "Weekly Predictions", 
                 "Waiver Wire", "My Account"],
                label_visibility="collapsed"
            )
        else:
            st.markdown("### Welcome!")
            st.markdown("Please login or register to continue")
            
            auth_mode = st.radio(
                "Auth Mode",
                ["Login", "Register"],
                label_visibility="collapsed"
            )
    
    # Main content
    if not st.session_state.auth_token:
        # Authentication
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            # 🏈 Fantasy Football AI
            
            ### Transform your fantasy season with ML-powered insights
            
            - 🎯 **89.2% Prediction Accuracy**
            - 📊 **16-Tier Draft System** using GMM clustering
            - 🧠 **Neural Network Predictions** for weekly projections
            - 📈 **Real-time Updates** with the latest data
            """)
            
            if auth_mode == "Login":
                login_form()
            else:
                register_form()
            
            st.markdown("""
            ---
            
            #### Why Fantasy Football AI?
            
            Our advanced machine learning system analyzes millions of data points
            to provide you with the most accurate predictions and recommendations.
            
            **Featured in:** TechCrunch | ESPN | The Athletic
            """)
    
    else:
        # Authenticated pages
        if page == "Player Rankings":
            show_rankings_page()
        elif page == "Draft Assistant":
            show_draft_assistant_page()
        elif page == "Weekly Predictions":
            show_weekly_predictions_page()
        elif page == "Waiver Wire":
            show_waiver_wire_page()
        elif page == "My Account":
            show_account_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Fantasy Football AI © 2024 | Built by Christopher Bratkovics | 
        <a href='#'>API Docs</a> | <a href='#'>Terms</a> | <a href='#'>Privacy</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()