"""
Enhanced Fantasy Football AI Streamlit App
Features tier badges, confidence intervals, and advanced visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any

# Configure page
st.set_page_config(
    page_title="Fantasy Football AI",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tier badges and styling
st.markdown("""
<style>
.tier-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    margin: 0.1rem;
    border-radius: 15px;
    font-weight: bold;
    font-size: 0.8rem;
    text-align: center;
    color: white;
}

.tier-1 { background: linear-gradient(45deg, #FF6B6B, #FF8E53); }
.tier-2 { background: linear-gradient(45deg, #4ECDC4, #44A08D); }
.tier-3 { background: linear-gradient(45deg, #45B7D1, #96C93D); }
.tier-4 { background: linear-gradient(45deg, #F7DC6F, #F39C12); }
.tier-5 { background: linear-gradient(45deg, #BB8FCE, #8E44AD); }
.tier-6 { background: linear-gradient(45deg, #85C1E9, #3498DB); }
.tier-7 { background: linear-gradient(45deg, #82E0AA, #27AE60); }
.tier-8 { background: linear-gradient(45deg, #F8C471, #E67E22); }
.tier-9 { background: linear-gradient(45deg, #F1948A, #E74C3C); }
.tier-10 { background: linear-gradient(45deg, #D7DBDD, #85929E); }
.tier-11 { background: linear-gradient(45deg, #AED6F1, #5DADE2); }
.tier-12 { background: linear-gradient(45deg, #A9DFBF, #58D68D); }
.tier-13 { background: linear-gradient(45deg, #F9E79F, #F4D03F); }
.tier-14 { background: linear-gradient(45deg, #FADBD8, #F1948A); }
.tier-15 { background: linear-gradient(45deg, #E8DAEF, #D2B4DE); }
.tier-16 { background: linear-gradient(45deg, #EBEDEF, #AEB6BF); }

.efficiency-high { color: #27AE60; font-weight: bold; }
.efficiency-medium { color: #F39C12; font-weight: bold; }
.efficiency-low { color: #E74C3C; font-weight: bold; }

.momentum-positive { color: #27AE60; }
.momentum-negative { color: #E74C3C; }
.momentum-neutral { color: #85929E; }

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}

.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #E74C3C 0%, #F39C12 50%, #27AE60 100%);
    position: relative;
}

.confidence-indicator {
    height: 12px;
    width: 3px;
    background: #2C3E50;
    border-radius: 2px;
    position: absolute;
    top: -2px;
}
</style>
""", unsafe_allow_html=True)


# Helper functions
def create_tier_badge(tier: int, tier_label: str) -> str:
    """Create HTML for tier badge"""
    return f'<span class="tier-badge tier-{tier}">Tier {tier}: {tier_label}</span>'


def create_efficiency_badge(efficiency: float) -> str:
    """Create HTML for efficiency rating"""
    if efficiency >= 1.2:
        css_class = "efficiency-high"
        label = "Elite"
    elif efficiency >= 1.0:
        css_class = "efficiency-medium" 
        label = "Good"
    else:
        css_class = "efficiency-low"
        label = "Below Average"
    
    return f'<span class="{css_class}">{efficiency:.3f} ({label})</span>'


def create_momentum_indicator(momentum: float, trend: str) -> str:
    """Create HTML for momentum indicator"""
    if momentum > 0.1:
        css_class = "momentum-positive"
        arrow = "↑"
    elif momentum < -0.1:
        css_class = "momentum-negative"
        arrow = "↓"
    else:
        css_class = "momentum-neutral"
        arrow = "→"
    
    return f'<span class="{css_class}">{arrow} {momentum:+.1%} ({trend})</span>'


def create_confidence_bar(confidence: float, width: int = 200) -> str:
    """Create HTML confidence bar with indicator"""
    position = confidence * width
    return f"""
    <div style="width: {width}px; position: relative;">
        <div class="confidence-bar"></div>
        <div class="confidence-indicator" style="left: {position}px;"></div>
        <div style="text-align: center; margin-top: 5px; font-size: 0.8rem;">
            {confidence:.1%} Confidence
        </div>
    </div>
    """


def mock_api_call(endpoint: str, params: Dict = None) -> Dict:
    """Mock API calls for demo purposes"""
    # In production, replace with actual API calls
    if "predictions" in endpoint:
        return {
            "player_name": "Jerry Jeudy",
            "position": "WR",
            "predictions": {
                "ensemble": {
                    "ppr": {
                        "point_estimate": 14.8,
                        "lower_bound": 8.2,
                        "upper_bound": 21.4
                    },
                    "standard": {
                        "point_estimate": 9.3,
                        "lower_bound": 4.1,
                        "upper_bound": 14.5
                    }
                }
            },
            "confidence": {"score": 0.78, "level": "High"},
            "draft_tier": {"tier": 3, "label": "WR1"},
            "momentum": {
                "score": 0.15,
                "trend": "up",
                "streak": {"type": "warm", "current": 2}
            },
            "efficiency_ratio": 1.14,
            "explanations": [
                "Models predict an average of 14.8 PPR points",
                "Player is in Tier 3 (WR1)",
                "Recent performance shows improvement",
                "Above-average efficiency in converting targets"
            ]
        }
    elif "efficiency" in endpoint:
        return {
            "efficiency_ratio": 1.14,
            "percentile_rank": 73.5,
            "efficiency_grade": "B+",
            "components": {
                "opportunity_efficiency": 1.08,
                "matchup_efficiency": 1.21,
                "game_script_efficiency": 1.09
            }
        }
    elif "momentum" in endpoint:
        return {
            "momentum_score": 0.15,
            "trend": "up",
            "indicators": {
                "momentum_3w": 0.12,
                "momentum_5w": 0.18,
                "consistency": 0.67
            },
            "predictions": {
                "breakout_probability": 0.31,
                "regression_probability": 0.18,
                "recommendation": "Buy"
            }
        }
    return {}


# Main app
def main():
    st.title("🏈 Fantasy Football AI")
    st.markdown("*Advanced ML-powered predictions with tier analysis and momentum detection*")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose Page",
            ["Player Analysis", "Tier Rankings", "Momentum Tracker", "Portfolio Optimizer"]
        )
        
        st.header("Settings")
        scoring_format = st.selectbox(
            "Scoring Format",
            ["PPR", "Standard", "Half-PPR"]
        )
        
        season = st.selectbox("Season", [2024, 2023, 2022])
        week = st.slider("Week", 1, 18, 10)
        
        st.header("Filters")
        positions = st.multiselect(
            "Positions",
            ["QB", "RB", "WR", "TE", "K"],
            default=["QB", "RB", "WR", "TE"]
        )
        
        tier_range = st.slider(
            "Tier Range",
            1, 16, (1, 8)
        )
    
    if page == "Player Analysis":
        player_analysis_page(scoring_format, season, week)
    elif page == "Tier Rankings":
        tier_rankings_page(positions, tier_range, scoring_format)
    elif page == "Momentum Tracker":
        momentum_tracker_page(positions, scoring_format)
    elif page == "Portfolio Optimizer":
        portfolio_optimizer_page(scoring_format)


def player_analysis_page(scoring_format: str, season: int, week: int):
    """Enhanced player analysis with tier badges and confidence intervals"""
    st.header("🔍 Player Analysis")
    
    # Player selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        player_name = st.text_input(
            "Enter Player Name",
            value="Jerry Jeudy",
            placeholder="e.g., Josh Allen, Derrick Henry"
        )
    
    with col2:
        analyze_btn = st.button("🚀 Analyze Player", type="primary")
    
    if analyze_btn or player_name:
        # Mock API call
        prediction_data = mock_api_call(f"/predictions/{player_name}")
        efficiency_data = mock_api_call(f"/efficiency/{player_name}")
        momentum_data = mock_api_call(f"/momentum/{player_name}")
        
        # Header with tier badge
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{prediction_data['player_name']} ({prediction_data['position']})")
            tier_html = create_tier_badge(
                prediction_data['draft_tier']['tier'],
                prediction_data['draft_tier']['label']
            )
            st.markdown(tier_html, unsafe_allow_html=True)
        
        with col2:
            efficiency_html = create_efficiency_badge(prediction_data['efficiency_ratio'])
            st.markdown(f"**Efficiency:** {efficiency_html}", unsafe_allow_html=True)
        
        with col3:
            momentum_html = create_momentum_indicator(
                prediction_data['momentum']['score'],
                prediction_data['momentum']['trend']
            )
            st.markdown(f"**Momentum:** {momentum_html}", unsafe_allow_html=True)
        
        # Main metrics with confidence intervals
        st.markdown("### 📊 Predictions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        ppr_pred = prediction_data['predictions']['ensemble']['ppr']
        std_pred = prediction_data['predictions']['ensemble']['standard']
        confidence = prediction_data['confidence']['score']
        
        with col1:
            st.metric(
                f"{scoring_format} Points",
                f"{ppr_pred['point_estimate']:.1f}",
                delta=f"±{(ppr_pred['upper_bound'] - ppr_pred['lower_bound'])/2:.1f}"
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta=prediction_data['confidence']['level']
            )
        
        with col3:
            st.metric(
                "Efficiency Grade",
                efficiency_data['efficiency_grade'],
                delta=f"{efficiency_data['percentile_rank']:.0f}th %ile"
            )
        
        with col4:
            breakout_prob = momentum_data['predictions']['breakout_probability']
            st.metric(
                "Breakout Probability",
                f"{breakout_prob:.1%}",
                delta=momentum_data['predictions']['recommendation']
            )
        
        # Confidence interval visualization
        st.markdown("### 📈 Prediction Range")
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=[ppr_pred['lower_bound'], ppr_pred['upper_bound']],
            y=[1, 1],
            mode='lines+markers',
            line=dict(width=8, color='rgba(74, 144, 226, 0.6)'),
            marker=dict(size=12, color=['#E74C3C', '#27AE60']),
            name='Confidence Interval',
            hovertemplate='<b>%{y}</b><br>Points: %{x:.1f}<extra></extra>'
        ))
        
        # Add point estimate
        fig.add_trace(go.Scatter(
            x=[ppr_pred['point_estimate']],
            y=[1],
            mode='markers',
            marker=dict(size=16, color='#2C3E50', symbol='diamond'),
            name='Prediction',
            hovertemplate='<b>Prediction</b><br>Points: %{x:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{scoring_format} Points Prediction with {confidence:.0%} Confidence",
            xaxis_title="Fantasy Points",
            yaxis=dict(visible=False),
            height=200,
            showlegend=True,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Efficiency Breakdown")
            
            # Efficiency components chart
            categories = ['Opportunity', 'Matchup', 'Game Script']
            values = [
                efficiency_data['components']['opportunity_efficiency'],
                efficiency_data['components']['matchup_efficiency'],
                efficiency_data['components']['game_script_efficiency']
            ]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                marker_color='rgba(74, 144, 226, 0.6)',
                line_color='rgba(74, 144, 226, 1)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.5, 1.5]
                    )
                ),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Momentum Indicators")
            
            # Momentum chart
            periods = ['3-Week', '5-Week', '10-Week']
            momentum_values = [
                momentum_data['indicators']['momentum_3w'],
                momentum_data['indicators']['momentum_5w'],
                momentum_data['indicators'].get('momentum_10w', 0.08)
            ]
            
            colors = ['#27AE60' if x > 0 else '#E74C3C' for x in momentum_values]
            
            fig = go.Figure(data=go.Bar(
                x=periods,
                y=momentum_values,
                marker_color=colors,
                text=[f"{x:+.1%}" for x in momentum_values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Momentum by Time Period",
                yaxis_title="Momentum Score",
                height=300,
                yaxis_tickformat='.1%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanations
        st.markdown("### 💡 Key Insights")
        
        for i, explanation in enumerate(prediction_data['explanations'], 1):
            st.markdown(f"**{i}.** {explanation}")


def tier_rankings_page(positions: List[str], tier_range: tuple, scoring_format: str):
    """Enhanced tier rankings with visual badges"""
    st.header("🏆 Tier Rankings")
    
    # Mock tier data
    tier_data = []
    for pos in positions:
        for tier in range(tier_range[0], tier_range[1] + 1):
            tier_data.extend([
                {
                    'Player': f'Player {tier}A',
                    'Position': pos,
                    'Tier': tier,
                    'Tier_Label': f'{pos}{tier}',
                    'Predicted_Points': 20 - tier + np.random.normal(0, 2),
                    'Efficiency': 1.3 - (tier * 0.05) + np.random.normal(0, 0.1),
                    'Confidence': 0.9 - (tier * 0.02) + np.random.normal(0, 0.05)
                }
            ])
    
    df = pd.DataFrame(tier_data)
    df = df.sort_values(['Tier', 'Predicted_Points'], ascending=[True, False])
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_efficiency = st.slider("Min Efficiency", 0.5, 1.5, 0.8, 0.1)
    
    with col2:
        min_confidence = st.slider("Min Confidence", 0.5, 1.0, 0.6, 0.05)
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Tier", "Predicted Points", "Efficiency", "Confidence"]
        )
    
    # Filter data
    filtered_df = df[
        (df['Efficiency'] >= min_efficiency) &
        (df['Confidence'] >= min_confidence)
    ]
    
    # Display tier rankings
    st.markdown("### 📋 Rankings")
    
    for _, row in filtered_df.head(20).iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 2])
        
        with col1:
            tier_badge = create_tier_badge(row['Tier'], row['Tier_Label'])
            st.markdown(f"**{row['Player']}** ({row['Position']}) {tier_badge}", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Points", f"{row['Predicted_Points']:.1f}")
        
        with col3:
            efficiency_badge = create_efficiency_badge(row['Efficiency'])
            st.markdown(efficiency_badge, unsafe_allow_html=True)
        
        with col4:
            confidence_bar = create_confidence_bar(row['Confidence'], 150)
            st.markdown(confidence_bar, unsafe_allow_html=True)
        
        with col5:
            if row['Efficiency'] > 1.1:
                st.success("🔥 Value Pick")
            elif row['Confidence'] > 0.85:
                st.info("🎯 Safe Play")
            else:
                st.warning("⚠️ Risky")


def momentum_tracker_page(positions: List[str], scoring_format: str):
    """Momentum tracking with alerts"""
    st.header("🚀 Momentum Tracker")
    
    # Breakout candidates
    st.subheader("🔥 Breakout Candidates")
    
    breakout_data = [
        {'Player': 'Tank Dell', 'Position': 'WR', 'Momentum': 0.28, 'Breakout_Prob': 0.74, 'Trend': 'strong_up'},
        {'Player': 'De\'Von Achane', 'Position': 'RB', 'Momentum': 0.31, 'Breakout_Prob': 0.68, 'Trend': 'up'},
        {'Player': 'Jordan Addison', 'Position': 'WR', 'Momentum': 0.22, 'Breakout_Prob': 0.61, 'Trend': 'up'},
    ]
    
    for player in breakout_data:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{player['Player']}** ({player['Position']})")
            
            with col2:
                momentum_html = create_momentum_indicator(player['Momentum'], player['Trend'])
                st.markdown(momentum_html, unsafe_allow_html=True)
            
            with col3:
                st.metric("Breakout Prob", f"{player['Breakout_Prob']:.0%}")
            
            with col4:
                st.success("🚀 Buy Alert")
    
    # Regression candidates
    st.subheader("📉 Regression Watch")
    
    regression_data = [
        {'Player': 'Puka Nacua', 'Position': 'WR', 'Momentum': -0.15, 'Regression_Prob': 0.58, 'Trend': 'down'},
        {'Player': 'Rachaad White', 'Position': 'RB', 'Momentum': -0.22, 'Regression_Prob': 0.71, 'Trend': 'strong_down'},
    ]
    
    for player in regression_data:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{player['Player']}** ({player['Position']})")
            
            with col2:
                momentum_html = create_momentum_indicator(player['Momentum'], player['Trend'])
                st.markdown(momentum_html, unsafe_allow_html=True)
            
            with col3:
                st.metric("Regression Prob", f"{player['Regression_Prob']:.0%}")
            
            with col4:
                st.error("⚠️ Sell Alert")
    
    # Momentum visualization
    st.subheader("📊 Momentum Distribution")
    
    # Create sample data
    np.random.seed(42)
    momentum_dist = np.random.normal(0, 0.15, 200)
    
    fig = go.Figure(data=go.Histogram(
        x=momentum_dist,
        nbinsx=20,
        marker_color='rgba(74, 144, 226, 0.7)',
        name='Players'
    ))
    
    # Add vertical lines for breakout/regression thresholds
    fig.add_vline(x=0.2, line_dash="dash", line_color="green", 
                  annotation_text="Breakout Threshold")
    fig.add_vline(x=-0.2, line_dash="dash", line_color="red", 
                  annotation_text="Regression Threshold")
    
    fig.update_layout(
        title="Player Momentum Distribution",
        xaxis_title="Momentum Score",
        yaxis_title="Number of Players",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def portfolio_optimizer_page(scoring_format: str):
    """Portfolio optimization tool"""
    st.header("⚖️ Portfolio Optimizer")
    
    st.markdown("*Optimize your lineup based on predictions, tiers, and correlations*")
    
    # Budget constraints
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_budget = st.number_input("Total Budget", value=200, min_value=100, max_value=500)
    
    with col2:
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
    
    with col3:
        optimize_for = st.selectbox("Optimize For", ["Ceiling", "Floor", "Expected Value"])
    
    # Position requirements
    st.subheader("Position Requirements")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        qb_count = st.selectbox("QB", [1, 2], index=0)
    
    with col2:
        rb_count = st.selectbox("RB", [2, 3, 4], index=0)
    
    with col3:
        wr_count = st.selectbox("WR", [2, 3, 4], index=1)
    
    with col4:
        te_count = st.selectbox("TE", [1, 2], index=0)
    
    if st.button("🎯 Optimize Lineup", type="primary"):
        # Mock optimization result
        optimal_lineup = [
            {'Player': 'Josh Allen', 'Position': 'QB', 'Salary': 45, 'Projected': 22.8, 'Tier': 1},
            {'Player': 'Christian McCaffrey', 'Position': 'RB', 'Salary': 55, 'Projected': 19.4, 'Tier': 1},
            {'Player': 'Alvin Kamara', 'Position': 'RB', 'Salary': 42, 'Projected': 16.8, 'Tier': 2},
            {'Player': 'Tyreek Hill', 'Position': 'WR', 'Salary': 48, 'Projected': 18.6, 'Tier': 1},
            {'Player': 'CeeDee Lamb', 'Position': 'WR', 'Salary': 46, 'Projected': 17.9, 'Tier': 1},
            {'Player': 'Amon-Ra St. Brown', 'Position': 'WR', 'Salary': 38, 'Projected': 15.2, 'Tier': 2},
            {'Player': 'Travis Kelce', 'Position': 'TE', 'Salary': 42, 'Projected': 14.8, 'Tier': 1},
        ]
        
        lineup_df = pd.DataFrame(optimal_lineup)
        
        st.success("✅ Optimization Complete!")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_salary = lineup_df['Salary'].sum()
        total_projected = lineup_df['Projected'].sum() 
        
        with col1:
            st.metric("Total Salary", f"${total_salary}", f"${total_budget - total_salary} remaining")
        
        with col2:
            st.metric("Projected Points", f"{total_projected:.1f}")
        
        with col3:
            st.metric("Points per $", f"{total_projected/total_salary:.2f}")
        
        with col4:
            tier_1_count = len(lineup_df[lineup_df['Tier'] == 1])
            st.metric("Tier 1 Players", tier_1_count)
        
        # Lineup display
        st.subheader("📋 Optimal Lineup")
        
        for _, player in lineup_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                tier_badge = create_tier_badge(player['Tier'], f"{player['Position']}{player['Tier']}")
                st.markdown(f"**{player['Player']}** ({player['Position']}) {tier_badge}", 
                           unsafe_allow_html=True)
            
            with col2:
                st.metric("Salary", f"${player['Salary']}")
            
            with col3:
                st.metric("Projected", f"{player['Projected']:.1f}")
            
            with col4:
                st.metric("Value", f"{player['Projected']/player['Salary']:.2f}")
            
            with col5:
                if player['Projected']/player['Salary'] > 0.4:
                    st.success("💎 Great Value")
                elif player['Tier'] == 1:
                    st.info("🔒 Safe Pick")
                else:
                    st.warning("⚠️ Risky")


if __name__ == "__main__":
    main()