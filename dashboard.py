import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pytz

# Try to import plotly, fall back gracefully if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Configuration
@dataclass
class AppConfig:
    """Simple configuration for the app"""
    
    SECTORS: Dict[str, List[str]] = field(default_factory=lambda: {
        "NIFTY50": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        "BANKNIFTY": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
        "PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "AUROPHARMA"],
        "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "EICHERMOT", "HEROMOTOCO"],
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        "GOLD": ["GOLD_SPOT", "GLD_ETF", "GOLDIAM", "TITAN"],
        "CRYPTO": ["BTC_USD", "ETH_USD", "BNB_USD", "ADA_USD"]
    })

config = AppConfig()

# Scoring system
class SimpleScoring:
    def __init__(self):
        self.planetary_weights = {
            "Jupiter": 2.0, "Venus": 1.5, "Mercury": 1.0, "Moon": 1.2,
            "Saturn": -2.0, "Mars": -1.5, "Rahu": -1.8, "Ketu": -1.2
        }
        
        self.aspect_weights = {
            "Trine": 1.2, "Sextile": 1.0, "Conjunction": 0.8,
            "Opposition": -1.0, "Square": -1.2
        }
    
    def calculate_sector_score(self, sector_name: str, date_str: str) -> float:
        """Calculate a simple score for a sector based on hash"""
        # Use hash for consistent but pseudo-random scoring
        hash_val = hash(sector_name + date_str) % 1000
        score = (hash_val - 500) / 100  # Range from -5 to +5
        return round(score, 2)
    
    def get_signal_strength(self, score: float) -> str:
        """Get signal strength based on score"""
        abs_score = abs(score)
        if abs_score >= 3.0:
            return "Very Strong"
        elif abs_score >= 2.0:
            return "Strong"
        elif abs_score >= 1.0:
            return "Moderate"
        elif abs_score >= 0.5:
            return "Weak"
        else:
            return "Minimal"
    
    def get_trend_direction(self, score: float) -> str:
        """Get trend direction"""
        if score >= 1.0:
            return "Bullish"
        elif score <= -1.0:
            return "Bearish"
        else:
            return "Neutral"

# Initialize scoring engine
scoring = SimpleScoring()

# Styling
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 5px solid #3b82f6;
            margin-bottom: 1rem;
        }
        
        .bullish-card {
            border-left-color: #16a34a;
            background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
        }
        
        .bearish-card {
            border-left-color: #dc2626;
            background: linear-gradient(135deg, #fef2f2 0%, #fefefe 100%);
        }
        
        .neutral-card {
            border-left-color: #ca8a04;
            background: linear-gradient(135deg, #fefce8 0%, #fffef7 100%);
        }
        
        .alert-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .alert-success {
            background-color: #dcfce7;
            border-left-color: #16a34a;
            color: #15803d;
        }
        
        .alert-warning {
            background-color: #fef3c7;
            border-left-color: #d97706;
            color: #92400e;
        }
        
        .alert-danger {
            background-color: #fef2f2;
            border-left-color: #dc2626;
            color: #b91c1c;
        }
    </style>
    """, unsafe_allow_html=True)

# Generate sample data
@st.cache_data(ttl=3600)
def generate_sample_aspects(date_input):
    """Generate sample planetary aspects"""
    aspects_data = []
    
    planets = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
    aspect_types = ["Conjunction", "Sextile", "Square", "Trine", "Opposition"]
    
    base_time = datetime.combine(date_input, dtime(9, 0))
    
    for i in range(10):  # Generate 10 sample aspects
        time_offset = timedelta(hours=i, minutes=30)
        event_time = base_time + time_offset
        
        planet_a = planets[i % len(planets)]
        planet_b = planets[(i + 2) % len(planets)]
        aspect = aspect_types[i % len(aspect_types)]
        
        # Calculate score
        score = scoring.planetary_weights.get(planet_a, 0) + scoring.planetary_weights.get(planet_b, 0)
        score *= scoring.aspect_weights.get(aspect, 1.0)
        score = round(score, 2)
        
        aspects_data.append({
            "Time": event_time.strftime("%H:%M"),
            "Planet A": planet_a,
            "Planet B": planet_b,
            "Aspect": aspect,
            "Score": score,
            "Signal": scoring.get_trend_direction(score),
            "Strength": scoring.get_signal_strength(score)
        })
    
    return pd.DataFrame(aspects_data)

# Main UI Components
def create_main_header():
    st.markdown("""
    <div class="main-header">
        <h1>🪐 Vedic Market Analytics</h1>
        <p>Advanced Astrological Market Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar controls"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    with st.sidebar.expander("📅 Date & Time", expanded=True):
        date_input = st.date_input("Analysis Date", value=datetime.today().date())
        timezone = st.selectbox("Timezone", ["Asia/Kolkata", "America/New_York", "Europe/London"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Start", value=dtime(9, 15))
        with col2:
            end_time = st.time_input("End", value=dtime(15, 30))
    
    with st.sidebar.expander("⚙️ Parameters", expanded=False):
        sensitivity = st.slider("Sensitivity", 0.5, 3.0, 1.5, 0.1)
        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
    
    return {
        'date': date_input,
        'timezone': timezone,
        'start_time': start_time,
        'end_time': end_time,
        'sensitivity': sensitivity,
        'confidence_threshold': confidence_threshold
    }

def analyze_sectors(user_config):
    """Analyze all sectors"""
    results = []
    date_str = str(user_config['date'])
    
    for sector_name, stocks in config.SECTORS.items():
        score = scoring.calculate_sector_score(sector_name, date_str)
        
        # Apply sensitivity multiplier
        adjusted_score = score * user_config['sensitivity']
        
        confidence = min(0.95, 0.6 + abs(adjusted_score) * 0.1)
        
        results.append({
            'Sector': sector_name,
            'Score': round(adjusted_score, 2),
            'Trend': scoring.get_trend_direction(adjusted_score),
            'Strength': scoring.get_signal_strength(adjusted_score),
            'Confidence': round(confidence, 2),
            'Stocks Count': len(stocks),
            'Top Stocks': ', '.join(stocks[:3])
        })
    
    return sorted(results, key=lambda x: x['Score'], reverse=True)

def create_executive_summary(sector_results):
    """Create executive summary dashboard"""
    st.markdown("## 📊 Executive Dashboard")
    
    if not sector_results:
        st.warning("No sector data available.")
        return
    
    # Calculate metrics
    scores = [r['Score'] for r in sector_results]
    avg_score = sum(scores) / len(scores)
    
    top_bullish = max(sector_results, key=lambda x: x['Score'])
    top_bearish = min(sector_results, key=lambda x: x['Score'])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card bullish-card">
            <h3>🚀 Top Bullish</h3>
            <h2>{top_bullish['Sector']}</h2>
            <p>Score: +{top_bullish['Score']:.2f}</p>
            <small>{top_bullish['Strength']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card bearish-card">
            <h3>📉 Top Bearish</h3>
            <h2>{top_bearish['Sector']}</h2>
            <p>Score: {top_bearish['Score']:.2f}</p>
            <small>{top_bearish['Strength']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sentiment = "Bullish" if avg_score > 0 else "Bearish"
        sentiment_class = "bullish-card" if avg_score > 0 else "bearish-card"
        st.markdown(f"""
        <div class="metric-card {sentiment_class}">
            <h3>📈 Market Sentiment</h3>
            <h2>{sentiment}</h2>
            <p>Avg: {avg_score:.2f}</p>
            <small>Overall Bias</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bullish_count = len([r for r in sector_results if r['Score'] > 0])
        st.markdown(f"""
        <div class="metric-card neutral-card">
            <h3>📊 Stats</h3>
            <h2>{bullish_count}/{len(sector_results)}</h2>
            <p>Bullish Sectors</p>
            <small>Market Balance</small>
        </div>
        """, unsafe_allow_html=True)

def create_alerts(sector_results):
    """Create alert system"""
    st.markdown("### 🚨 Market Alerts")
    
    alerts = []
    
    for result in sector_results:
        abs_score = abs(result['Score'])
        
        if abs_score >= 3.0:
            alert_type = "CRITICAL"
            alert_class = "alert-danger"
        elif abs_score >= 2.0:
            alert_type = "HIGH"
            alert_class = "alert-warning"
        elif abs_score >= 1.5:
            alert_type = "MODERATE"
            alert_class = "alert-success"
        else:
            continue
        
        direction = "BULLISH" if result['Score'] > 0 else "BEARISH"
        
        alerts.append({
            'type': alert_type,
            'direction': direction,
            'sector': result['Sector'],
            'score': result['Score'],
            'confidence': result['Confidence'],
            'class': alert_class
        })
    
    if alerts:
        for alert in alerts[:5]:  # Show top 5
            st.markdown(f"""
            <div class="alert-box {alert['class']}">
                <strong>{alert['type']} {alert['direction']} SIGNAL</strong><br>
                <strong>Sector:</strong> {alert['sector']}<br>
                <strong>Score:</strong> {alert['score']:.2f} | <strong>Confidence:</strong> {alert['confidence']:.1%}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box alert-success">
            ✅ <strong>All Clear</strong><br>
            No extreme signals detected. Market conditions appear stable.
        </div>
        """, unsafe_allow_html=True)

def create_chart(sector_results):
    """Create sector performance chart"""
    if not sector_results or not PLOTLY_AVAILABLE:
        return None
    
    df = pd.DataFrame(sector_results)
    
    fig = px.bar(
        df,
        x='Score',
        y='Sector',
        orientation='h',
        color='Score',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Sector Performance Ranking",
        height=400
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig

# Main Application
def main():
    # Page config
    st.set_page_config(
        page_title="Vedic Market Analytics",
        page_icon="🪐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    apply_custom_css()
    
    # Header
    create_main_header()
    
    # Sidebar
    user_config = create_sidebar()
    
    # Main content
    try:
        # Analyze sectors
        with st.spinner("🔮 Analyzing planetary influences..."):
            sector_results = analyze_sectors(user_config)
        
        # Executive summary
        create_executive_summary(sector_results)
        
        # Alerts
        create_alerts(sector_results)
        
        # Tabs
        tabs = st.tabs([
            "📊 Sectors",
            "📈 Charts", 
            "🪐 Aspects",
            "⚙️ Settings"
        ])
        
        with tabs[0]:
            st.markdown("### 🏭 Sector Analysis")
            
            if sector_results:
                df = pd.DataFrame(sector_results)
                st.dataframe(df, use_container_width=True, height=400)
                
                # Detailed view
                st.markdown("### 🎯 Sector Details")
                selected_sector = st.selectbox("Select sector:", [r['Sector'] for r in sector_results])
                
                if selected_sector:
                    selected_data = next(r for r in sector_results if r['Sector'] == selected_sector)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score", f"{selected_data['Score']:.2f}")
                    with col2:
                        st.metric("Trend", selected_data['Trend'])
                    with col3:
                        st.metric("Confidence", f"{selected_data['Confidence']:.1%}")
                    
                    st.markdown(f"**Top Stocks:** {selected_data['Top Stocks']}")
            else:
                st.warning("No sector data available.")
        
        with tabs[1]:
            st.markdown("### 📈 Performance Charts")
            
            if PLOTLY_AVAILABLE and sector_results:
                chart = create_chart(sector_results)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Simple metrics
                scores = [r['Score'] for r in sector_results]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Score", f"{sum(scores)/len(scores):.2f}")
                with col2:
                    st.metric("Highest Score", f"{max(scores):.2f}")
                with col3:
                    st.metric("Lowest Score", f"{min(scores):.2f}")
                
                # Simple bar chart
                chart_df = pd.DataFrame(sector_results)[['Sector', 'Score']]
                st.bar_chart(chart_df.set_index('Sector'))
                
            else:
                if not PLOTLY_AVAILABLE:
                    st.info("Install plotly for advanced charts: `pip install plotly`")
                
                if sector_results:
                    chart_df = pd.DataFrame(sector_results)[['Sector', 'Score']]
                    st.bar_chart(chart_df.set_index('Sector'))
        
        with tabs[2]:
            st.markdown("### 🪐 Planetary Aspects")
            
            aspects_df = generate_sample_aspects(user_config['date'])
            
            if not aspects_df.empty:
                st.dataframe(aspects_df, use_container_width=True)
                
                # Summary
                st.markdown("### 📊 Aspects Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Aspects", len(aspects_df))
                with col2:
                    bullish_aspects = len(aspects_df[aspects_df['Score'] > 0])
                    st.metric("Bullish Aspects", bullish_aspects)
                with col3:
                    avg_score = aspects_df['Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}")
            else:
                st.info("No aspects found for selected date.")
        
        with tabs[3]:
            st.markdown("### ⚙️ System Settings")
            
            # Current config
            st.markdown("#### Current Configuration")
            config_df = pd.DataFrame([
                {"Parameter": "Date", "Value": str(user_config['date'])},
                {"Parameter": "Timezone", "Value": user_config['timezone']},
                {"Parameter": "Sensitivity", "Value": f"{user_config['sensitivity']:.1f}"},
                {"Parameter": "Confidence Threshold", "Value": f"{user_config['confidence_threshold']:.1%}"}
            ])
            st.dataframe(config_df, use_container_width=True)
            
            # System info
            st.markdown("#### System Information")
            st.markdown(f"""
            - **Plotly Status:** {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available'}
            - **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **Version:** 1.0.0 Simple
            - **Sectors Loaded:** {len(config.SECTORS)}
            """)
            
            if not PLOTLY_AVAILABLE:
                st.info("💡 Install plotly for enhanced charts: `pip install plotly`")
            
            # Export
            if st.button("📥 Export Results"):
                if sector_results:
                    export_df = pd.DataFrame(sector_results)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"vedic_analysis_{user_config['date']}.csv",
                        "text/csv"
                    )
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please try refreshing the page or adjusting your settings.")

if __name__ == "__main__":
    main()
