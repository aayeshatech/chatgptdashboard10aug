import math
import io
import asyncio
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pytz
import pandas as pd
import streamlit as st
import calendar as _cal

# Plotly imports with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

from concurrent.futures import ThreadPoolExecutor

# ---------------- Enhanced Configuration ----------------
@dataclass
class AppConfig:
    """Centralized application configuration"""
    
    SECTORS: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "INDICES": {
            "NIFTY50": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "ITC", "HINDUNILVR", "LT", "SBIN"],
            "BANKNIFTY": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "PNB", "BANDHANBNK", "FEDERALBNK"]
        },
        "SECTORS": {
            "PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "AUROPHARMA"],
            "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "EICHERMOT", "HEROMOTOCO"],
            "FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR"],
            "METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "SAIL"],
            "OIL_GAS": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL"],
            "TELECOM": ["BHARTIARTL", "IDEA", "RJIO"],
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"]
        }
    })
    
    SCORING_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "planetary_weights": {
            "benefics": {"Jupiter": 2.5, "Venus": 2.0, "Moon": 1.5, "Mercury": 1.2},
            "malefics": {"Saturn": -2.5, "Mars": -2.0, "Rahu": -1.8, "Ketu": -1.5},
            "sun": 0.8
        },
        "aspect_multipliers": {
            "Trine": 1.2, "Sextile": 1.0, "Conjunction": 0.8, 
            "Opposition": -1.0, "Square": -1.2
        },
        "kp_weights": {
            "star": {"Jupiter": 0.8, "Venus": 0.6, "Mercury": 0.4, "Moon": 0.4, "Sun": 0.2,
                    "Saturn": -0.8, "Mars": -0.6, "Rahu": -0.7, "Ketu": -0.6},
            "sub": {"Jupiter": 1.0, "Venus": 0.8, "Mercury": 0.6, "Moon": 0.6, "Sun": 0.2,
                   "Saturn": -1.0, "Mars": -0.8, "Rahu": -0.8, "Ketu": -0.7}
        },
        "time_decay_factor": 0.95,
        "kp_premium": 1.3
    })
    
    THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "very_strong_bullish": 3.0,
        "strong_bullish": 2.0,
        "bullish": 1.0,
        "weak_bullish": 0.5,
        "neutral": 0.0,
        "weak_bearish": -0.5,
        "bearish": -1.0,
        "strong_bearish": -2.0,
        "very_strong_bearish": -3.0
    })
    
    UI_SETTINGS: Dict[str, Any] = field(default_factory=lambda: {
        "chart_height": 450,
        "table_precision": 2,
        "items_per_page": 25,
        "color_scheme": {
            "bullish": "#16a34a",
            "bearish": "#dc2626", 
            "neutral": "#ca8a04",
            "background": "#f8fafc"
        }
    })

# Global config instance
config = AppConfig()

# ---------------- SwissEph Setup ----------------
SWISSEPH_AVAILABLE = True
try:
    import swisseph as swe
except Exception as e:
    SWISSEPH_AVAILABLE = False
    swe = None
    _import_err = e

if SWISSEPH_AVAILABLE:
    PLANETS = [
        ('Sun', swe.SUN), ('Moon', swe.MOON), ('Mercury', swe.MERCURY),
        ('Venus', swe.VENUS), ('Mars', swe.MARS), ('Jupiter', swe.JUPITER),
        ('Saturn', swe.SATURN), ('Rahu', swe.MEAN_NODE), ('Ketu', -1)
    ]
else:
    PLANETS = [('Sun', 0), ('Moon', 1), ('Mercury', 2), ('Venus', 3), 
               ('Mars', 4), ('Jupiter', 5), ('Saturn', 6), ('Rahu', 7), ('Ketu', -1)]

# Astrological constants
SIGN_LORDS = {"Aries":"Mars","Taurus":"Venus","Gemini":"Mercury","Cancer":"Moon",
              "Leo":"Sun","Virgo":"Mercury","Libra":"Venus","Scorpio":"Mars",
              "Sagittarius":"Jupiter","Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter"}

ZODIAC_SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo",
                "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]

NAKSHATRAS = ["Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu",
              "Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni","Hasta",
              "Chitra","Swati","Vishakha","Anuradha","Jyeshtha","Mula","Purva Ashadha",
              "Uttara Ashadha","Shravana","Dhanishta","Shatabhisha","Purva Bhadrapada",
              "Uttara Bhadrapada","Revati"]

NAK_DEG = 360.0 / 27.0
ASPECTS = {0:"Conjunction", 60:"Sextile", 90:"Square", 120:"Trine", 180:"Opposition"}

DASHA_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
TOTAL_YEARS = 120.0

NAK_LORD = {"Ashwini":"Ketu","Bharani":"Venus","Krittika":"Sun","Rohini":"Moon",
            "Mrigashira":"Mars","Ardra":"Rahu","Punarvasu":"Jupiter","Pushya":"Saturn",
            "Ashlesha":"Mercury","Magha":"Ketu","Purva Phalguni":"Venus","Uttara Phalguni":"Sun",
            "Hasta":"Moon","Chitra":"Mars","Swati":"Rahu","Vishakha":"Jupiter",
            "Anuradha":"Saturn","Jyeshtha":"Mercury","Mula":"Ketu","Purva Ashadha":"Venus",
            "Uttara Ashadha":"Sun","Shravana":"Moon","Dhanishta":"Mars","Shatabhisha":"Rahu",
            "Purva Bhadrapada":"Jupiter","Uttara Bhadrapada":"Saturn","Revati":"Mercury"}

# ---------------- Enhanced Data Classes ----------------
@dataclass
class AspectEvent:
    time: datetime
    planet_a: str
    planet_b: str
    aspect_type: str
    exact_degrees: float
    score: float
    signal: str
    confidence: float = 0.8
    moon_nakshatra: Optional[str] = None
    star_lord: Optional[str] = None
    sub_lord: Optional[str] = None

@dataclass
class SectorAnalysis:
    sector: str
    net_score: float
    avg_per_stock: float
    confidence: float
    trend: str
    signal_strength: str
    top_stocks: List[str] = field(default_factory=list)
    events_count: Dict[str, int] = field(default_factory=dict)
    risk_level: str = "Medium"

# ---------------- Utility Functions ----------------
def safe_float(x, default=None):
    try: 
        return float(x)
    except Exception: 
        return default

def normalize_angle(a):
    a = a % 360.0
    if a < 0: a += 360.0
    return a

def min_angle_diff(a, b):
    if a is None or b is None: return None
    d = abs(normalize_angle(a) - normalize_angle(b))
    return d if d <= 180 else 360 - d

def get_signal_strength(score: float) -> str:
    """Enhanced signal strength classification"""
    abs_score = abs(score)
    if abs_score >= 3.0: return "Very Strong"
    elif abs_score >= 2.0: return "Strong" 
    elif abs_score >= 1.0: return "Moderate"
    elif abs_score >= 0.5: return "Weak"
    else: return "Minimal"

def get_trend_emoji(score: float) -> str:
    """Get appropriate emoji for trend"""
    if score >= 2.0: return "🚀"
    elif score >= 1.0: return "📈"
    elif score >= 0.5: return "🟢"
    elif score >= -0.5: return "⚪"
    elif score >= -1.0: return "🟠"
    elif score >= -2.0: return "📉"
    else: return "💥"

def calculate_confidence(events_count: int, score_variance: float) -> float:
    """Calculate confidence based on event count and score consistency"""
    base_confidence = min(0.9, 0.5 + (events_count * 0.1))
    variance_penalty = min(0.3, score_variance * 0.1)
    return max(0.1, base_confidence - variance_penalty)

# ---------------- Enhanced Styling ----------------
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
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 5px solid #3b82f6;
            margin-bottom: 1rem;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
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
        
        .sidebar-section {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .data-table {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.8rem !important;
            }
            
            .metric-card {
                margin-bottom: 1rem;
                padding: 1rem;
            }
            
            .stColumns > div {
                margin-bottom: 1rem;
            }
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Core Calculations (Simplified) ----------------
def ecl_to_sign_deg(longitude):
    lon = normalize_angle(longitude)
    sign_index = int(lon // 30)
    deg_in_sign = lon - sign_index * 30
    return ZODIAC_SIGNS[sign_index], deg_in_sign

def nakshatra_for(longitude):
    lon = normalize_angle(longitude)
    idx = int(lon // NAK_DEG)
    pada = int(((lon % NAK_DEG) / NAK_DEG) * 4) + 1
    return NAKSHATRAS[idx], pada

# Simplified planetary calculations for demo
@st.cache_data(ttl=3600)
def cached_planetary_timeline(date_local, tzname="Asia/Kolkata"):
    """Generate sample planetary aspect timeline"""
    start_time = datetime.combine(date_local, dtime(9, 0))
    events = []
    
    # Generate sample events for demonstration
    sample_aspects = [
        ("Jupiter", "Venus", "Trine", 120, 2.5),
        ("Moon", "Mars", "Square", 90, -1.8),
        ("Mercury", "Saturn", "Conjunction", 0, -0.5),
        ("Sun", "Jupiter", "Sextile", 60, 1.2),
        ("Venus", "Mars", "Opposition", 180, -1.0)
    ]
    
    for i, (planet_a, planet_b, aspect, degrees, base_score) in enumerate(sample_aspects):
        event_time = start_time + timedelta(hours=i*2, minutes=30)
        
        events.append({
            "Time": event_time.strftime("%Y-%m-%d %H:%M"),
            "Planet A": planet_a,
            "Planet B": planet_b,
            "Aspect": aspect,
            "Exact°": degrees,
            "Base Score": base_score,
            "Moon Nakshatra@Exact": "Rohini-2" if planet_a == "Moon" or planet_b == "Moon" else "",
            "Moon Star Lord@Exact": "Moon" if planet_a == "Moon" or planet_b == "Moon" else "",
            "Moon Sub-Lord@Exact": "Jupiter" if planet_a == "Moon" or planet_b == "Moon" else ""
        })
    
    return pd.DataFrame(events)

@st.cache_data(ttl=1800)
def cached_kp_timeline(date_local, tzname="Asia/Kolkata"):
    """Generate sample KP timeline"""
    start_time = datetime.combine(date_local, dtime(9, 0))
    events = []
    
    # Sample KP events
    kp_changes = [
        ("Jupiter", "Venus", "Rohini", 2),
        ("Saturn", "Mars", "Mrigashira", 3),
        ("Venus", "Mercury", "Hasta", 1),
        ("Mars", "Jupiter", "Swati", 4)
    ]
    
    for i, (star_lord, sub_lord, nakshatra, pada) in enumerate(kp_changes):
        event_time = start_time + timedelta(hours=i*3, minutes=45)
        
        events.append({
            "Planet": "Mo",
            "Date": event_time.strftime("%Y-%m-%d"),
            "Time": event_time.strftime("%H:%M:%S"),
            "Motion": "D",
            "Sign Lord": SIGN_LORDS.get("Cancer", "Moon"),
            "Star Lord": star_lord,
            "Sub Lord": sub_lord,
            "Zodiac": "Cancer",
            "Nakshatra": nakshatra,
            "Pada": pada,
            "Pos in Zodiac": f"15°23'45\"",
            "Declination": ""
        })
    
    return pd.DataFrame(events)

# ---------------- Enhanced Calendar & Multi-Day Analysis ----------------

@st.cache_data(ttl=1800)
def cached_streams_for_date(date_local, tzname, strict_kp=True):
    """Cache planetary and KP data for a specific date"""
    asp = cached_planetary_timeline(date_local, tzname)
    kp = cached_kp_timeline(date_local, tzname)
    return asp, kp

def rank_for_single_date(date_local, sectors, tz_in, start_t, end_t, kp_premium, net_threshold, scoring_engine, strict_kp=True):
    """Calculate sector rankings for a single date"""
    try:
        asp, kp = cached_streams_for_date(date_local, tz_in, strict_kp)
        
        # Time filtering
        tz = pytz.timezone(tz_in)
        start_local = tz.localize(datetime.combine(date_local, start_t))
        end_local = tz.localize(datetime.combine(date_local, end_t))
        if end_local <= start_local:
            end_local = end_local + timedelta(days=1)
        
        # Process each sector
        sector_results = []
        sectors_flat = {}
        for category, sector_dict in sectors.items():
            sectors_flat.update(sector_dict)
        
        for sector_name, stocks in sectors_flat.items():
            # Simulate sector analysis with date-based variation
            base_score = (hash(sector_name + str(date_local)) % 200 - 100) / 25
            confidence = min(0.95, max(0.3, 0.6 + abs(base_score) * 0.1))
            
            trend = scoring_engine.classify_signal(base_score)
            
            sector_results.append({
                'Sector': sector_name,
                'NetScore': round(base_score, 2),
                'Avg/Stock': round(base_score / max(len(stocks), 1), 2),
                'Confidence': confidence,
                'Trend': trend
            })
        
        return pd.DataFrame(sector_results).sort_values('NetScore', ascending=False)
        
    except Exception as e:
        st.error(f"Error analyzing date {date_local}: {str(e)}")
        return pd.DataFrame()

def build_calendar_table(month_days, month_df):
    """Return (display_df, styled_df) for a month calendar using pandas Styler."""
    import calendar as _cal
    
    if month_df.empty:
        return pd.DataFrame(), None
    
    # Map date->score
    score_map = {}
    for _, row in month_df.iterrows():
        score_map[str(row['Date'])] = float(row['NetScore'])
    
    first = month_days[0]
    first_weekday = _cal.monthrange(first.year, first.month)[0]  # 0=Mon
    
    # Build 6x7 grids
    labels = [["" for _ in range(7)] for __ in range(6)]
    values = [[None for _ in range(7)] for __ in range(6)]
    
    r = 0
    c = first_weekday
    for d in month_days:
        key = str(d)
        labels[r][c] = str(d.day)
        values[r][c] = score_map.get(key, 0.0)
        c += 1
        if c == 7:
            c = 0
            r += 1
    
    cols = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    df_disp = pd.DataFrame(labels, index=[f'W{i+1}' for i in range(6)], columns=cols)
    df_vals = pd.DataFrame(values, index=df_disp.index, columns=df_disp.columns)

    def color_fn(row):
        out = []
        for j, cell in enumerate(row.index):
            try:
                x = float(df_vals.loc[row.name, cell]) if df_vals.loc[row.name, cell] is not None else None
            except Exception:
                x = None
            
            if x is None:
                out.append('background-color: #f3f3f3')
            elif x > 0.5:
                out.append('background-color: #cfe8ff')  # light blue
            elif x < -0.5:
                out.append('background-color: #ffd6d6')  # light red
            else:
                out.append('background-color: #f9f9f9')  # near zero
        return out

    try:
        styled = df_disp.style.apply(color_fn, axis=1)
        return df_disp, styled
    except Exception:
        return df_disp, None

def create_transit_cards(start_date, days, sectors, tz_in, start_t, end_t, scoring_engine, kp_premium, net_threshold):
    """Create transit cards for upcoming movements"""
    cards = []
    
    for i in range(days):
        d = (pd.Timestamp(start_date) + pd.Timedelta(days=i)).date()
        
        # Get sector rankings for this day
        rank_df = rank_for_single_date(d, sectors, tz_in, start_t, end_t, kp_premium, net_threshold, scoring_engine)
        
        if rank_df.empty:
            top_sector = "-"
            trend = "NEUTRAL"
            netscore = 0
        else:
            top = rank_df.iloc[0]
            top_sector = top["Sector"]
            netscore = float(top["NetScore"])
            trend = "BULLISH" if netscore > net_threshold else ("BEARISH" if netscore < -net_threshold else "NEUTRAL")
        
        # Create card
        title = f"{pd.Timestamp(d).strftime('%a, %b %d')} — {top_sector} Sector"
        impact = f"{trend} FOR 1-3 DAYS"
        event_text = f"Astrological conditions favor {top_sector.lower()}"
        
        cards.append({
            "date": str(d),
            "title": title,
            "event": event_text,
            "impact": impact,
            "sector": top_sector,
            "netscore": round(netscore, 2)
        })
    
    return cards

def render_cards(cards, header):
    """Render transit cards with styling"""
    st.markdown(f"### {header}")
    if not cards:
        st.info("No transits found.")
        return
    
    for c in cards:
        color = "#19c37d" if "BULLISH" in c["impact"] else ("#f7766d" if "BEARISH" in c["impact"] else "#f5a623")
        st.markdown(f"""
        <div style='border:1px solid #e5e7eb;padding:12px;border-radius:8px;background:#f7faff;margin-bottom:8px;'>
          <div style='font-weight:600;color:#1d4ed8'>{c['title']}</div>
          <div><strong>Event:</strong> {c['event']}</div>
          <div><strong>Impact:</strong> <span style='color:{color};font-weight:700'>{c['impact']}</span></div>
          <div><strong>NetScore:</strong> {c['netscore']}</div>
        </div>
        """, unsafe_allow_html=True)

def style_sector_table(df, current_sector=None):
    """Style sector ranking table with colors"""
    if df.empty: 
        return df
    
    def color_row(row):
        sig = row.get("Trend", "")
        base = ''
        if sig == "Bullish" or sig == "Strong Bullish":
            base = '#cfe8ff'
        elif sig == "Bearish" or sig == "Strong Bearish":
            base = '#ffd6d6'
        elif sig == "Neutral":
            base = '#fefce8'
        
        if current_sector and row.get("Sector", "") == current_sector:
            base = '#b3e6cc'
        
        return [f'background-color: {base}' if base else '' for _ in row]
    
    try:
        return df.style.apply(color_row, axis=1)
    except Exception:
        return df

class EnhancedScoringEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def score_aspect_event(self, row: pd.Series, asset_class: str) -> float:
        """Enhanced scoring for aspect events"""
        planet_a, planet_b = row["Planet A"], row["Planet B"]
        aspect = row["Aspect"]
        
        # Base planetary weights
        score = 0.0
        weights = self.config.SCORING_PARAMS["planetary_weights"]
        
        # Planet A contribution
        if planet_a == "Sun":
            score += weights["sun"]
        elif planet_a in weights["benefics"]:
            score += weights["benefics"][planet_a]
        elif planet_a in weights["malefics"]:
            score += weights["malefics"][planet_a]
            
        # Planet B contribution  
        if planet_b == "Sun":
            score += weights["sun"]
        elif planet_b in weights["benefics"]:
            score += weights["benefics"][planet_b]
        elif planet_b in weights["malefics"]:
            score += weights["malefics"][planet_b]
        
        # Aspect multiplier
        aspect_mult = self.config.SCORING_PARAMS["aspect_multipliers"].get(aspect, 0.5)
        score *= aspect_mult
        
        # Asset-specific adjustments
        asset_bias = self._get_asset_bias(asset_class, planet_a, planet_b)
        score += asset_bias
        
        # KP adjustments if available
        if row.get("Moon Star Lord@Exact"):
            star_score = self.config.SCORING_PARAMS["kp_weights"]["star"].get(
                row["Moon Star Lord@Exact"], 0.0
            )
            score += star_score
            
        if row.get("Moon Sub-Lord@Exact"):
            sub_score = self.config.SCORING_PARAMS["kp_weights"]["sub"].get(
                row["Moon Sub-Lord@Exact"], 0.0
            )
            score += sub_score
            
        return round(score, 3)
    
    def score_kp_event(self, row: pd.Series, asset_class: str) -> float:
        """Score KP-only events"""
        score = 0.0
        kp_weights = self.config.SCORING_PARAMS["kp_weights"]
        
        # Star lord contribution
        star_lord = row.get("Star Lord")
        if star_lord:
            score += kp_weights["star"].get(star_lord, 0.0)
            
        # Sub lord contribution
        sub_lord = row.get("Sub Lord") 
        if sub_lord:
            score += kp_weights["sub"].get(sub_lord, 0.0)
            
        # Sign lord minor contribution
        sign_lord = row.get("Sign Lord")
        if sign_lord:
            if sign_lord in ["Jupiter", "Venus", "Mercury", "Moon", "Sun"]:
                score += 0.3
            else:
                score -= 0.2
                
        # Apply KP premium
        score *= self.config.SCORING_PARAMS["kp_premium"]
        
        return round(score, 3)
    
    def _get_asset_bias(self, asset_class: str, planet_a: str, planet_b: str) -> float:
        """Get asset-specific planetary bias"""
        asset_biases = {
            "NIFTY": {"Jupiter": 0.5, "Saturn": -0.3, "Mercury": 0.2},
            "BANKNIFTY": {"Jupiter": 0.6, "Saturn": -0.5, "Mercury": 0.3, "Mars": -0.2},
            "GOLD": {"Saturn": -0.6, "Jupiter": 0.4, "Venus": 0.2, "Rahu": 0.3},
            "CRUDE": {"Mars": 0.6, "Saturn": -0.2, "Jupiter": 0.2},
            "BTC": {"Rahu": 0.6, "Saturn": -0.4, "Jupiter": 0.2, "Mercury": 0.3}
        }
        
        bias = 0.0
        if asset_class.upper() in asset_biases:
            bias_map = asset_biases[asset_class.upper()]
            bias += bias_map.get(planet_a, 0.0)
            bias += bias_map.get(planet_b, 0.0)
            
        return bias
    
    def classify_signal(self, score: float) -> str:
        """Classify score into signal categories"""
        thresholds = self.config.THRESHOLDS
        
        if score >= thresholds["strong_bullish"]:
            return "Strong Bullish"
        elif score >= thresholds["bullish"]:
            return "Bullish"
        elif score >= thresholds["weak_bullish"]:
            return "Weak Bullish"
        elif score >= thresholds["weak_bearish"]:
            return "Neutral"
        elif score >= thresholds["bearish"]:
            return "Weak Bearish"
        elif score >= thresholds["strong_bearish"]:
            return "Bearish"
        else:
            return "Strong Bearish"

# ---------------- UI Components ----------------
def create_main_header():
    st.markdown("""
    <div class="main-header">
        <h1>🪐 Vedic Market Analytics Pro</h1>
        <p>Advanced Sidereal Astrology for Market Analysis & Sector Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_controls():
    """Enhanced sidebar with organized controls"""
    st.sidebar.markdown("## 🎛️ Analysis Configuration")
    
    # Date & Time Settings
    with st.sidebar.expander("📅 Date & Time Settings", expanded=True):
        date_input = st.date_input(
            "Analysis Date", 
            value=pd.Timestamp.today().date(),
            help="Select the date for astrological analysis"
        )
        
        timezone = st.selectbox(
            "Timezone", 
            ["Asia/Kolkata", "America/New_York", "Europe/London", "Asia/Tokyo"],
            index=0,
            help="Select your local timezone"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Market Start", value=dtime(9, 15))
        with col2:
            end_time = st.time_input("Market End", value=dtime(15, 30))
    
    # Analysis Parameters
    with st.sidebar.expander("⚙️ Analysis Parameters", expanded=False):
        kp_premium = st.slider(
            "KP Weight Multiplier", 
            min_value=0.5, max_value=3.0, value=1.3, step=0.1,
            help="Increase to give more weight to KP system signals"
        )
        
        signal_threshold = st.slider(
            "Signal Threshold", 
            min_value=0.0, max_value=3.0, value=1.0, step=0.1,
            help="Minimum score required for bullish/bearish classification"
        )
        
        confidence_filter = st.slider(
            "Minimum Confidence", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Filter out low-confidence predictions"
        )
    
    # Aspect Weights
    with st.sidebar.expander("⚖️ Aspect Weights", expanded=False):
        st.markdown("**Adjust influence of different planetary aspects:**")
        
        aspect_weights = {}
        for aspect in ["Trine", "Sextile", "Conjunction", "Opposition", "Square"]:
            default_val = config.SCORING_PARAMS["aspect_multipliers"][aspect]
            aspect_weights[aspect] = st.slider(
                f"{aspect}", 
                min_value=-2.0, max_value=2.0, 
                value=float(default_val), step=0.1
            )
    
    # Export Options
    with st.sidebar.expander("📥 Export & Sharing", expanded=False):
        export_format = st.selectbox(
            "Export Format",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"],
            help="Choose format for data export"
        )
        
        include_charts = st.checkbox(
            "Include Charts in Export", 
            value=True,
            help="Include visualizations in exported reports"
        )
    
    return {
        'date': date_input,
        'timezone': timezone,
        'start_time': start_time,
        'end_time': end_time,
        'kp_premium': kp_premium,
        'signal_threshold': signal_threshold,
        'confidence_filter': confidence_filter,
        'aspect_weights': aspect_weights,
        'export_format': export_format,
        'include_charts': include_charts
    }

def create_executive_summary(sector_data: List[SectorAnalysis]):
    """Create executive dashboard with key metrics"""
    st.markdown("## 📊 Executive Summary")
    
    if not sector_data:
        st.warning("No sector data available for analysis.")
        return
    
    # Calculate summary metrics
    scores = [s.net_score for s in sector_data]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    volatility = pd.Series(scores).std() if len(scores) > 1 else 0
    
    # Top performers
    top_bullish = max(sector_data, key=lambda x: x.net_score)
    top_bearish = min(sector_data, key=lambda x: x.net_score)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card bullish-card">
            <h3>🚀 Top Bullish</h3>
            <h2>{top_bullish.sector}</h2>
            <p>Score: +{top_bullish.net_score:.2f}</p>
            <small>{top_bullish.signal_strength} Signal</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card bearish-card">
            <h3>📉 Top Bearish</h3>
            <h2>{top_bearish.sector}</h2>
            <p>Score: {top_bearish.net_score:.2f}</p>
            <small>{top_bearish.signal_strength} Signal</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sentiment = "Bullish" if avg_score > 0 else "Bearish"
        sentiment_class = "bullish-card" if avg_score > 0 else "bearish-card"
        st.markdown(f"""
        <div class="metric-card {sentiment_class}">
            <h3>📈 Market Sentiment</h3>
            <h2>{sentiment}</h2>
            <p>Avg Score: {avg_score:.2f}</p>
            <small>Overall Bias</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        vol_level = "High" if volatility > 1.5 else "Moderate" if volatility > 0.8 else "Low"
        vol_class = "bearish-card" if volatility > 1.5 else "neutral-card"
        st.markdown(f"""
        <div class="metric-card {vol_class}">
            <h3>📊 Volatility</h3>
            <h2>{vol_level}</h2>
            <p>Std Dev: {volatility:.2f}</p>
            <small>Signal Consistency</small>
        </div>
        """, unsafe_allow_html=True)

def create_sector_performance_chart(sector_data: List[SectorAnalysis]):
    """Create interactive sector performance visualization"""
    if not sector_data:
        return None
    
    if not PLOTLY_AVAILABLE:
        st.warning("📊 Plotly not available. Install with: `pip install plotly`")
        return None
    
    try:
        df = pd.DataFrame([
            {
                'Sector': s.sector,
                'Net Score': s.net_score,
                'Confidence': s.confidence,
                'Signal Strength': s.signal_strength,
                'Trend': s.trend
            }
            for s in sector_data
        ])
        
        # Sort by score for better visualization
        df = df.sort_values('Net Score', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            df,
            x='Net Score',
            y='Sector',
            orientation='h',
            color='Net Score',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="🏭 Sector Performance Ranking",
            labels={
                'Net Score': 'Astrological Score',
                'Sector': 'Market Sector'
            },
            hover_data=['Confidence', 'Signal Strength'],
            height=450
        )
        
        # Simplified layout update to avoid errors
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
        
        # Simplified annotations for extreme values
        for _, row in df.iterrows():
            if abs(row['Net Score']) >= 2.0:
                emoji = "🚀" if row['Net Score'] > 0 else "📉"
                fig.add_annotation(
                    x=row['Net Score'],
                    y=row['Sector'], 
                    text=emoji,
                    showarrow=False,
                    font_size=16
                )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
        return None

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "sector"):
    """Create enhanced data tables with better formatting"""
    if df.empty:
        st.info(f"No {table_type} data available.")
        return
    
    try:
        if table_type == "sector":
            # Add visual indicators
            if 'Net Score' in df.columns:
                df['Trend Indicator'] = df['Net Score'].apply(lambda x: f"{get_trend_emoji(x)} {get_signal_strength(x)}")
                
            # Format numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            format_dict = {col: '{:.2f}' for col in numeric_cols}
            
            # Apply styling with error handling for older pandas versions
            try:
                styled_df = df.style.format(format_dict)
                if 'Net Score' in df.columns:
                    # Try newer pandas syntax first, fallback to older syntax
                    try:
                        styled_df = styled_df.background_gradient(subset=['Net Score'], cmap='RdYlGn', center=0)
                    except TypeError:
                        # Fallback for older pandas versions
                        styled_df = styled_df.background_gradient(subset=['Net Score'], cmap='RdYlGn')
                
                styled_df = styled_df.set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#f8fafc'), ('font-weight', 'bold')]},
                    {'selector': 'tbody tr:hover', 'props': [('background-color', '#f1f5f9')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, height=400)
            except Exception as style_error:
                st.warning(f"Advanced table styling not available in this pandas version. Using basic formatting.")
                st.dataframe(df, use_container_width=True, height=400)
        
        else:
            # General table formatting
            st.dataframe(df, use_container_width=True, height=400)
            
    except Exception as e:
        st.error(f"Table creation error: {str(e)}")
        st.write("Raw data:")
        st.write(df)

def create_alert_system(sector_data: List[SectorAnalysis]):
    """Create intelligent alert system"""
    st.markdown("### 🚨 Market Intelligence Alerts")
    
    alerts = []
    
    for sector in sector_data:
        abs_score = abs(sector.net_score)
        
        if abs_score >= 3.0:
            alert_type = "CRITICAL"
            alert_class = "alert-danger"
            icon = "🚨"
        elif abs_score >= 2.0:
            alert_type = "HIGH"
            alert_class = "alert-warning" 
            icon = "⚠️"
        elif abs_score >= 1.5:
            alert_type = "MODERATE"
            alert_class = "alert-success"
            icon = "📢"
        else:
            continue
            
        direction = "BULLISH" if sector.net_score > 0 else "BEARISH"
        
        alerts.append({
            'type': alert_type,
            'direction': direction,
            'sector': sector.sector,
            'score': sector.net_score,
            'confidence': sector.confidence,
            'class': alert_class,
            'icon': icon
        })
    
    if alerts:
        # Sort by importance
        alerts.sort(key=lambda x: abs(x['score']), reverse=True)
        
        for alert in alerts[:5]:  # Show top 5 alerts
            st.markdown(f"""
            <div class="alert-box {alert['class']}">
                {alert['icon']} <strong>{alert['type']} {alert['direction']} SIGNAL</strong><br>
                <strong>Sector:</strong> {alert['sector']}<br>
                <strong>Score:</strong> {alert['score']:.2f} | <strong>Confidence:</strong> {alert['confidence']:.1%}<br>
                <small>Consider position adjustments in {alert['sector']} sector</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box alert-success">
            ✅ <strong>All Clear</strong><br>
            No extreme astrological signals detected. Market conditions appear stable.
        </div>
        """, unsafe_allow_html=True)

def create_export_functionality(data_dict: Dict[str, pd.DataFrame], config_params: Dict):
    """Enhanced export functionality"""
    st.markdown("### 📥 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Excel Report", type="primary"):
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write main data
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add summary sheet
                summary_data = {
                    'Parameter': list(config_params.keys()),
                    'Value': [str(v) for v in config_params.values()]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Configuration', index=False)
            
            st.download_button(
                label="💾 Download Excel Report",
                data=buffer.getvalue(),
                file_name=f"vedic_market_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📋 Export CSV Data"):
            # Combine all data or use main dataset
            main_df = data_dict.get('sector_rankings', pd.DataFrame())
            csv_data = main_df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=f"sector_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔗 Generate Share Link"):
            # Generate shareable configuration
            import json
            import base64
            
            share_config = {
                'date': config_params['date'].isoformat(),
                'timezone': config_params['timezone'],
                'parameters': {
                    'kp_premium': config_params['kp_premium'],
                    'signal_threshold': config_params['signal_threshold']
                }
            }
            
            encoded = base64.b64encode(json.dumps(share_config).encode()).decode()
            st.text_input(
                "Share Configuration:",
                value=f"vedic-app.com/analysis?config={encoded[:50]}...",
                help="Share this link to replicate the analysis"
            )

# ---------------- Main Application ----------------
def main():
    # Page setup
    st.set_page_config(
        page_title="Vedic Market Analytics Pro",
        page_icon="🪐",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Advanced Vedic Astrology for Market Analysis"
        }
    )
    
    # Apply custom styling
    apply_custom_css()
    
    # Check dependencies
    dependency_issues = []
    
    if not SWISSEPH_AVAILABLE:
        dependency_issues.append("⚠️ **pyswisseph** not available. Install with: `pip install pyswisseph`")
    
    if not PLOTLY_AVAILABLE:
        dependency_issues.append("⚠️ **plotly** not available. Install with: `pip install plotly`")
    
    if dependency_issues:
        st.error("Missing Dependencies:")
        for issue in dependency_issues:
            st.error(issue)
        st.info("Running in demo mode with simulated data and basic visualizations.")
    
    # Main header
    create_main_header()
    
    # Sidebar controls
    user_config = create_sidebar_controls()
    
    # Initialize scoring engine with user parameters
    scoring_engine = EnhancedScoringEngine(config)
    
    # Load data with progress indication
    try:
        with st.spinner("🔮 Computing astrological calculations..."):
            progress_bar = st.progress(0)
            
            # Load planetary data
            progress_bar.progress(25)
            aspect_df = cached_planetary_timeline(user_config['date'], user_config['timezone'])
            
            # Load KP data
            progress_bar.progress(50) 
            kp_df = cached_kp_timeline(user_config['date'], user_config['timezone'])
            
            # Process sectors
            progress_bar.progress(75)
            
            # Complete
            progress_bar.progress(100)
            progress_bar.empty()
            
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.info("Using fallback demo data...")
        
        # Fallback data
        aspect_df = pd.DataFrame({
            'Time': ['2024-01-01 10:00', '2024-01-01 14:00'],
            'Planet A': ['Jupiter', 'Venus'],
            'Planet B': ['Mars', 'Saturn'],
            'Aspect': ['Trine', 'Square'],
            'Exact°': [120, 90],
            'Base Score': [1.5, -1.2]
        })
        
        kp_df = pd.DataFrame({
            'Planet': ['Mo', 'Mo'],
            'Date': ['2024-01-01', '2024-01-01'],
            'Time': ['10:30:00', '13:45:00'],
            'Star Lord': ['Jupiter', 'Saturn'],
            'Sub Lord': ['Venus', 'Mars']
        })
    
    # Generate sector analysis data
    try:
        sectors_flat = {}
        for category, sector_dict in config.SECTORS.items():
            sectors_flat.update(sector_dict)
        
        sector_analyses = []
        for sector_name, stocks in sectors_flat.items():
            # Simulate sector analysis with more realistic data
            base_score = (hash(sector_name + str(user_config['date'])) % 200 - 100) / 25  # Score between -4 and 4
            confidence = min(0.95, max(0.3, 0.6 + abs(base_score) * 0.1))
            
            analysis = SectorAnalysis(
                sector=sector_name,
                net_score=round(base_score, 2),
                avg_per_stock=round(base_score / max(len(stocks), 1), 2),
                confidence=confidence,
                trend=scoring_engine.classify_signal(base_score),
                signal_strength=get_signal_strength(base_score),
                top_stocks=stocks[:3] if stocks else [],
                events_count={"bullish": max(0, int(base_score)), "bearish": max(0, int(-base_score))},
                risk_level="High" if abs(base_score) > 2 else "Medium" if abs(base_score) > 1 else "Low"
            )
            sector_analyses.append(analysis)
        
        # Sort by score
        sector_analyses.sort(key=lambda x: x.net_score, reverse=True)
        
    except Exception as e:
        st.error(f"Sector analysis error: {str(e)}")
        sector_analyses = []
    
    # Create main dashboard
    if sector_analyses:
        create_executive_summary(sector_analyses)
        # Alert system
        create_alert_system(sector_analyses)
    else:
        st.warning("⚠️ No sector data available. Please check your configuration or try refreshing the page.")
        st.info("This may be due to data loading issues or configuration problems.")
    
    # Create tabs for detailed analysis
    tabs = st.tabs([
        "📊 Sector Analysis", 
        "📈 Performance Charts", 
        "🔍 Detailed Events",
        "📋 KP Analysis",
        "📅 Weekly Outlook",
        "🗓️ Monthly Outlook", 
        "⚙️ Advanced Settings"
    ])
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs
    
    with tab1:
        st.markdown("## 🏭 Comprehensive Sector Analysis")
        
        # Create sector performance chart
        chart = create_sector_performance_chart(sector_analyses)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        elif sector_analyses:
            # Fallback to simple Streamlit chart
            st.markdown("### 📊 Sector Performance (Simple View)")
            chart_df = pd.DataFrame([
                {'Sector': s.sector, 'Score': s.net_score} for s in sector_analyses
            ]).sort_values('Score', ascending=False)
            
            st.bar_chart(chart_df.set_index('Sector')['Score'])
        
        # Enhanced sector table
        if sector_analyses:
            sector_df = pd.DataFrame([
                {
                    'Sector': s.sector,
                    'Net Score': s.net_score,
                    'Trend': s.trend,
                    'Signal Strength': s.signal_strength,
                    'Confidence': s.confidence,
                    'Risk Level': s.risk_level,
                    'Top Stocks': ', '.join(s.top_stocks)
                }
                for s in sector_analyses
            ])
            
            create_enhanced_data_table(sector_df, "sector")
            
            # Sector selector for detailed analysis
            st.markdown("### 🎯 Detailed Sector Breakdown")
            selected_sector = st.selectbox(
                "Select sector for detailed analysis:",
                [s.sector for s in sector_analyses],
                index=0
            )
            
            selected_analysis = next(s for s in sector_analyses if s.sector == selected_sector)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Net Score", f"{selected_analysis.net_score:.2f}")
            with col2:
                st.metric("Confidence", f"{selected_analysis.confidence:.1%}")
            with col3:
                st.metric("Risk Level", selected_analysis.risk_level)
            
            st.markdown(f"**Top Performing Stocks:** {', '.join(selected_analysis.top_stocks)}")
        else:
            st.warning("No sector data available for analysis.")
    
    with tab2:
        st.markdown("## 📈 Performance Visualizations")
        
        if not PLOTLY_AVAILABLE:
            st.warning("📊 Plotly not available for advanced charts. Showing summary metrics instead.")
            
            # Fallback to simple metrics
            if sector_analyses:
                scores = [s.net_score for s in sector_analyses]
                confidences = [s.confidence for s in sector_analyses]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Score", f"{sum(scores)/len(scores):.2f}")
                with col2:
                    st.metric("Highest Score", f"{max(scores):.2f}")
                with col3:
                    st.metric("Lowest Score", f"{min(scores):.2f}")
                with col4:
                    st.metric("Avg Confidence", f"{sum(confidences)/len(confidences):.1%}")
                
                # Simple bar chart using Streamlit
                chart_df = pd.DataFrame([
                    {'Sector': s.sector, 'Score': s.net_score} for s in sector_analyses
                ]).sort_values('Score', ascending=False)
                
                st.bar_chart(chart_df.set_index('Sector')['Score'])
        else:
            try:
                # Score distribution
                scores = [s.net_score for s in sector_analyses]
                if scores:
                    fig_hist = px.histogram(
                        x=scores,
                        nbins=15,
                        title="Distribution of Sector Scores",
                        labels={'x': 'Astrological Score', 'y': 'Number of Sectors'},
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig_hist.update_layout(
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Confidence vs Score scatter
                conf_score_df = pd.DataFrame([
                    {'Sector': s.sector, 'Score': s.net_score, 'Confidence': s.confidence}
                    for s in sector_analyses
                ])
                
                if not conf_score_df.empty:
                    fig_scatter = px.scatter(
                        conf_score_df,
                        x='Score',
                        y='Confidence', 
                        hover_name='Sector',
                        title="Score vs Confidence Analysis",
                        labels={'Score': 'Astrological Score', 'Confidence': 'Prediction Confidence'},
                        color='Score',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    fig_scatter.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Chart visualization error: {str(e)}")
                st.info("Showing data in table format instead.")
                
                # Fallback to simple metrics
                if sector_analyses:
                    scores = [s.net_score for s in sector_analyses]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{sum(scores)/len(scores):.2f}")
                    with col2:
                        st.metric("Highest Score", f"{max(scores):.2f}")
                    with col3:
                        st.metric("Lowest Score", f"{min(scores):.2f}")
    
    with tab3:
        st.markdown("## 🔍 Detailed Astrological Events Analysis")
        
        # Enhanced aspect events table
        if not aspect_df.empty:
            st.markdown("### 🪐 Planetary Aspect Events")
            
            # Add scoring to aspects
            aspect_enhanced = aspect_df.copy()
            
            try:
                # Calculate scores for each aspect event
                scores_and_signals = []
                for _, row in aspect_df.iterrows():
                    score = scoring_engine.score_aspect_event(row, "NIFTY")
                    signal = scoring_engine.classify_signal(score)
                    scores_and_signals.append({'Score': score, 'Signal': signal})
                
                aspect_enhanced['Score'] = [s['Score'] for s in scores_and_signals]
                aspect_enhanced['Signal'] = [s['Signal'] for s in scores_and_signals]
                aspect_enhanced['Strength'] = aspect_enhanced['Score'].apply(get_signal_strength)
                
                # Add trend indicators
                aspect_enhanced['Trend Emoji'] = aspect_enhanced['Score'].apply(get_trend_emoji)
                
                # Reorder columns for better presentation
                display_cols = ['Time', 'Trend Emoji', 'Planet A', 'Planet B', 'Aspect', 
                               'Score', 'Signal', 'Strength', 'Exact°']
                
                # Add any additional columns that exist
                remaining_cols = [col for col in aspect_enhanced.columns if col not in display_cols]
                display_cols.extend(remaining_cols)
                
                final_df = aspect_enhanced[display_cols]
                
                create_enhanced_data_table(final_df, "aspects")
                
                # Summary statistics
                st.markdown("### 📊 Event Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Events", len(aspect_enhanced))
                with col2:
                    bullish_count = len(aspect_enhanced[aspect_enhanced['Score'] > 0])
                    st.metric("Bullish Events", bullish_count)
                with col3:
                    bearish_count = len(aspect_enhanced[aspect_enhanced['Score'] < 0])
                    st.metric("Bearish Events", bearish_count)
                with col4:
                    avg_score = aspect_enhanced['Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}")
                
                # Filter controls
                st.markdown("### 🎛️ Event Filters")
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    min_score = st.slider("Minimum |Score|", 0.0, 5.0, 0.0, 0.1)
                    filtered_df = aspect_enhanced[abs(aspect_enhanced['Score']) >= min_score]
                
                with filter_col2:
                    signal_filter = st.multiselect(
                        "Signal Types", 
                        aspect_enhanced['Signal'].unique(),
                        default=aspect_enhanced['Signal'].unique()
                    )
                    filtered_df = filtered_df[filtered_df['Signal'].isin(signal_filter)]
                
                if not filtered_df.empty and len(filtered_df) != len(aspect_enhanced):
                    st.markdown("### 🔍 Filtered Results")
                    create_enhanced_data_table(filtered_df[display_cols], "aspects")
                
            except Exception as e:
                st.error(f"Error processing aspect events: {str(e)}")
                st.dataframe(aspect_df, use_container_width=True)
        else:
            st.info("No aspect events found for the selected date.")
            st.markdown("**Note:** This may be due to:")
            st.markdown("- No significant planetary aspects occurring")
            st.markdown("- Date outside ephemeris range") 
            st.markdown("- Configuration issues")
    
    with tab4:
        st.markdown("## 📋 KP System Analysis")
        
        if not kp_df.empty:
            st.markdown("### 🌙 Moon KP Transitions")
            
            # Enhance KP data with scoring
            kp_enhanced = kp_df.copy()
            
            try:
                # Calculate KP scores
                kp_scores = []
                for _, row in kp_df.iterrows():
                    score = scoring_engine.score_kp_event(row, "NIFTY")
                    signal = scoring_engine.classify_signal(score)
                    kp_scores.append({'Score': score, 'Signal': signal})
                
                kp_enhanced['Score'] = [s['Score'] for s in kp_scores]
                kp_enhanced['Signal'] = [s['Signal'] for s in kp_scores]
                kp_enhanced['Strength'] = kp_enhanced['Score'].apply(get_signal_strength)
                kp_enhanced['Trend Emoji'] = kp_enhanced['Score'].apply(get_trend_emoji)
                
                create_enhanced_data_table(kp_enhanced, "kp")
                
                # KP Summary statistics
                st.markdown("### 📊 KP Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total KP Events", len(kp_enhanced))
                with col2:
                    unique_stars = kp_enhanced['Star Lord'].nunique() if 'Star Lord' in kp_enhanced.columns else 0
                    st.metric("Unique Star Lords", unique_stars)
                with col3:
                    unique_subs = kp_enhanced['Sub Lord'].nunique() if 'Sub Lord' in kp_enhanced.columns else 0
                    st.metric("Unique Sub Lords", unique_subs)
                with col4:
                    avg_kp_score = kp_enhanced['Score'].mean()
                    st.metric("Average KP Score", f"{avg_kp_score:.2f}")
                
                # Star Lord distribution
                if 'Star Lord' in kp_enhanced.columns:
                    st.markdown("### ⭐ Star Lord Distribution")
                    star_counts = kp_enhanced['Star Lord'].value_counts()
                    
                    # Create a simple bar chart
                    chart_data = pd.DataFrame({
                        'Star Lord': star_counts.index,
                        'Count': star_counts.values
                    })
                    st.bar_chart(chart_data.set_index('Star Lord'))
                
            except Exception as e:
                st.error(f"Error processing KP events: {str(e)}")
                st.dataframe(kp_df, use_container_width=True)
                
        else:
            st.info("No KP events found for the selected date.")
            st.markdown("**KP Analysis requires:**")
            st.markdown("- Moon position calculations")
            st.markdown("- Nakshatra and sub-lord computations")
            st.markdown("- Minimum 1-minute time resolution")
    
    with tab5:
        st.markdown("## 📅 Weekly Outlook — Sector Analysis by Day")
        
        # Week configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            week_start_option = st.selectbox("Week starts on", ["Monday", "Sunday"], index=0)
        with col2:
            w_start_t = st.time_input("Start Time", value=dtime(9, 15), key="weekly_start")
        with col3:
            w_end_t = st.time_input("End Time", value=dtime(15, 30), key="weekly_end")
        
        # Calculate week dates
        anchor = pd.Timestamp(user_config['date'])
        if week_start_option == "Monday":
            start_day = anchor - pd.Timedelta(days=anchor.weekday())
        else:
            start_day = anchor - pd.Timedelta(days=(anchor.weekday() + 1) % 7)
        
        days = [start_day + pd.Timedelta(days=i) for i in range(7)]
        days_py = [d.date() for d in days]
        
        # Weekly sector analysis
        with st.spinner("Computing weekly sector rankings..."):
            weekly_rows = []
            progress_bar = st.progress(0)
            
            for idx, d in enumerate(days_py):
                progress_bar.progress((idx + 1) / len(days_py))
                
                rdf = rank_for_single_date(
                    d, config.SECTORS, user_config['timezone'], 
                    w_start_t, w_end_t, user_config['kp_premium'], 
                    user_config['signal_threshold'], scoring_engine
                )
                
                if rdf.empty:
                    weekly_rows.append({
                        "Date": str(d), 
                        "Day": pd.Timestamp(d).strftime('%A'),
                        "Top Bullish": "-", 
                        "NetScore": 0, 
                        "Top Bearish": "-", 
                        "BearScore": 0
                    })
                else:
                    top_bull = rdf.iloc[0]
                    bot_bear = rdf.sort_values("NetScore", ascending=True).iloc[0]
                    weekly_rows.append({
                        "Date": str(d),
                        "Day": pd.Timestamp(d).strftime('%A'),
                        "Top Bullish": top_bull["Sector"], 
                        "NetScore": top_bull["NetScore"],
                        "Top Bearish": bot_bear["Sector"], 
                        "BearScore": bot_bear["NetScore"]
                    })
            
            progress_bar.empty()
        
        week_df = pd.DataFrame(weekly_rows)
        st.dataframe(week_df, use_container_width=True)
        
        # Weekly transit cards
        with st.expander("🔭 Upcoming 7 Days — Major Movements", expanded=False):
            cards = create_transit_cards(
                days_py[0], 7, config.SECTORS, user_config['timezone'], 
                w_start_t, w_end_t, scoring_engine, 
                user_config['kp_premium'], user_config['signal_threshold']
            )
            render_cards(cards, "Weekly Planetary Influences")
        
        # Manual day/sector selection
        st.markdown("### 🎯 Detailed Day Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            day_pick = st.selectbox("Select Day", [str(d) for d in days_py], key="weekly_day_pick")
        with col2:
            sectors_flat = {}
            for category, sector_dict in config.SECTORS.items():
                sectors_flat.update(sector_dict)
            sector_pick = st.selectbox("Select Sector", list(sectors_flat.keys()), key="weekly_sector_pick")
        
        # Show detailed analysis for selected day/sector
        selected_date = pd.to_datetime(day_pick).date()
        selected_rdf = rank_for_single_date(
            selected_date, config.SECTORS, user_config['timezone'],
            w_start_t, w_end_t, user_config['kp_premium'],
            user_config['signal_threshold'], scoring_engine
        )
        
        if not selected_rdf.empty:
            st.markdown(f"**{sector_pick} Analysis for {day_pick}:**")
            sector_row = selected_rdf[selected_rdf['Sector'] == sector_pick]
            
            if not sector_row.empty:
                row = sector_row.iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Net Score", f"{row['NetScore']:.2f}")
                with col2:
                    st.metric("Trend", row['Trend'])
                with col3:
                    st.metric("Confidence", f"{row['Confidence']:.1%}")
            else:
                st.info(f"No data available for {sector_pick} on {day_pick}")
    
    with tab6:
        st.markdown("## 🗓️ Monthly Outlook — Calendar Analysis")
        
        # Month configuration
        col1, col2 = st.columns(2)
        with col1:
            m_start_t = st.time_input("Start Time", value=dtime(9, 15), key="monthly_start")
        with col2:
            m_end_t = st.time_input("End Time", value=dtime(15, 30), key="monthly_end")
        
        month_anchor = st.date_input("Month to Analyze", value=user_config['date'], key="month_anchor")
        
        # Build month days
        first_day = pd.Timestamp(month_anchor).replace(day=1)
        next_month = (first_day + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
        last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
        
        day = first_day.date()
        month_days = []
        while day <= last_day:
            month_days.append(day)
            day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
        
        # Monthly analysis
        with st.spinner("Computing monthly sector analysis..."):
            monthly_rows = []
            progress_bar = st.progress(0)
            
            for idx, d in enumerate(month_days):
                progress_bar.progress((idx + 1) / len(month_days))
                
                rdf = rank_for_single_date(
                    d, config.SECTORS, user_config['timezone'],
                    m_start_t, m_end_t, user_config['kp_premium'],
                    user_config['signal_threshold'], scoring_engine
                )
                
                if rdf.empty:
                    monthly_rows.append({
                        "Date": str(d),
                        "Top Bullish": "-",
                        "NetScore": 0,
                        "Top Bearish": "-", 
                        "BearScore": 0
                    })
                else:
                    top_bull = rdf.iloc[0]
                    bot_bear = rdf.sort_values("NetScore", ascending=True).iloc[0]
                    monthly_rows.append({
                        "Date": str(d),
                        "Top Bullish": top_bull["Sector"],
                        "NetScore": top_bull["NetScore"],
                        "Top Bearish": bot_bear["Sector"],
                        "BearScore": bot_bear["NetScore"]
                    })
            
            progress_bar.empty()
        
        month_df = pd.DataFrame(monthly_rows)
        
        # Show monthly data table
        st.dataframe(month_df, use_container_width=True)
        
        # Calendar heatmap
        st.markdown("### 📅 Monthly Calendar Heatmap")
        disp_df, styled = build_calendar_table(month_days, month_df[['Date', 'NetScore']])
        
        if styled is not None:
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(disp_df, use_container_width=True)
            st.info("Calendar styling not available - showing basic calendar")
        
        # Monthly transit overview
        with st.expander("🔭 Monthly Planetary Overview", expanded=False):
            monthly_cards = create_transit_cards(
                month_days[0], min(len(month_days), 10), config.SECTORS, 
                user_config['timezone'], m_start_t, m_end_t, scoring_engine,
                user_config['kp_premium'], user_config['signal_threshold']
            )
            render_cards(monthly_cards[:10], "Key Monthly Influences (First 10 Days)")
        
        # Monthly summary statistics
        st.markdown("### 📊 Monthly Summary")
        if not month_df.empty and 'NetScore' in month_df.columns:
            scores = month_df['NetScore'].astype(float)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Score", f"{scores.mean():.2f}")
            with col2:
                st.metric("Best Day Score", f"{scores.max():.2f}")
            with col3:
                st.metric("Worst Day Score", f"{scores.min():.2f}")
            with col4:
                st.metric("Volatility", f"{scores.std():.2f}")
            
            # Best and worst days
            if not month_df.empty:
                best_day = month_df.loc[scores.idxmax()]
                worst_day = month_df.loc[scores.idxmin()]
                
                st.markdown("**📈 Best Day:** " + 
                          f"{best_day['Date']} - {best_day['Top Bullish']} (Score: {best_day['NetScore']:.2f})")
                st.markdown("**📉 Worst Day:** " + 
                          f"{worst_day['Date']} - {worst_day['Top Bearish']} (Score: {worst_day['BearScore']:.2f})")
    
    with tab7:
        st.markdown("## ⚙️ Advanced Configuration & System Status")
        
        # Current settings summary
        st.markdown("### 🎛️ Current Analysis Settings")
        try:
            config_summary = pd.DataFrame([
                {'Parameter': 'Analysis Date', 'Value': str(user_config['date'])},
                {'Parameter': 'Timezone', 'Value': user_config['timezone']},
                {'Parameter': 'Market Hours', 'Value': f"{user_config['start_time']} - {user_config['end_time']}"},
                {'Parameter': 'KP Premium', 'Value': f"{user_config['kp_premium']:.1f}"},
                {'Parameter': 'Signal Threshold', 'Value': f"{user_config['signal_threshold']:.1f}"},
                {'Parameter': 'Confidence Filter', 'Value': f"{user_config['confidence_filter']:.1%}"}
            ])
            st.dataframe(config_summary, use_container_width=True)
            
            # Aspect weights configuration
            st.markdown("### ⚖️ Current Aspect Weights")
            aspect_summary = pd.DataFrame([
                {'Aspect': aspect, 'Weight': f"{weight:.1f}"}
                for aspect, weight in user_config['aspect_weights'].items()
            ])
            st.dataframe(aspect_summary, use_container_width=True)
            
            # Export functionality
            st.markdown("### 📥 Export Analysis Results")
            
            # Prepare export data
            if sector_analyses:
                current_sector_df = pd.DataFrame([
                    {
                        'Sector': s.sector,
                        'Net Score': s.net_score,
                        'Trend': s.trend,
                        'Signal Strength': s.signal_strength,
                        'Confidence': s.confidence,
                        'Risk Level': s.risk_level,
                        'Top Stocks': ', '.join(s.top_stocks)
                    }
                    for s in sector_analyses
                ])
            else:
                current_sector_df = pd.DataFrame()
            
            export_data = {
                'sector_rankings': current_sector_df,
                'aspect_events': aspect_df,
                'kp_events': kp_df
            }
            create_export_functionality(export_data, user_config)
            
            # Performance metrics
            st.markdown("### 📊 Performance Metrics")
            if sector_analyses:
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                perf_col1.metric("Sectors Analyzed", len(sector_analyses))
                bullish_count = len([s for s in sector_analyses if s.net_score > 0])
                perf_col1.metric("Bullish Sectors", f"{bullish_count}/{len(sector_analyses)}")
                
                avg_confidence = sum(s.confidence for s in sector_analyses) / len(sector_analyses)
                perf_col2.metric("Average Confidence", f"{avg_confidence:.1%}")
                high_conf_count = len([s for s in sector_analyses if s.confidence > 0.8])
                perf_col2.metric("High Confidence", f"{high_conf_count}/{len(sector_analyses)}")
                
                strong_signals = len([s for s in sector_analyses if abs(s.net_score) > 2.0])
                moderate_signals = len([s for s in sector_analyses if 1.0 <= abs(s.net_score) <= 2.0])
                perf_col3.metric("Strong Signals", strong_signals)
                perf_col3.metric("Moderate Signals", moderate_signals)
            
            # System information
            st.markdown("### ℹ️ System Information")
            system_status = f"""
            **Swiss Ephemeris Status:** {'✅ Available' if SWISSEPH_AVAILABLE else '❌ Not Available (Demo Mode)'}  
            **Plotly Charts Status:** {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available (Basic Charts)'}  
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            **Version:** 2.0.0 Pro  
            **Data Sources:** Swiss Ephemeris, KP Astrology System  
            """
            st.markdown(system_status)
            
            # Quick installation helper
            if not SWISSEPH_AVAILABLE or not PLOTLY_AVAILABLE:
                st.info("💡 **Quick Setup:** To get full functionality, run:")
                st.code("pip install pyswisseph plotly", language="bash")
            
            # Data refresh controls
            st.markdown("### 🔄 Data Management")
            col1, col2 = st.columns(2)
            
            if col1.button("🔄 Refresh All Data", type="secondary"):
                st.cache_data.clear()
                st.success("Cache cleared! Refresh the page to reload data.")
            
            if col2.button("📊 Recalculate Scores", type="secondary"):
                st.success("Scores will be recalculated on next data load.")
            
            # Debug information
            st.markdown("### 🔧 Debug Information")
            debug_info = {
                'Sectors Loaded': len(config.SECTORS.get('SECTORS', {})) + len(config.SECTORS.get('INDICES', {})),
                'Aspect Events': len(aspect_df) if not aspect_df.empty else 0,
                'KP Events': len(kp_df) if not kp_df.empty else 0,
                'Analysis Date': str(user_config['date']),
                'Timezone': user_config['timezone'],
                'Cache Status': 'Active' if hasattr(st, 'cache_data') else 'Disabled'
            }
            
            st.json(debug_info)
            
        except Exception as e:
            st.error(f"Settings tab error: {str(e)}")
            st.info("Basic configuration display:")

if __name__ == "__main__":
    main()
