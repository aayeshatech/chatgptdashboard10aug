import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pytz
import calendar as _cal
import io
import json
import base64

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
    """Complete configuration for the app"""
    
    SECTORS: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "INDICES": {
            "NIFTY50": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "ITC", "HINDUNILVR", "LT", "SBIN"],
            "BANKNIFTY": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "PNB", "BANDHANBNK", "FEDERALBNK"],
            "DOWJONES": ["AAPL", "MSFT", "UNH", "GS", "HD", "MCD", "V", "CAT", "AMGN", "CRM"]
        },
        "SECTORS": {
            "PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "AUROPHARMA"],
            "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "EICHERMOT", "HEROMOTOCO"],
            "FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR"],
            "METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "SAIL"],
            "OIL_GAS": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL"],
            "TELECOM": ["BHARTIARTL", "IDEA", "RJIO"],
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"]
        },
        "COMMODITIES": {
            "GOLD": ["GOLD_SPOT", "GLD_ETF", "GOLDIAM", "TITAN", "KALYAN"],
            "SILVER": ["SILVER_SPOT", "SLV_ETF", "HINDZINC", "VEDL", "SILVERCORP"],
            "CRUDE": ["CRUDE_WTI", "CRUDE_BRENT", "RELIANCE", "ONGC", "IOC"],
            "CRYPTO": ["BTC_USD", "ETH_USD", "BNB_USD", "ADA_USD", "SOL_USD"],
            "ENERGY": ["POWERGRID", "NTPC", "COALINDIA", "ADANIGREEN", "TATAPOWER"]
        }
    })

config = AppConfig()

# Transit-to-Sector Mapping System
class TransitSectorMapping:
    def __init__(self):
        # Planetary influences on sectors
        self.planetary_sector_influence = {
            "Jupiter": {
                "strong": ["BANKNIFTY", "PHARMA", "FMCG", "NIFTY50"],
                "moderate": ["IT", "AUTO", "TELECOM"],
                "weak": ["METAL", "OIL_GAS"]
            },
            "Venus": {
                "strong": ["FMCG", "AUTO", "GOLD", "SILVER"],
                "moderate": ["PHARMA", "IT", "CRYPTO"],
                "weak": ["METAL", "ENERGY"]
            },
            "Mars": {
                "strong": ["METAL", "ENERGY", "OIL_GAS", "AUTO"],
                "moderate": ["NIFTY50", "CRUDE"],
                "weak": ["PHARMA", "FMCG"]
            },
            "Mercury": {
                "strong": ["IT", "TELECOM", "BANKNIFTY", "CRYPTO"],
                "moderate": ["NIFTY50", "AUTO"],
                "weak": ["METAL", "OIL_GAS"]
            },
            "Saturn": {
                "strong": ["METAL", "ENERGY", "OIL_GAS"],
                "moderate": ["AUTO", "BANKNIFTY"],
                "weak": ["IT", "PHARMA", "FMCG"]
            },
            "Moon": {
                "strong": ["FMCG", "PHARMA", "SILVER"],
                "moderate": ["NIFTY50", "BANKNIFTY"],
                "weak": ["METAL", "CRYPTO"]
            },
            "Sun": {
                "strong": ["ENERGY", "GOLD", "NIFTY50"],
                "moderate": ["PHARMA", "AUTO"],
                "weak": ["SILVER", "CRYPTO"]
            },
            "Rahu": {
                "strong": ["CRYPTO", "IT", "TELECOM"],
                "moderate": ["METAL", "ENERGY"],
                "weak": ["FMCG", "PHARMA"]
            },
            "Ketu": {
                "strong": ["GOLD", "SILVER", "PHARMA"],
                "moderate": ["IT", "ENERGY"],
                "weak": ["AUTO", "FMCG"]
            }
        }
        
        # Transit effect durations (in days)
        self.effect_duration = {
            "Jupiter": {"fast": 15, "slow": 45, "major": 120},
            "Saturn": {"fast": 30, "slow": 90, "major": 365},
            "Mars": {"fast": 3, "slow": 7, "major": 21},
            "Venus": {"fast": 5, "slow": 12, "major": 30},
            "Mercury": {"fast": 2, "slow": 5, "major": 15},
            "Sun": {"fast": 1, "slow": 3, "major": 7},
            "Moon": {"fast": 1, "slow": 2, "major": 3},
            "Rahu": {"fast": 21, "slow": 60, "major": 180},
            "Ketu": {"fast": 21, "slow": 60, "major": 180}
        }
    
    def get_affected_sectors(self, planet_a: str, planet_b: str, aspect: str, score: float) -> Dict:
        """Get sectors affected by a planetary transit"""
        affected_sectors = {"bullish": [], "bearish": [], "neutral": []}
        
        # Determine if transit is bullish or bearish
        is_bullish = score > 0
        
        # Get sectors influenced by both planets
        for planet in [planet_a, planet_b]:
            if planet in self.planetary_sector_influence:
                influence = self.planetary_sector_influence[planet]
                
                # Strong influence sectors
                for sector in influence.get("strong", []):
                    if is_bullish:
                        if sector not in affected_sectors["bullish"]:
                            affected_sectors["bullish"].append(sector)
                    else:
                        if sector not in affected_sectors["bearish"]:
                            affected_sectors["bearish"].append(sector)
                
                # Moderate influence sectors
                for sector in influence.get("moderate", []):
                    if sector not in affected_sectors["bullish"] and sector not in affected_sectors["bearish"]:
                        if abs(score) > 1.0:  # Only if strong transit
                            if is_bullish:
                                affected_sectors["bullish"].append(sector)
                            else:
                                affected_sectors["bearish"].append(sector)
                        else:
                            affected_sectors["neutral"].append(sector)
        
        return affected_sectors
    
    def calculate_effect_duration(self, planet_a: str, planet_b: str, aspect: str, score: float) -> int:
        """Calculate how many days the transit effect will last"""
        # Base duration from slower planet
        planets = [planet_a, planet_b]
        max_duration = 0
        
        for planet in planets:
            if planet in self.effect_duration:
                duration_map = self.effect_duration[planet]
                
                # Determine duration type based on score intensity
                abs_score = abs(score)
                if abs_score >= 2.5:
                    duration = duration_map["major"]
                elif abs_score >= 1.5:
                    duration = duration_map["slow"]
                else:
                    duration = duration_map["fast"]
                
                max_duration = max(max_duration, duration)
        
        # Aspect type modifier
        aspect_modifiers = {
            "Conjunction": 1.5,
            "Opposition": 1.3,
            "Square": 1.2,
            "Trine": 1.0,
            "Sextile": 0.8
        }
        
        modifier = aspect_modifiers.get(aspect, 1.0)
        final_duration = int(max_duration * modifier)
        
        return max(1, min(final_duration, 365))  # Between 1 and 365 days

# Initialize transit mapping
transit_mapping = TransitSectorMapping()

# Enhanced Scoring system
class EnhancedScoring:
    def __init__(self):
        self.planetary_weights = {
            "benefics": {"Jupiter": 2.5, "Venus": 2.0, "Mercury": 1.2, "Moon": 1.5},
            "malefics": {"Saturn": -2.5, "Mars": -2.0, "Rahu": -1.8, "Ketu": -1.5},
            "sun": 0.8
        }
        
        self.aspect_weights = {
            "Trine": 1.2, "Sextile": 1.0, "Conjunction": 0.8,
            "Opposition": -1.0, "Square": -1.2
        }
        
        self.asset_biases = {
            "NIFTY50": {"Jupiter": 0.5, "Saturn": -0.3, "Mercury": 0.2},
            "BANKNIFTY": {"Jupiter": 0.6, "Saturn": -0.5, "Mercury": 0.3},
            "GOLD": {"Saturn": -0.6, "Jupiter": 0.4, "Venus": 0.2},
            "CRYPTO": {"Rahu": 0.7, "Saturn": -0.4, "Mercury": 0.4}
        }
    
    def calculate_sector_score(self, sector_name: str, date_str: str, hour: int = 12) -> Dict:
        """Calculate comprehensive score for a sector"""
        # Base score using hash for consistency
        base_hash = hash(sector_name + date_str + str(hour)) % 1000
        base_score = (base_hash - 500) / 100  # Range from -5 to +5
        
        # Add time-based variation
        time_factor = np.sin(hour * np.pi / 12) * 0.5  # Sinusoidal variation
        final_score = base_score + time_factor
        
        # Calculate confidence
        confidence = min(0.95, 0.6 + abs(final_score) * 0.1)
        
        # Risk level
        abs_score = abs(final_score)
        if abs_score > 2.5:
            risk_level = "High"
        elif abs_score > 1.0:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'score': round(final_score, 2),
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'trend': self.get_trend_direction(final_score),
            'strength': self.get_signal_strength(final_score)
        }
    
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
    
    def get_trend_emoji(self, score: float) -> str:
        """Get appropriate emoji for trend"""
        if score >= 2.0:
            return "🚀"
        elif score >= 1.0:
            return "📈"
        elif score >= 0.5:
            return "🟢"
        elif score >= -0.5:
            return "⚪"
        elif score >= -1.0:
            return "🟠"
        elif score >= -2.0:
            return "📉"
        else:
            return "💥"

# Initialize scoring engine
scoring = EnhancedScoring()

# Planetary Transit System
class PlanetaryTransits:
    def __init__(self):
        self.planets = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]
        self.aspects = ["Conjunction", "Sextile", "Square", "Trine", "Opposition"]
        self.nakshatras = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu"]
    
    def generate_daily_transits_enhanced(self, date_input: datetime.date) -> pd.DataFrame:
        """Generate enhanced planetary transits with sector mapping"""
        transits = []
        base_time = datetime.combine(date_input, dtime(9, 0))
        
        # Generate 15-20 transit events throughout the day
        for i in range(18):
            event_time = base_time + timedelta(minutes=i*25 + np.random.randint(-10, 10))
            
            # Select planets and aspect
            planet_a = self.planets[i % len(self.planets)]
            planet_b = self.planets[(i + 3) % len(self.planets)]
            aspect = self.aspects[i % len(self.aspects)]
            
            # Calculate score
            score = self._calculate_transit_score(planet_a, planet_b, aspect, i)
            
            # Get affected sectors
            affected_sectors = transit_mapping.get_affected_sectors(planet_a, planet_b, aspect, score)
            
            # Calculate effect duration
            effect_duration = transit_mapping.calculate_effect_duration(planet_a, planet_b, aspect, score)
            
            # Moon position for KP
            moon_nak = self.nakshatras[i % len(self.nakshatras)]
            
            transits.append({
                "Time": event_time.strftime("%H:%M"),
                "Planet A": planet_a,
                "Planet B": planet_b,
                "Aspect": aspect,
                "Exact°": round((i * 13.7) % 360, 1),
                "Score": score,
                "Signal": scoring.get_trend_direction(score),
                "Strength": scoring.get_signal_strength(score),
                "Effect Duration (Days)": effect_duration,
                "Bullish Sectors": ", ".join(affected_sectors["bullish"][:3]),
                "Bearish Sectors": ", ".join(affected_sectors["bearish"][:3]),
                "Neutral Sectors": ", ".join(affected_sectors["neutral"][:2]),
                "Moon Nakshatra": moon_nak,
                "Star Lord": self.planets[(i + 1) % len(self.planets)],
                "Sub Lord": self.planets[(i + 2) % len(self.planets)]
            })
        
        return pd.DataFrame(transits)
    
    def get_sector_specific_transits(self, date_input: datetime.date, sector_name: str) -> pd.DataFrame:
        """Get transits specifically affecting a chosen sector"""
        all_transits = self.generate_daily_transits_enhanced(date_input)
        
        # Filter transits affecting the selected sector
        sector_transits = []
        
        for _, transit in all_transits.iterrows():
            is_affected = (
                sector_name in transit["Bullish Sectors"] or
                sector_name in transit["Bearish Sectors"] or
                sector_name in transit["Neutral Sectors"]
            )
            
            if is_affected:
                # Determine sector effect
                if sector_name in transit["Bullish Sectors"]:
                    sector_effect = "Bullish"
                    sector_score = abs(transit["Score"])
                elif sector_name in transit["Bearish Sectors"]:
                    sector_effect = "Bearish"
                    sector_score = -abs(transit["Score"])
                else:
                    sector_effect = "Neutral"
                    sector_score = transit["Score"] * 0.3
                
                sector_transits.append({
                    "Time": transit["Time"],
                    "Transit": f"{transit['Planet A']} {transit['Aspect']} {transit['Planet B']}",
                    "Effect on Sector": sector_effect,
                    "Sector Score": round(sector_score, 2),
                    "Duration (Days)": transit["Effect Duration (Days)"],
                    "Strength": transit["Strength"],
                    "Star Lord": transit["Star Lord"],
                    "Sub Lord": transit["Sub Lord"]
                })
        
        return pd.DataFrame(sector_transits)

# Stock-Level Analysis System
class StockAnalysisEngine:
    def __init__(self):
        # Stock-specific planetary influences (simplified)
        self.stock_planetary_bias = {
            # IT Stocks
            "TCS": {"Mercury": 0.8, "Jupiter": 0.6, "Saturn": -0.3},
            "INFY": {"Mercury": 0.7, "Jupiter": 0.5, "Rahu": 0.4},
            "WIPRO": {"Mercury": 0.6, "Venus": 0.4, "Saturn": -0.2},
            
            # Banking Stocks
            "HDFCBANK": {"Jupiter": 0.9, "Venus": 0.5, "Saturn": -0.4},
            "ICICIBANK": {"Jupiter": 0.8, "Mercury": 0.4, "Mars": -0.3},
            "SBIN": {"Jupiter": 0.7, "Sun": 0.4, "Saturn": -0.5},
            
            # Auto Stocks
            "TATAMOTORS": {"Mars": 0.8, "Venus": 0.6, "Saturn": -0.4},
            "MARUTI": {"Venus": 0.7, "Mercury": 0.5, "Jupiter": 0.4},
            
            # Pharma Stocks
            "SUNPHARMA": {"Jupiter": 0.8, "Moon": 0.6, "Venus": 0.4},
            "CIPLA": {"Jupiter": 0.7, "Venus": 0.5, "Mercury": 0.3},
            
            # Commodity Related
            "RELIANCE": {"Sun": 0.8, "Mars": 0.6, "Jupiter": 0.5},
            "TATASTEEL": {"Mars": 0.9, "Saturn": -0.6, "Sun": 0.4}
        }
    
    def analyze_stock_transits(self, stock_symbol: str, sector_transits: pd.DataFrame, 
                              start_date: datetime.date, days_ahead: int = 7) -> Dict:
        """Analyze how planetary transits affect a specific stock"""
        
        stock_bias = self.stock_planetary_bias.get(stock_symbol, {})
        stock_analysis = {
            "symbol": stock_symbol,
            "bullish_periods": [],
            "bearish_periods": [],
            "neutral_periods": [],
            "peak_bullish_time": None,
            "peak_bearish_time": None,
            "total_effect_days": 0,
            "next_major_event": None
        }
        
        if sector_transits.empty:
            return stock_analysis
        
        current_date = start_date
        max_bullish_score = -999
        max_bearish_score = 999
        
        # Analyze each transit's effect on the stock
        for _, transit in sector_transits.iterrows():
            # Extract planets from transit description
            transit_parts = transit["Transit"].split()
            if len(transit_parts) >= 3:
                planet_a = transit_parts[0]
                planet_b = transit_parts[2]
                
                # Calculate stock-specific score
                stock_score = 0
                if planet_a in stock_bias:
                    stock_score += stock_bias[planet_a]
                if planet_b in stock_bias:
                    stock_score += stock_bias[planet_b]
                
                # Apply sector effect
                sector_multiplier = 1.0
                if transit["Effect on Sector"] == "Bullish":
                    sector_multiplier = 1.2
                elif transit["Effect on Sector"] == "Bearish":
                    sector_multiplier = -1.2
                else:
                    sector_multiplier = 0.5
                
                final_score = stock_score * sector_multiplier
                
                # Determine time windows
                start_time = datetime.strptime(transit["Time"], "%H:%M").time()
                duration_hours = min(6, transit["Duration (Days)"]) # Convert to hours for intraday
                end_time = (datetime.combine(current_date, start_time) + timedelta(hours=duration_hours)).time()
                
                time_window = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
                
                # Categorize the period
                if final_score > 0.5:
                    stock_analysis["bullish_periods"].append({
                        "time": time_window,
                        "score": round(final_score, 2),
                        "strength": transit["Strength"],
                        "duration_days": transit["Duration (Days)"],
                        "planets": f"{planet_a}-{planet_b}"
                    })
                    
                    if final_score > max_bullish_score:
                        max_bullish_score = final_score
                        stock_analysis["peak_bullish_time"] = transit["Time"]
                
                elif final_score < -0.5:
                    stock_analysis["bearish_periods"].append({
                        "time": time_window,
                        "score": round(final_score, 2),
                        "strength": transit["Strength"],
                        "duration_days": transit["Duration (Days)"],
                        "planets": f"{planet_a}-{planet_b}"
                    })
                    
                    if final_score < max_bearish_score:
                        max_bearish_score = final_score
                        stock_analysis["peak_bearish_time"] = transit["Time"]
                
                else:
                    stock_analysis["neutral_periods"].append({
                        "time": time_window,
                        "score": round(final_score, 2),
                        "strength": transit["Strength"],
                        "duration_days": transit["Duration (Days)"],
                        "planets": f"{planet_a}-{planet_b}"
                    })
                
                # Track total effect duration
                stock_analysis["total_effect_days"] += transit["Duration (Days)"]
        
        # Find next major event
        future_events = [p for p in stock_analysis["bullish_periods"] + stock_analysis["bearish_periods"] 
                        if abs(p["score"]) >= 1.0]
        if future_events:
            stock_analysis["next_major_event"] = future_events[0]
        
        return stock_analysis

# Initialize stock analysis engine
stock_engine = StockAnalysisEngine()
        """Generate planetary transits for a single day"""
        transits = []
        base_time = datetime.combine(date_input, dtime(9, 0))
        
        # Generate 15-20 transit events throughout the day
        for i in range(18):
            event_time = base_time + timedelta(minutes=i*25 + np.random.randint(-10, 10))
            
            # Select planets and aspect
            planet_a = self.planets[i % len(self.planets)]
            planet_b = self.planets[(i + 3) % len(self.planets)]
            aspect = self.aspects[i % len(self.aspects)]
            
            # Calculate score
            score = self._calculate_transit_score(planet_a, planet_b, aspect, i)
            
            # Moon position for KP
            moon_nak = self.nakshatras[i % len(self.nakshatras)]
            
            transits.append({
                "Time": event_time.strftime("%H:%M"),
                "Planet A": planet_a,
                "Planet B": planet_b,
                "Aspect": aspect,
                "Exact°": (i * 13.7) % 360,  # Varied degrees
                "Score": score,
                "Signal": scoring.get_trend_direction(score),
                "Strength": scoring.get_signal_strength(score),
                "Moon Nakshatra": moon_nak,
                "Star Lord": self.planets[(i + 1) % len(self.planets)],
                "Sub Lord": self.planets[(i + 2) % len(self.planets)]
            })
        
        return pd.DataFrame(transits)
    
    def _calculate_transit_score(self, planet_a: str, planet_b: str, aspect: str, seed: int) -> float:
        """Calculate score for a transit"""
        # Base planetary influences
        score = 0.0
        
        # Planet A influence
        if planet_a in ["Jupiter", "Venus", "Mercury", "Moon"]:
            score += 1.0 + (seed % 3) * 0.5
        elif planet_a in ["Saturn", "Mars", "Rahu", "Ketu"]:
            score -= 1.0 + (seed % 3) * 0.5
        else:  # Sun
            score += 0.5
        
        # Planet B influence
        if planet_b in ["Jupiter", "Venus", "Mercury", "Moon"]:
            score += 0.8 + (seed % 2) * 0.3
        elif planet_b in ["Saturn", "Mars", "Rahu", "Ketu"]:
            score -= 0.8 + (seed % 2) * 0.3
        else:  # Sun
            score += 0.3
        
        # Aspect influence
        aspect_multiplier = scoring.aspect_weights.get(aspect, 1.0)
        score *= aspect_multiplier
        
        # Add some randomness
        score += (seed % 100 - 50) / 100
        
        return round(score, 2)
    
    def generate_upcoming_transits(self, start_date: datetime.date, days: int = 7) -> List[Dict]:
        """Generate upcoming major transits"""
        upcoming = []
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            
            # Generate 2-3 major transits per day
            for j in range(2 + i % 2):
                transit_time = datetime.combine(current_date, dtime(10 + j * 4, 30))
                
                planet_a = self.planets[(i + j) % len(self.planets)]
                planet_b = self.planets[(i + j + 2) % len(self.planets)]
                aspect = self.aspects[(i + j) % len(self.aspects)]
                
                score = self._calculate_transit_score(planet_a, planet_b, aspect, i * 10 + j)
                
                upcoming.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": transit_time.strftime("%H:%M"),
                    "Event": f"{planet_a} {aspect} {planet_b}",
                    "Score": score,
                    "Impact": "High" if abs(score) > 2 else "Medium" if abs(score) > 1 else "Low",
                    "Duration": f"{2 + j} hours",
                    "Sectors Affected": self._get_affected_sectors(planet_a, planet_b, score)
                })
        
        return upcoming
    
    def _get_affected_sectors(self, planet_a: str, planet_b: str, score: float) -> str:
        """Get sectors most affected by transit"""
        sector_influences = {
            "Jupiter": ["BANKNIFTY", "PHARMA", "IT"],
            "Venus": ["FMCG", "AUTO", "GOLD"],
            "Mars": ["METAL", "ENERGY", "OIL_GAS"],
            "Saturn": ["INFRASTRUCTURE", "METAL"],
            "Mercury": ["IT", "TELECOM", "BANKNIFTY"],
            "Rahu": ["CRYPTO", "TECH"],
            "Moon": ["FMCG", "PHARMA"]
        }
        
        affected = set()
        affected.update(sector_influences.get(planet_a, []))
        affected.update(sector_influences.get(planet_b, []))
        
        return ", ".join(list(affected)[:3]) if affected else "General Market"

# Initialize transit system
transits = PlanetaryTransits()

# Styling (same as before)
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
        
        .transit-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# UI Components
def create_main_header():
    st.markdown("""
    <div class="main-header">
        <h1>🪐 Vedic Market Analytics Pro</h1>
        <p>Complete Astrological Market Analysis & Planetary Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create comprehensive sidebar controls"""
    st.sidebar.markdown("## 🎛️ Analysis Configuration")
    
    with st.sidebar.expander("📅 Date & Time Settings", expanded=True):
        date_input = st.date_input("Analysis Date", value=datetime.today().date())
        timezone = st.selectbox("Timezone", ["Asia/Kolkata", "America/New_York", "Europe/London", "Asia/Tokyo"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Market Start", value=dtime(9, 15))
        with col2:
            end_time = st.time_input("Market End", value=dtime(15, 30))
    
    with st.sidebar.expander("⚙️ Analysis Parameters", expanded=False):
        kp_premium = st.slider("KP Weight Multiplier", 0.5, 3.0, 1.3, 0.1)
        sensitivity = st.slider("Signal Sensitivity", 0.5, 3.0, 1.5, 0.1)
        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
        time_resolution = st.selectbox("Time Resolution", ["15 minutes", "30 minutes", "1 hour"])
    
    with st.sidebar.expander("⚖️ Aspect Weights", expanded=False):
        st.markdown("**Adjust planetary aspect influences:**")
        aspect_weights = {}
        for aspect in ["Trine", "Sextile", "Conjunction", "Opposition", "Square"]:
            default_val = scoring.aspect_weights.get(aspect, 1.0)
            aspect_weights[aspect] = st.slider(f"{aspect}", -2.0, 2.0, float(default_val), 0.1)
    
    with st.sidebar.expander("📥 Export & Sharing", expanded=False):
        export_format = st.selectbox("Export Format", ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"])
        include_charts = st.checkbox("Include Charts", value=True)
        include_transits = st.checkbox("Include Transit Data", value=True)
    
    return {
        'date': date_input,
        'timezone': timezone,
        'start_time': start_time,
        'end_time': end_time,
        'kp_premium': kp_premium,
        'sensitivity': sensitivity,
        'confidence_threshold': confidence_threshold,
        'time_resolution': time_resolution,
        'aspect_weights': aspect_weights,
        'export_format': export_format,
        'include_charts': include_charts,
        'include_transits': include_transits
    }

def analyze_sectors_with_transits(user_config, time_hour=12):
    """Enhanced sector analysis with planetary transit filtering"""
    results = []
    date_str = str(user_config['date'])
    
    # Get daily transits for filtering
    daily_transits = transits.generate_daily_transits_enhanced(user_config['date'])
    
    # Flatten all sectors
    all_sectors = {}
    for category, sector_dict in config.SECTORS.items():
        all_sectors.update(sector_dict)
    
    for sector_name, stocks in all_sectors.items():
        # Base sector analysis
        analysis = scoring.calculate_sector_score(sector_name, date_str, time_hour)
        adjusted_score = analysis['score'] * user_config['sensitivity']
        
        # Get planetary transit effects for this sector
        transit_effects = get_sector_transit_effects(sector_name, daily_transits)
        
        # Combine base score with transit effects
        final_score = adjusted_score + transit_effects['score_adjustment']
        
        results.append({
            'Sector': sector_name,
            'Net Score': round(final_score, 2),
            'Base Score': round(adjusted_score, 2),
            'Transit Adjustment': round(transit_effects['score_adjustment'], 2),
            'Avg/Stock': round(final_score / len(stocks), 2),
            'Trend': scoring.get_trend_direction(final_score),
            'Signal Strength': scoring.get_signal_strength(final_score),
            'Confidence': analysis['confidence'],
            'Risk Level': analysis['risk_level'],
            'Stocks Count': len(stocks),
            'Top Stocks': ', '.join(stocks[:3]),
            'Emoji': scoring.get_trend_emoji(final_score),
            'Affecting Transits': transit_effects['count'],
            'Effect Duration': transit_effects['max_duration'],
            'Primary Planet': transit_effects['primary_planet']
        })
    
    return sorted(results, key=lambda x: x['Net Score'], reverse=True)

def get_sector_transit_effects(sector_name: str, daily_transits: pd.DataFrame) -> Dict:
    """Calculate how planetary transits affect a specific sector"""
    effects = {
        'score_adjustment': 0.0,
        'count': 0,
        'max_duration': 0,
        'primary_planet': 'None'
    }
    
    if daily_transits.empty:
        return effects
    
    planets_affecting = {}
    
    for _, transit in daily_transits.iterrows():
        sector_effect = 0.0
        
        # Check if sector is mentioned in transit effects
        if sector_name in transit.get("Bullish Sectors", ""):
            sector_effect = abs(transit["Score"]) * 0.8
            effects['count'] += 1
        elif sector_name in transit.get("Bearish Sectors", ""):
            sector_effect = -abs(transit["Score"]) * 0.8
            effects['count'] += 1
        elif sector_name in transit.get("Neutral Sectors", ""):
            sector_effect = transit["Score"] * 0.3
            effects['count'] += 1
        
        if sector_effect != 0:
            effects['score_adjustment'] += sector_effect
            effects['max_duration'] = max(effects['max_duration'], transit.get("Effect Duration (Days)", 0))
            
            # Track planets
            planet_a = transit.get("Planet A", "")
            planet_b = transit.get("Planet B", "")
            
            for planet in [planet_a, planet_b]:
                if planet:
                    planets_affecting[planet] = planets_affecting.get(planet, 0) + abs(sector_effect)
    
    # Find primary affecting planet
    if planets_affecting:
        effects['primary_planet'] = max(planets_affecting, key=planets_affecting.get)
    
    return effects

def generate_weekly_analysis_enhanced(user_config):
    """Enhanced weekly analysis with planetary transit filtering"""
    weekly_data = []
    start_date = user_config['date']
    
    # Get week dates
    start_of_week = start_date - timedelta(days=start_date.weekday())
    
    for i in range(7):
        day_date = start_of_week + timedelta(days=i)
        day_config = {**user_config, 'date': day_date}
        day_results = analyze_sectors_with_transits(day_config, time_hour=12)
        
        if day_results:
            # Separate bullish and bearish sectors by transit effects
            bullish_sectors = [r for r in day_results if r['Net Score'] > 0 and r['Affecting Transits'] > 0]
            bearish_sectors = [r for r in day_results if r['Net Score'] < 0 and r['Affecting Transits'] > 0]
            
            top_sector = day_results[0]
            bottom_sector = day_results[-1]
            
            # Calculate market sentiment based on transit-affected sectors
            bullish_transit_count = len(bullish_sectors)
            bearish_transit_count = len(bearish_sectors)
            
            weekly_data.append({
                'Date': day_date.strftime('%Y-%m-%d'),
                'Day': day_date.strftime('%A'),
                'Top Bullish': top_sector['Sector'],
                'Bullish Score': top_sector['Net Score'],
                'Bullish Transits': bullish_transit_count,
                'Top Bearish': bottom_sector['Sector'],
                'Bearish Score': bottom_sector['Net Score'],
                'Bearish Transits': bearish_transit_count,
                'Market Sentiment': 'Bullish' if bullish_transit_count > bearish_transit_count else 'Bearish',
                'Volatility': round(abs(top_sector['Net Score'] - bottom_sector['Net Score']), 2),
                'Primary Planet': top_sector['Primary Planet'],
                'Max Effect Duration': top_sector['Effect Duration']
            })
    
    return pd.DataFrame(weekly_data)

def generate_monthly_analysis_enhanced(user_config):
    """Enhanced monthly analysis with planetary transit filtering"""
    monthly_data = []
    
    # Get month boundaries
    first_day = user_config['date'].replace(day=1)
    if first_day.month == 12:
        last_day = first_day.replace(year=first_day.year + 1, month=1) - timedelta(days=1)
    else:
        last_day = first_day.replace(month=first_day.month + 1) - timedelta(days=1)
    
    current_date = first_day
    while current_date <= last_day:
        day_config = {**user_config, 'date': current_date}
        day_results = analyze_sectors_with_transits(day_config, time_hour=12)
        
        if day_results:
            # Filter only transit-affected sectors
            transit_affected = [r for r in day_results if r['Affecting Transits'] > 0]
            
            if transit_affected:
                avg_score = sum(r['Net Score'] for r in transit_affected) / len(transit_affected)
                top_sector = transit_affected[0]
                
                # Count major transits (effect duration > 7 days)
                major_transits = len([r for r in transit_affected if r['Effect Duration'] > 7])
                
                monthly_data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': current_date.day,
                    'Average Score': round(avg_score, 2),
                    'Top Sector': top_sector['Sector'],
                    'Top Score': top_sector['Net Score'],
                    'Market Bias': 'Bullish' if avg_score > 0 else 'Bearish',
                    'Transit Affected Sectors': len(transit_affected),
                    'Major Transits': major_transits,
                    'Primary Planet': top_sector['Primary Planet'],
                    'Max Duration': max(r['Effect Duration'] for r in transit_affected)
                })
            else:
                # Day with minimal transit effects
                monthly_data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': current_date.day,
                    'Average Score': 0.0,
                    'Top Sector': 'None',
                    'Top Score': 0.0,
                    'Market Bias': 'Neutral',
                    'Transit Affected Sectors': 0,
                    'Major Transits': 0,
                    'Primary Planet': 'None',
                    'Max Duration': 0
                })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(monthly_data)

def create_sector_filter_interface():
    """Create filtering interface for sectors by planetary transits"""
    st.markdown("### 🔍 Filter Sectors by Planetary Transits")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_filter = st.multiselect(
            "📈 Trend Filter",
            ["Bullish", "Bearish", "Neutral"],
            default=["Bullish", "Bearish"]
        )
    
    with col2:
        planet_filter = st.multiselect(
            "🪐 Primary Planet",
            ["Jupiter", "Venus", "Mars", "Mercury", "Saturn", "Sun", "Moon", "Rahu", "Ketu"],
            default=[]
        )
    
    with col3:
        duration_filter = st.slider(
            "⏰ Min Effect Duration (Days)",
            0, 30, 0
        )
    
    return {
        'trend_filter': trend_filter,
        'planet_filter': planet_filter,
        'duration_filter': duration_filter
    }

def apply_sector_filters(sector_results: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to sector results"""
    filtered_results = sector_results.copy()
    
    # Apply trend filter
    if filters['trend_filter']:
        filtered_results = [r for r in filtered_results if r['Trend'] in filters['trend_filter']]
    
    # Apply planet filter
    if filters['planet_filter']:
        filtered_results = [r for r in filtered_results if r['Primary Planet'] in filters['planet_filter']]
    
    # Apply duration filter
    if filters['duration_filter'] > 0:
        filtered_results = [r for r in filtered_results if r['Effect Duration'] >= filters['duration_filter']]
    
    return filtered_results

def create_stock_analysis_interface(sector_results: List[Dict], user_config: Dict):
    """Create interface for detailed stock analysis within sectors"""
    st.markdown("### 🎯 Stock-Level Transit Analysis")
    
    if not sector_results:
        st.warning("No sectors available for stock analysis.")
        return
    
    # Sector selection
    sector_names = [r['Sector'] for r in sector_results if r['Affecting Transits'] > 0]
    
    if not sector_names:
        st.info("No sectors with active planetary transits found.")
        return
    
    selected_sector = st.selectbox(
        "📊 Select Sector for Stock Analysis",
        sector_names,
        help="Choose a sector to see individual stock analysis"
    )
    
    if selected_sector:
        # Get sector data
        sector_data = next(r for r in sector_results if r['Sector'] == selected_sector)
        
        # Display sector overview
        st.markdown(f"#### 🏭 {selected_sector} Sector Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sector Score", f"{sector_data['Net Score']:.2f}")
        with col2:
            st.metric("Active Transits", sector_data['Affecting Transits'])
        with col3:
            st.metric("Effect Duration", f"{sector_data['Effect Duration']} days")
        with col4:
            st.metric("Primary Planet", sector_data['Primary Planet'])
        
        # Get sector-specific transits
        sector_transits = transits.get_sector_specific_transits(user_config['date'], selected_sector)
        
        if not sector_transits.empty:
            st.markdown("#### 🪐 Transits Affecting This Sector")
            st.dataframe(sector_transits, use_container_width=True)
            
            # Get stocks in this sector
            all_sectors = {}
            for category, sector_dict in config.SECTORS.items():
                all_sectors.update(sector_dict)
            
            sector_stocks = all_sectors.get(selected_sector, [])
            
            if sector_stocks:
                st.markdown("#### 📈 Individual Stock Analysis")
                
                # Stock selection
                selected_stock = st.selectbox(
                    "🎯 Select Stock for Detailed Analysis",
                    sector_stocks,
                    help="Choose a stock to see detailed timing analysis"
                )
                
                if selected_stock:
                    # Analyze the selected stock
                    stock_analysis = stock_engine.analyze_stock_transits(
                        selected_stock, 
                        sector_transits, 
                        user_config['date'], 
                        days_ahead=7
                    )
                    
                    # Display stock analysis
                    st.markdown(f"##### 📊 {selected_stock} - Detailed Transit Analysis")
                    
                    # Stock metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Bullish Periods", len(stock_analysis["bullish_periods"]))
                    with col2:
                        st.metric("Bearish Periods", len(stock_analysis["bearish_periods"]))
                    with col3:
                        st.metric("Total Effect Days", stock_analysis["total_effect_days"])
                    with col4:
                        peak_time = stock_analysis["peak_bullish_time"] or stock_analysis["peak_bearish_time"] or "None"
                        st.metric("Peak Impact Time", peak_time)
                    
                    # Bullish periods
                    if stock_analysis["bullish_periods"]:
                        st.markdown("##### 🟢 Bullish Time Windows")
                        for period in stock_analysis["bullish_periods"]:
                            st.markdown(f"""
                            <div class="alert-box alert-success">
                                <strong>⏰ {period['time']}</strong><br>
                                <strong>Score:</strong> +{period['score']:.2f} | 
                                <strong>Strength:</strong> {period['strength']} | 
                                <strong>Duration:</strong> {period['duration_days']} days<br>
                                <strong>Planets:</strong> {period['planets']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Bearish periods
                    if stock_analysis["bearish_periods"]:
                        st.markdown("##### 🔴 Bearish Time Windows")
                        for period in stock_analysis["bearish_periods"]:
                            st.markdown(f"""
                            <div class="alert-box alert-danger">
                                <strong>⏰ {period['time']}</strong><br>
                                <strong>Score:</strong> {period['score']:.2f} | 
                                <strong>Strength:</strong> {period['strength']} | 
                                <strong>Duration:</strong> {period['duration_days']} days<br>
                                <strong>Planets:</strong> {period['planets']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Neutral periods
                    if stock_analysis["neutral_periods"]:
                        st.markdown("##### ⚪ Neutral Time Windows")
                        for period in stock_analysis["neutral_periods"]:
                            st.markdown(f"""
                            <div class="alert-box" style="background-color: #f9fafb; border-left-color: #6b7280;">
                                <strong>⏰ {period['time']}</strong><br>
                                <strong>Score:</strong> {period['score']:.2f} | 
                                <strong>Strength:</strong> {period['strength']} | 
                                <strong>Duration:</strong> {period['duration_days']} days<br>
                                <strong>Planets:</strong> {period['planets']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Next major event
                    if stock_analysis["next_major_event"]:
                        event = stock_analysis["next_major_event"]
                        st.markdown("##### 🎯 Next Major Event")
                        event_class = "alert-success" if event["score"] > 0 else "alert-danger"
                        st.markdown(f"""
                        <div class="alert-box {event_class}">
                            <strong>📅 Next Significant Movement</strong><br>
                            <strong>Time:</strong> {event['time']}<br>
                            <strong>Expected Impact:</strong> {event['score']:.2f}<br>
                            <strong>Duration:</strong> {event['duration_days']} days<br>
                            <strong>Planets:</strong> {event['planets']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Export stock analysis
                    if st.button(f"📥 Export {selected_stock} Analysis"):
                        # Create export data
                        export_data = {
                            "Stock": selected_stock,
                            "Sector": selected_sector,
                            "Analysis_Date": str(user_config['date']),
                            "Bullish_Periods": stock_analysis["bullish_periods"],
                            "Bearish_Periods": stock_analysis["bearish_periods"],
                            "Neutral_Periods": stock_analysis["neutral_periods"],
                            "Peak_Bullish_Time": stock_analysis["peak_bullish_time"],
                            "Peak_Bearish_Time": stock_analysis["peak_bearish_time"],
                            "Total_Effect_Days": stock_analysis["total_effect_days"],
                            "Next_Major_Event": stock_analysis["next_major_event"]
                        }
                        
                        st.download_button(
                            "💾 Download Stock Analysis",
                            json.dumps(export_data, indent=2),
                            f"{selected_stock}_transit_analysis_{user_config['date']}.json",
                            "application/json"
                        )
            else:
                st.info(f"No stocks found in {selected_sector} sector configuration.")
        else:
            st.info(f"No planetary transits affecting {selected_sector} sector today.")

# Analysis Functions
def analyze_sectors_comprehensive(user_config, time_hour=12):
    """Comprehensive sector analysis"""
    results = []
    date_str = str(user_config['date'])
    
    # Flatten all sectors
    all_sectors = {}
    for category, sector_dict in config.SECTORS.items():
        all_sectors.update(sector_dict)
    
    for sector_name, stocks in all_sectors.items():
        analysis = scoring.calculate_sector_score(sector_name, date_str, time_hour)
        
        # Apply user sensitivity
        adjusted_score = analysis['score'] * user_config['sensitivity']
        
        results.append({
            'Sector': sector_name,
            'Net Score': round(adjusted_score, 2),
            'Avg/Stock': round(adjusted_score / len(stocks), 2),
            'Trend': scoring.get_trend_direction(adjusted_score),
            'Signal Strength': scoring.get_signal_strength(adjusted_score),
            'Confidence': analysis['confidence'],
            'Risk Level': analysis['risk_level'],
            'Stocks Count': len(stocks),
            'Top Stocks': ', '.join(stocks[:3]),
            'Emoji': scoring.get_trend_emoji(adjusted_score)
        })
    
    return sorted(results, key=lambda x: x['Net Score'], reverse=True)

def generate_weekly_analysis(user_config):
    """Generate week-by-week analysis"""
    weekly_data = []
    start_date = user_config['date']
    
    # Get week dates
    start_of_week = start_date - timedelta(days=start_date.weekday())
    
    for i in range(7):
        day_date = start_of_week + timedelta(days=i)
        day_results = analyze_sectors_comprehensive(
            {**user_config, 'date': day_date}, 
            time_hour=12
        )
        
        if day_results:
            top_sector = day_results[0]
            bottom_sector = day_results[-1]
            
            weekly_data.append({
                'Date': day_date.strftime('%Y-%m-%d'),
                'Day': day_date.strftime('%A'),
                'Top Bullish': top_sector['Sector'],
                'Bullish Score': top_sector['Net Score'],
                'Top Bearish': bottom_sector['Sector'],
                'Bearish Score': bottom_sector['Net Score'],
                'Market Sentiment': 'Bullish' if top_sector['Net Score'] > abs(bottom_sector['Net Score']) else 'Bearish',
                'Volatility': round(abs(top_sector['Net Score'] - bottom_sector['Net Score']), 2)
            })
    
    return pd.DataFrame(weekly_data)

def generate_monthly_analysis(user_config):
    """Generate monthly calendar analysis"""
    monthly_data = []
    
    # Get month boundaries
    first_day = user_config['date'].replace(day=1)
    if first_day.month == 12:
        last_day = first_day.replace(year=first_day.year + 1, month=1) - timedelta(days=1)
    else:
        last_day = first_day.replace(month=first_day.month + 1) - timedelta(days=1)
    
    current_date = first_day
    while current_date <= last_day:
        day_results = analyze_sectors_comprehensive(
            {**user_config, 'date': current_date}, 
            time_hour=12
        )
        
        if day_results:
            avg_score = sum(r['Net Score'] for r in day_results) / len(day_results)
            top_sector = day_results[0]
            
            monthly_data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day': current_date.day,
                'Average Score': round(avg_score, 2),
                'Top Sector': top_sector['Sector'],
                'Top Score': top_sector['Net Score'],
                'Market Bias': 'Bullish' if avg_score > 0 else 'Bearish'
            })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(monthly_data)

def generate_intraday_analysis(user_config, selected_symbol):
    """Generate detailed intraday analysis"""
    intraday_data = []
    
    start_datetime = datetime.combine(user_config['date'], user_config['start_time'])
    end_datetime = datetime.combine(user_config['date'], user_config['end_time'])
    
    # Time interval based on resolution
    if user_config['time_resolution'] == "15 minutes":
        interval = 15
    elif user_config['time_resolution'] == "30 minutes":
        interval = 30
    else:
        interval = 60
    
    current_time = start_datetime
    
    while current_time <= end_datetime:
        hour = current_time.hour
        
        # Get sector analysis for this time
        sector_analysis = scoring.calculate_sector_score(selected_symbol, str(user_config['date']), hour)
        score = sector_analysis['score'] * user_config['sensitivity']
        
        # Generate planetary aspect for this time
        transit_score = (hash(str(current_time) + selected_symbol) % 200 - 100) / 50
        combined_score = (score + transit_score) / 2
        
        signal = "BULLISH" if combined_score > 0.5 else "BEARISH" if combined_score < -0.5 else "NEUTRAL"
        
        intraday_data.append({
            'Time': current_time.strftime('%H:%M'),
            'Symbol': selected_symbol,
            'Score': round(combined_score, 2),
            'Signal': signal,
            'Strength': scoring.get_signal_strength(combined_score),
            'Confidence': sector_analysis['confidence'],
            'Entry Signal': 'BUY' if combined_score > 1.5 else 'SELL' if combined_score < -1.5 else 'HOLD',
            'Risk Level': sector_analysis['risk_level']
        })
        
        current_time += timedelta(minutes=interval)
    
    return pd.DataFrame(intraday_data)

# UI Creation Functions
def create_executive_summary(sector_results):
    """Executive dashboard with comprehensive metrics"""
    st.markdown("## 📊 Executive Dashboard")
    
    if not sector_results:
        st.warning("No sector data available.")
        return
    
    # Calculate comprehensive metrics
    scores = [r['Net Score'] for r in sector_results]
    avg_score = sum(scores) / len(scores)
    volatility = np.std(scores)
    
    top_bullish = max(sector_results, key=lambda x: x['Net Score'])
    top_bearish = min(sector_results, key=lambda x: x['Net Score'])
    
    # Count sectors by trend
    bullish_count = len([r for r in sector_results if r['Net Score'] > 0])
    bearish_count = len([r for r in sector_results if r['Net Score'] < 0])
    
    # Display comprehensive metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card bullish-card">
            <h3>{top_bullish['Emoji']} Top Bullish</h3>
            <h2>{top_bullish['Sector']}</h2>
            <p>Score: +{top_bullish['Net Score']:.2f}</p>
            <small>{top_bullish['Signal Strength']} • {top_bullish['Risk Level']} Risk</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card bearish-card">
            <h3>{top_bearish['Emoji']} Top Bearish</h3>
            <h2>{top_bearish['Sector']}</h2>
            <p>Score: {top_bearish['Net Score']:.2f}</p>
            <small>{top_bearish['Signal Strength']} • {top_bearish['Risk Level']} Risk</small>
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
            <small>Overall Market Bias</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card neutral-card">
            <h3>📊 Market Balance</h3>
            <h2>{bullish_count}:{bearish_count}</h2>
            <p>Bullish vs Bearish</p>
            <small>Volatility: {volatility:.2f}</small>
        </div>
        """, unsafe_allow_html=True)

def create_alerts_system(sector_results):
    """Advanced alert system"""
    st.markdown("### 🚨 Market Intelligence Alerts")
    
    alerts = []
    
    for result in sector_results:
        abs_score = abs(result['Net Score'])
        
        if abs_score >= 3.0:
            alert_type = "🚨 CRITICAL"
            alert_class = "alert-danger"
        elif abs_score >= 2.0:
            alert_type = "⚠️ HIGH"
            alert_class = "alert-warning"
        elif abs_score >= 1.5:
            alert_type = "📢 MODERATE"
            alert_class = "alert-success"
        else:
            continue
        
        direction = "BULLISH" if result['Net Score'] > 0 else "BEARISH"
        
        alerts.append({
            'type': alert_type,
            'direction': direction,
            'sector': result['Sector'],
            'score': result['Net Score'],
            'confidence': result['Confidence'],
            'risk': result['Risk Level'],
            'class': alert_class
        })
    
    if alerts:
        for alert in alerts[:6]:  # Show top 6
            st.markdown(f"""
            <div class="alert-box {alert['class']}">
                <strong>{alert['type']} {alert['direction']} SIGNAL</strong><br>
                <strong>Sector:</strong> {alert['sector']}<br>
                <strong>Score:</strong> {alert['score']:.2f} | <strong>Confidence:</strong> {alert['confidence']:.1%} | <strong>Risk:</strong> {alert['risk']}<br>
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

def create_upcoming_transits_display(user_config):
    """Display upcoming planetary transits"""
    st.markdown("### 🔭 Upcoming Planetary Transits")
    
    upcoming = transits.generate_upcoming_transits(user_config['date'], days=7)
    
    if upcoming:
        for transit in upcoming[:10]:  # Show next 10 transits
            impact_class = "alert-danger" if transit['Impact'] == "High" else "alert-warning" if transit['Impact'] == "Medium" else "alert-success"
            
            st.markdown(f"""
            <div class="transit-card">
                <strong>📅 {transit['Date']} at {transit['Time']}</strong><br>
                <strong>Event:</strong> {transit['Event']}<br>
                <strong>Impact:</strong> <span class="{impact_class}">{transit['Impact']}</span> • 
                <strong>Duration:</strong> {transit['Duration']}<br>
                <strong>Score:</strong> {transit['Score']:.2f} • 
                <strong>Affected Sectors:</strong> {transit['Sectors Affected']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major transits detected in the upcoming period.")

def create_performance_chart(sector_results):
    """Create comprehensive performance chart"""
    if not sector_results or not PLOTLY_AVAILABLE:
        return None
    
    df = pd.DataFrame(sector_results)
    
    # Main performance chart
    fig = px.bar(
        df,
        x='Net Score',
        y='Sector',
        orientation='h',
        color='Net Score',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="📊 Sector Performance Ranking",
        hover_data=['Signal Strength', 'Confidence', 'Risk Level'],
        height=600
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_x=0.5
    )
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    
    # Add annotations for extreme values
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

def create_intraday_chart(intraday_df, symbol):
    """Create intraday analysis chart"""
    if not PLOTLY_AVAILABLE or intraday_df.empty:
        return None
    
    fig = go.Figure()
    
    # Main score line
    fig.add_trace(go.Scatter(
        x=intraday_df['Time'],
        y=intraday_df['Score'],
        mode='lines+markers',
        name='Planetary Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Color code by signal
    bullish_df = intraday_df[intraday_df['Signal'] == 'BULLISH']
    bearish_df = intraday_df[intraday_df['Signal'] == 'BEARISH']
    
    if not bullish_df.empty:
        fig.add_trace(go.Scatter(
            x=bullish_df['Time'],
            y=bullish_df['Score'],
            mode='markers',
            name='Bullish Signals',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if not bearish_df.empty:
        fig.add_trace(go.Scatter(
            x=bearish_df['Time'],
            y=bearish_df['Score'],
            mode='markers',
            name='Bearish Signals',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Add horizontal reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=1.5, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Strong Bullish")
    fig.add_hline(y=-1.5, line_dash="dot", line_color="red", opacity=0.5, annotation_text="Strong Bearish")
    
    fig.update_layout(
        title=f"🔮 Intraday Planetary Analysis - {symbol}",
        xaxis_title="Time",
        yaxis_title="Planetary Score",
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# Main Application
def main():
    # Page config
    st.set_page_config(
        page_title="Vedic Market Analytics Pro",
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
    
    # Main analysis
    try:
        with st.spinner("🔮 Computing comprehensive astrological analysis..."):
            sector_results = analyze_sectors_comprehensive(user_config)
            daily_transits = transits.generate_daily_transits(user_config['date'])
        
        # Executive summary
        create_executive_summary(sector_results)
        
        # Alerts
        create_alerts_system(sector_results)
        
        # Upcoming transits
        create_upcoming_transits_display(user_config)
        
        # Main tabs with all features
        tabs = st.tabs([
            "📊 Sector Analysis",
            "📈 Performance Charts", 
            "🪐 Daily Transits",
            "📅 Weekly Outlook",
            "🗓️ Monthly Calendar",
            "⚡ Intraday Workshop",
            "🔭 Transit Predictions",
            "⚙️ Advanced Settings"
        ])
        
        # Tab 1: Sector Analysis
        with tabs[0]:
            st.markdown("### 🏭 Comprehensive Sector Analysis")
            
            if sector_results:
                # Display sector table
                sector_df = pd.DataFrame(sector_results)
                st.dataframe(sector_df, use_container_width=True, height=500)
                
                # Detailed sector view
                st.markdown("### 🎯 Detailed Sector Analysis")
                selected_sector = st.selectbox("Select sector for detailed analysis:", [r['Sector'] for r in sector_results])
                
                if selected_sector:
                    selected_data = next(r for r in sector_results if r['Sector'] == selected_sector)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Net Score", f"{selected_data['Net Score']:.2f}")
                    with col2:
                        st.metric("Trend", selected_data['Trend'])
                    with col3:
                        st.metric("Confidence", f"{selected_data['Confidence']:.1%}")
                    with col4:
                        st.metric("Risk Level", selected_data['Risk Level'])
                    
                    st.markdown(f"**Top Stocks:** {selected_data['Top Stocks']}")
                    st.markdown(f"**Signal Strength:** {selected_data['Signal Strength']}")
            else:
                st.warning("No sector data available.")
        
        # Tab 2: Performance Charts
        with tabs[1]:
            st.markdown("### 📈 Performance Visualizations")
            
            if PLOTLY_AVAILABLE and sector_results:
                chart = create_performance_chart(sector_results)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Additional metrics
                scores = [r['Net Score'] for r in sector_results]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Score", f"{sum(scores)/len(scores):.2f}")
                with col2:
                    st.metric("Highest Score", f"{max(scores):.2f}")
                with col3:
                    st.metric("Lowest Score", f"{min(scores):.2f}")
                with col4:
                    st.metric("Volatility", f"{np.std(scores):.2f}")
                
                # Score distribution
                fig_hist = px.histogram(
                    x=scores,
                    nbins=15,
                    title="Distribution of Sector Scores",
                    labels={'x': 'Score', 'y': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            else:
                if not PLOTLY_AVAILABLE:
                    st.info("📊 Install plotly for advanced charts: `pip install plotly`")
                
                if sector_results:
                    chart_df = pd.DataFrame(sector_results)[['Sector', 'Net Score']]
                    st.bar_chart(chart_df.set_index('Sector'))
        
        # Tab 3: Daily Transits
        with tabs[2]:
            st.markdown("### 🪐 Today's Planetary Transits")
            
            if not daily_transits.empty:
                st.dataframe(daily_transits, use_container_width=True, height=500)
                
                # Transit summary
                st.markdown("### 📊 Transit Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transits", len(daily_transits))
                with col2:
                    bullish_transits = len(daily_transits[daily_transits['Score'] > 0])
                    st.metric("Bullish Transits", bullish_transits)
                with col3:
                    bearish_transits = len(daily_transits[daily_transits['Score'] < 0])
                    st.metric("Bearish Transits", bearish_transits)
                with col4:
                    avg_score = daily_transits['Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}")
                
                # Best and worst transits
                if not daily_transits.empty:
                    best_transit = daily_transits.loc[daily_transits['Score'].idxmax()]
                    worst_transit = daily_transits.loc[daily_transits['Score'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="alert-box alert-success">
                            <h4>🌟 Best Transit Today</h4>
                            <p><strong>Time:</strong> {best_transit['Time']}</p>
                            <p><strong>Aspect:</strong> {best_transit['Planet A']} {best_transit['Aspect']} {best_transit['Planet B']}</p>
                            <p><strong>Score:</strong> +{best_transit['Score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="alert-box alert-danger">
                            <h4>⚠️ Most Challenging Transit</h4>
                            <p><strong>Time:</strong> {worst_transit['Time']}</p>
                            <p><strong>Aspect:</strong> {worst_transit['Planet A']} {worst_transit['Aspect']} {worst_transit['Planet B']}</p>
                            <p><strong>Score:</strong> {worst_transit['Score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No transits generated for selected date.")
        
        # Tab 4: Weekly Outlook
        with tabs[3]:
            st.markdown("### 📅 Weekly Market Outlook")
            
            with st.spinner("Generating weekly analysis..."):
                weekly_df = generate_weekly_analysis(user_config)
            
            if not weekly_df.empty:
                st.dataframe(weekly_df, use_container_width=True)
                
                # Weekly summary
                st.markdown("### 📊 Weekly Summary")
                bullish_days = len(weekly_df[weekly_df['Market Sentiment'] == 'Bullish'])
                avg_volatility = weekly_df['Volatility'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bullish Days", f"{bullish_days}/7")
                with col2:
                    st.metric("Average Volatility", f"{avg_volatility:.2f}")
                with col3:
                    best_day = weekly_df.loc[weekly_df['Bullish Score'].idxmax()]
                    st.metric("Best Day", best_day['Day'])
                
                # Weekly chart
                if PLOTLY_AVAILABLE:
                    fig_weekly = px.line(
                        weekly_df,
                        x='Day',
                        y=['Bullish Score', 'Bearish Score'],
                        title="Weekly Score Trends",
                        markers=True
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.warning("Unable to generate weekly analysis.")
        
        # Tab 5: Monthly Calendar
        with tabs[4]:
            st.markdown("### 🗓️ Monthly Market Calendar")
            
            with st.spinner("Generating monthly analysis..."):
                monthly_df = generate_monthly_analysis(user_config)
            
            if not monthly_df.empty:
                st.dataframe(monthly_df, use_container_width=True, height=400)
                
                # Monthly metrics
                st.markdown("### 📊 Monthly Summary")
                bullish_days = len(monthly_df[monthly_df['Market Bias'] == 'Bullish'])
                avg_score = monthly_df['Average Score'].mean()
                best_day = monthly_df.loc[monthly_df['Top Score'].idxmax()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bullish Days", f"{bullish_days}/{len(monthly_df)}")
                with col2:
                    st.metric("Monthly Average", f"{avg_score:.2f}")
                with col3:
                    st.metric("Best Day", f"{best_day['Day']}")
                
                # Calendar heatmap simulation
                st.markdown("### 📅 Calendar Heatmap")
                if PLOTLY_AVAILABLE:
                    # Create a simple calendar visualization
                    monthly_df['Week'] = (monthly_df['Day'] - 1) // 7
                    monthly_df['Weekday'] = (monthly_df['Day'] - 1) % 7
                    
                    fig_cal = px.scatter(
                        monthly_df,
                        x='Weekday',
                        y='Week',
                        size='Top Score',
                        color='Average Score',
                        hover_data=['Date', 'Top Sector'],
                        title="Monthly Calendar View",
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    fig_cal.update_layout(height=300)
                    st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.warning("Unable to generate monthly analysis.")
        
        # Tab 6: Intraday Workshop
        with tabs[5]:
            st.markdown("### ⚡ Intraday Planetary Workshop")
            
            # Symbol selection
            all_symbols = []
            for category, sector_dict in config.SECTORS.items():
                for sector, stocks in sector_dict.items():
                    all_symbols.extend(stocks)
            
            # Add popular indices
            popular_symbols = ['NIFTY', 'BANKNIFTY', 'GOLD', 'SILVER', 'CRUDE', 'BTC', 'ETH']
            all_symbols.extend(popular_symbols)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_symbol = st.selectbox("📈 Select Symbol", sorted(set(all_symbols)))
            with col2:
                analysis_type = st.selectbox("Analysis Type", ["Real-time", "Backtest", "Forecast"])
            
            if st.button("🔮 Generate Intraday Analysis", type="primary"):
                with st.spinner(f"Analyzing planetary influences for {selected_symbol}..."):
                    intraday_df = generate_intraday_analysis(user_config, selected_symbol)
                
                if not intraday_df.empty:
                    # Summary metrics
                    st.markdown(f"### 📊 {selected_symbol} Intraday Analysis")
                    
                    bullish_signals = len(intraday_df[intraday_df['Signal'] == 'BULLISH'])
                    bearish_signals = len(intraday_df[intraday_df['Signal'] == 'BEARISH'])
                    buy_signals = len(intraday_df[intraday_df['Entry Signal'] == 'BUY'])
                    sell_signals = len(intraday_df[intraday_df['Entry Signal'] == 'SELL'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🟢 Bullish Periods", bullish_signals)
                    with col2:
                        st.metric("🔴 Bearish Periods", bearish_signals)
                    with col3:
                        st.metric("📈 Buy Signals", buy_signals)
                    with col4:
                        st.metric("📉 Sell Signals", sell_signals)
                    
                    # Intraday chart
                    if PLOTLY_AVAILABLE:
                        intraday_chart = create_intraday_chart(intraday_df, selected_symbol)
                        if intraday_chart:
                            st.plotly_chart(intraday_chart, use_container_width=True)
                    
                    # Detailed table
                    st.markdown("### 🔍 Detailed Timing Analysis")
                    st.dataframe(intraday_df, use_container_width=True, height=400)
                    
                    # Best trading windows
                    buy_opportunities = intraday_df[intraday_df['Entry Signal'] == 'BUY']
                    sell_opportunities = intraday_df[intraday_df['Entry Signal'] == 'SELL']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if not buy_opportunities.empty:
                            st.markdown("### 🟢 Buy Opportunities")
                            for _, row in buy_opportunities.iterrows():
                                st.markdown(f"""
                                <div class="alert-box alert-success">
                                    <strong>⏰ {row['Time']}</strong><br>
                                    Score: {row['Score']:.2f} | Confidence: {row['Confidence']:.1%}<br>
                                    Strength: {row['Strength']} | Risk: {row['Risk Level']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No buy opportunities detected.")
                    
                    with col2:
                        if not sell_opportunities.empty:
                            st.markdown("### 🔴 Sell Opportunities")
                            for _, row in sell_opportunities.iterrows():
                                st.markdown(f"""
                                <div class="alert-box alert-danger">
                                    <strong>⏰ {row['Time']}</strong><br>
                                    Score: {row['Score']:.2f} | Confidence: {row['Confidence']:.1%}<br>
                                    Strength: {row['Strength']} | Risk: {row['Risk Level']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No sell opportunities detected.")
                    
                    # Export intraday data
                    if st.button("📥 Export Intraday Analysis"):
                        csv_data = intraday_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download CSV",
                            csv_data,
                            f"{selected_symbol}_intraday_{user_config['date']}.csv",
                            "text/csv"
                        )
                else:
                    st.warning("Unable to generate intraday analysis.")
            else:
                # Default information
                st.markdown("""
                ### 🌟 Welcome to Intraday Planetary Workshop!
                
                **Features:**
                - ⏰ **Minute-by-minute analysis** with customizable time resolution
                - 🎯 **Entry/Exit signals** based on planetary alignments
                - 📊 **Real-time scoring** with confidence levels
                - 🔄 **Multiple analysis modes**: Real-time, Backtest, Forecast
                - 📈 **Interactive charts** with signal visualization
                - 💾 **Export capabilities** for further analysis
                
                **How to Use:**
                1. Select your trading symbol
                2. Choose analysis type
                3. Click "Generate Intraday Analysis"
                4. Review bullish/bearish periods
                5. Use buy/sell signals for timing
                """)
        
        # Tab 7: Transit Predictions
        with tabs[6]:
            st.markdown("### 🔭 Planetary Transit Predictions")
            
            # Prediction period selection
            col1, col2 = st.columns(2)
            with col1:
                prediction_days = st.slider("Prediction Period (days)", 1, 30, 7)
            with col2:
                impact_filter = st.multiselect("Impact Level", ["High", "Medium", "Low"], default=["High", "Medium"])
            
            # Generate predictions
            predictions = transits.generate_upcoming_transits(user_config['date'], days=prediction_days)
            
            if impact_filter:
                filtered_predictions = [p for p in predictions if p['Impact'] in impact_filter]
            else:
                filtered_predictions = predictions
            
            if filtered_predictions:
                st.markdown(f"### 📅 Next {prediction_days} Days - Major Transits")
                
                for i, transit in enumerate(filtered_predictions[:20]):  # Show top 20
                    impact_class = "alert-danger" if transit['Impact'] == "High" else "alert-warning" if transit['Impact'] == "Medium" else "alert-success"
                    
                    st.markdown(f"""
                    <div class="transit-card">
                        <strong>📅 {transit['Date']} at {transit['Time']}</strong><br>
                        <strong>🪐 Event:</strong> {transit['Event']}<br>
                        <strong>📊 Impact:</strong> <span style="font-weight: bold; color: {'red' if transit['Impact'] == 'High' else 'orange' if transit['Impact'] == 'Medium' else 'green'}">{transit['Impact']}</span> • 
                        <strong>⏱️ Duration:</strong> {transit['Duration']}<br>
                        <strong>🎯 Score:</strong> {transit['Score']:.2f} • 
                        <strong>🏭 Sectors:</strong> {transit['Sectors Affected']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction summary
                st.markdown("### 📊 Prediction Summary")
                high_impact = len([t for t in filtered_predictions if t['Impact'] == 'High'])
                bullish_transits = len([t for t in filtered_predictions if t['Score'] > 0])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Impact Events", high_impact)
                with col2:
                    st.metric("Bullish Transits", f"{bullish_transits}/{len(filtered_predictions)}")
                with col3:
                    avg_score = sum(t['Score'] for t in filtered_predictions) / len(filtered_predictions)
                    st.metric("Average Impact", f"{avg_score:.2f}")
            else:
                st.info("No major transits detected for the selected criteria.")
        
        # Tab 8: Advanced Settings
        with tabs[7]:
            st.markdown("### ⚙️ Advanced Configuration & System Status")
            
            # Current settings
            st.markdown("#### 🎛️ Current Configuration")
            config_df = pd.DataFrame([
                {"Parameter": "Analysis Date", "Value": str(user_config['date'])},
                {"Parameter": "Timezone", "Value": user_config['timezone']},
                {"Parameter": "Market Hours", "Value": f"{user_config['start_time']} - {user_config['end_time']}"},
                {"Parameter": "KP Premium", "Value": f"{user_config['kp_premium']:.1f}"},
                {"Parameter": "Sensitivity", "Value": f"{user_config['sensitivity']:.1f}"},
                {"Parameter": "Confidence Threshold", "Value": f"{user_config['confidence_threshold']:.1%}"},
                {"Parameter": "Time Resolution", "Value": user_config['time_resolution']}
            ])
            st.dataframe(config_df, use_container_width=True)
            
            # System information
            st.markdown("#### ℹ️ System Information")
            st.markdown(f"""
            - **🪐 Plotly Charts:** {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available'}
            - **⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **🔢 Version:** 2.0.0 Complete Edition
            - **📊 Sectors Loaded:** {sum(len(s) for category in config.SECTORS.values() for s in category.values())}
            - **🪐 Transit Engine:** Active
            - **📈 Intraday Analysis:** Available
            - **📅 Multi-timeframe:** Available
            """)
            
            # Performance metrics
            if sector_results:
                st.markdown("#### 📊 Analysis Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sectors Analyzed", len(sector_results))
                with col2:
                    bullish_sectors = len([s for s in sector_results if s['Net Score'] > 0])
                    st.metric("Bullish Sectors", f"{bullish_sectors}/{len(sector_results)}")
                with col3:
                    avg_confidence = sum(s['Confidence'] for s in sector_results) / len(sector_results)
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            # Export all data
            st.markdown("#### 📥 Export Complete Analysis")
            if st.button("📊 Export All Data", type="primary"):
                export_data = {
                    'sectors': pd.DataFrame(sector_results),
                    'daily_transits': daily_transits,
                    'configuration': config_df
                }
                
                if user_config['export_format'] == "Excel (.xlsx)":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        for sheet_name, df in export_data.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    st.download_button(
                        "💾 Download Excel Report",
                        buffer.getvalue(),
                        f"vedic_complete_analysis_{user_config['date']}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif user_config['export_format'] == "JSON (.json)":
                    json_data = {
                        'sectors': export_data['sectors'].to_dict('records'),
                        'daily_transits': export_data['daily_transits'].to_dict('records'),
                        'configuration': export_data['configuration'].to_dict('records'),
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        "💾 Download JSON Data",
                        json.dumps(json_data, indent=2),
                        f"vedic_complete_analysis_{user_config['date']}.json",
                        "application/json"
                    )
                
                else:  # CSV
                    csv_data = export_data['sectors'].to_csv(index=False)
                    st.download_button(
                        "💾 Download CSV",
                        csv_data,
                        f"vedic_sectors_analysis_{user_config['date']}.csv",
                        "text/csv"
                    )
            
            # Quick setup
            if not PLOTLY_AVAILABLE:
                st.info("💡 **Quick Setup:** For enhanced charts, run: `pip install plotly`")
            
            # Data management
            st.markdown("#### 🔄 Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh All Data"):
                    st.rerun()
            
            with col2:
                if st.button("📊 Recalculate Analysis"):
                    st.success("Analysis will refresh on next interaction.")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please try refreshing the page or adjusting your settings.")
        st.markdown("**Error Details:**")
        st.code(str(e))

if __name__ == "__main__":
    main()
