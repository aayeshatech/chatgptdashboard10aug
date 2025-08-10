import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.express as px
import swisseph as swe
import numpy as np

# ================================
# CONFIG
# ================================
DATA_PATH = "data"
TIMEZONE_OFFSET = 5.5
AYANAMSHA = swe.SIDM_LAHIRI
START_YEAR = 2025
END_YEAR = 2030

WATCHLIST = ["TATASTEEL", "RELIANCE", "INFY", "HDFCBANK"]

SYMBOL_RULERS = {
    "TATASTEEL": ["Mars", "Saturn"],
    "RELIANCE": ["Jupiter", "Sun"],
    "INFY": ["Mercury", "Moon"],
    "HDFCBANK": ["Venus", "Jupiter"]
}

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
    "Rahu": swe.MEAN_NODE,
    "Ketu": swe.MEAN_NODE
}

NAKSHATRAS = [
    ("Ashwini", "Ketu"), ("Bharani", "Venus"), ("Krittika", "Sun"), ("Rohini", "Moon"), ("Mrigashirsha", "Mars"),
    ("Ardra", "Rahu"), ("Punarvasu", "Jupiter"), ("Pushya", "Saturn"), ("Ashlesha", "Mercury"), ("Magha", "Ketu"),
    ("Purva Phalguni", "Venus"), ("Uttara Phalguni", "Sun"), ("Hasta", "Moon"), ("Chitra", "Mars"),
    ("Swati", "Rahu"), ("Vishakha", "Jupiter"), ("Anuradha", "Saturn"), ("Jyeshtha", "Mercury"),
    ("Mula", "Ketu"), ("Purva Ashadha", "Venus"), ("Uttara Ashadha", "Sun"), ("Shravana", "Moon"),
    ("Dhanishta", "Mars"), ("Shatabhisha", "Rahu"), ("Purva Bhadrapada", "Jupiter"),
    ("Uttara Bhadrapada", "Saturn"), ("Revati", "Mercury")
]

# ================================
# ASTRO HELPER FUNCTIONS
# ================================
def get_rashi(longitude):
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    return signs[int(longitude // 30)]

def get_nakshatra(longitude):
    total_deg = longitude % 360
    nak_index = int(total_deg // (13 + 1/3))
    pada = int(((total_deg % (13 + 1/3)) // (3 + 1/3)) + 1)
    nakshatra, lord = NAKSHATRAS[nak_index]
    return nakshatra, pada, lord

def bullish_bearish_for_symbol(ruling_planets, planet_positions):
    strength_count = 0
    weak_count = 0
    for rp in ruling_planets:
        if rp in planet_positions:
            rashi, retro = planet_positions[rp]
            if rashi in ["Taurus", "Cancer", "Leo", "Sagittarius", "Pisces"] and retro == "D":
                strength_count += 1
            else:
                weak_count += 1
    if strength_count > weak_count:
        return "🟢 Bullish"
    elif weak_count > strength_count:
        return "🔴 Bearish"
    else:
        return "⚪ Neutral"

def detect_aspect(lon1, lon2):
    diff = abs(lon1 - lon2) % 360
    aspects = {0: "Conjunction", 60: "Sextile", 90: "Square", 120: "Trine", 180: "Opposition"}
    orb = 3
    for angle, name in aspects.items():
        if abs(diff - angle) <= orb:
            return name
    return None

# ================================
# EPHEMERIS GENERATOR
# ================================
def generate_ephemeris_for_year(year):
    swe.set_sid_mode(AYANAMSHA)
    start_dt = datetime(year, 1, 1, 0, 0)
    end_dt = datetime(year, 12, 31, 23, 0)

    hourly_rows = []
    monthly_summary = {symbol: [] for symbol in SYMBOL_RULERS}

    last_moon_nak = None
    last_moon_sign = None

    dt_cursor = start_dt
    while dt_cursor <= end_dt:
        jd = swe.julday(dt_cursor.year, dt_cursor.month, dt_cursor.day,
                        dt_cursor.hour - TIMEZONE_OFFSET)
        planet_positions = {}
        event_list = []

        for planet_name, planet_code in PLANETS.items():
            lon, lat, dist, lon_speed = swe.calc_ut(jd, planet_code)[:4]
            if planet_name == "Ketu":
                lon = (lon + 180) % 360
            rashi = get_rashi(lon)
            nak, pada, lord = get_nakshatra(lon)
            retro = "R" if lon_speed < 0 else "D"
            planet_positions[planet_name] = (rashi, retro, lon)

            if planet_name == "Moon":
                if last_moon_nak and nak != last_moon_nak:
                    event_list.append(f"Moon enters {nak} Nakshatra")
                if last_moon_sign and rashi != last_moon_sign:
                    event_list.append(f"Moon enters {rashi} Sign")
                last_moon_nak = nak
                last_moon_sign = rashi

        for p1, (rashi1, retro1, lon1) in planet_positions.items():
            for p2, (rashi2, retro2, lon2) in planet_positions.items():
                if p1 != p2:
                    aspect = detect_aspect(lon1, lon2)
                    if aspect:
                        event_list.append(f"{p1} {aspect} {p2}")

        for symbol, rulers in SYMBOL_RULERS.items():
            trend = bullish_bearish_for_symbol(rulers, planet_positions)
            hourly_rows.append({
                "DateTime": dt_cursor.strftime("%Y-%m-%d %H:%M"),
                "Symbol": symbol,
                "Trend": trend,
                "Events": "; ".join(event_list)
            })
            monthly_summary[symbol].append(trend)

        dt_cursor += timedelta(hours=1)

    os.makedirs(DATA_PATH, exist_ok=True)
    pd.DataFrame(hourly_rows).to_csv(os.path.join(DATA_PATH, f"ephemeris_{year}.csv"), index=False)
    summary_rows = []
    for symbol, trends in monthly_summary.items():
        main_trend = max(set(trends), key=trends.count)
        summary_rows.append({"Symbol": symbol, "Year": year, "MainTrend": main_trend})
    pd.DataFrame(summary_rows).to_csv(os.path.join(DATA_PATH, f"summary_{year}.csv"), index=False)

# ================================
# DATA LOADER (AUTO GENERATES IF MISSING)
# ================================
@st.cache_data
def load_ephemeris(date):
    year = pd.to_datetime(date).year
    file_path = os.path.join(DATA_PATH, f"ephemeris_{year}.csv")
    if not os.path.exists(file_path):
        with st.spinner(f"Generating ephemeris for {year}..."):
            generate_ephemeris_for_year(year)
    df = pd.read_csv(file_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df

@st.cache_data
def load_summary(year):
    file_path = os.path.join(DATA_PATH, f"summary_{year}.csv")
    if not os.path.exists(file_path):
        with st.spinner(f"Generating summary for {year}..."):
            generate_ephemeris_for_year(year)
    return pd.read_csv(file_path)

# ================================
# STREAMLIT DASHBOARD
# ================================
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("🔮 Astro Market Dashboard (Auto-Generating Data)")

tabs = st.tabs(["📅 Today Market", "⏳ Intraday Astro Analysis", "📋 Watchlist", "🌙 Upcoming Transit"])

# Today Market
with tabs[0]:
    date = st.date_input("Select Date", datetime(2025, 8, 10))
    symbols = st.multiselect("Select Symbols", WATCHLIST, default=WATCHLIST)
    df = load_ephemeris(date)
    today_df = df[df["DateTime"].dt.date == pd.to_datetime(date).date()]
    st.dataframe(today_df[today_df["Symbol"].isin(symbols)][["DateTime", "Symbol", "Trend", "Events"]])

# Intraday
with tabs[1]:
    date = st.date_input("Intraday Date", datetime(2025, 8, 10), key="intraday")
    symbol = st.selectbox("Symbol", WATCHLIST)
    start_time = st.time_input("Start Time", datetime.strptime("09:00", "%H:%M").time())
    end_time = st.time_input("End Time", datetime.strptime("15:30", "%H:%M").time())
    df = load_ephemeris(date)
    intraday_df = df[(df["Symbol"] == symbol) & (df["DateTime"].dt.date == pd.to_datetime(date).date())]
    intraday_df = intraday_df[(intraday_df["DateTime"].dt.time >= start_time) & (intraday_df["DateTime"].dt.time <= end_time)]
    st.dataframe(intraday_df[["DateTime", "Symbol", "Trend", "Events"]])
    fig = px.timeline(intraday_df, x_start="DateTime", x_end="DateTime", y="Symbol", color="Trend",
                      color_discrete_map={"🟢 Bullish": "green", "🔴 Bearish": "red", "⚪ Neutral": "gray"},
                      hover_data=["Events"])
    st.plotly_chart(fig, use_container_width=True)

# Watchlist
with tabs[2]:
    date = st.date_input("Watchlist Date", datetime(2025, 8, 10), key="watchlist")
    symbols = st.multiselect("Symbols", WATCHLIST, default=WATCHLIST)
    summary_df = load_summary(pd.to_datetime(date).year)
    st.dataframe(summary_df[summary_df["Symbol"].isin(symbols)])

# Upcoming Transit
with tabs[3]:
    date = st.date_input("Transit Date", datetime(2025, 8, 10), key="transit")
    symbols = st.multiselect("Transit Symbols", WATCHLIST, default=WATCHLIST)
    trend_filter = st.selectbox("Trend Filter", ["All", "🟢 Bullish", "🔴 Bearish", "⚪ Neutral"])
    df = load_ephemeris(date)
    future_df = df[df["DateTime"] > pd.to_datetime(date)]
    if trend_filter != "All":
        future_df = future_df[future_df["Trend"] == trend_filter]
    result = []
    for sym in symbols:
        sym_data = future_df[future_df["Symbol"] == sym]
        if not sym_data.empty:
            next_event = sym_data.iloc[0]
            result.append({"Symbol": sym, "NextDateTime": next_event["DateTime"], "Trend": next_event["Trend"], "Event": next_event["Events"]})
    st.dataframe(pd.DataFrame(result))
