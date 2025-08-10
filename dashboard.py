import streamlit as st
import pandas as pd
from datetime import datetime
import random

# ======== MOCK Astro Data Generator (Replace with your real astro logic) ========
def generate_mock_astro_data(date: str, symbols: list):
    rows = []
    for sym in symbols:
        for hour in range(0, 24):
            trend = random.choice(["Bullish", "Bearish", "Neutral"])
            astro_event = random.choice([
                "Moon Conjunct Jupiter", "Sun Trine Saturn", "Venus Opposite Mars",
                "Mercury Sextile Venus", "Mars Square Neptune"
            ])
            remark = "Bullish" if trend == "Bullish" else "Bearish" if trend == "Bearish" else "Neutral"
            rows.append({
                "DateTime": f"{date} {hour:02d}:00",
                "Symbol": sym,
                "Trend": trend,
                "AstroEvent": astro_event,
                "Remark": remark
            })
    return pd.DataFrame(rows)

# ======== Load Watchlists ========
def load_watchlist(file_path):
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

watchlists = {
    "EYE FUTURE WATCHLIST": load_watchlist("/mnt/data/Eye_d16ec.txt"),
    "WATCHLIST 2": load_watchlist("/mnt/data/Watchlist (2)_8e9c8.txt"),
    "FUTURE LIST": load_watchlist("/mnt/data/FUTURE_e8298.txt")
}

# ======== UI Layout ========
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("📅 Today Market — Astro Timeline")

# Month & Date Selector
months = list(range(1, 13))
selected_month = st.selectbox("Select Month", months, index=datetime.now().month - 1)

dates_in_month = list(range(1, 32))
selected_day = st.selectbox("Select Date", dates_in_month, index=datetime.now().day - 1)

selected_date = f"2025-{selected_month:02d}-{selected_day:02d}"

# Watchlist Selector
watchlist_name = st.selectbox("Select Watchlist", list(watchlists.keys()))
symbols_list = watchlists[watchlist_name]

if not symbols_list:
    st.error(f"No symbols found in {watchlist_name}. Please upload a valid watchlist.")
    st.stop()

# Trend Filter
analysis_type = st.radio("Filter Events", ["All", "Bullish", "Bearish"], horizontal=True)

# ======== Generate / Load Data ========
df = generate_mock_astro_data(selected_date, symbols_list)

# Apply Filter
if analysis_type != "All":
    df = df[df["Trend"] == analysis_type]

if df.empty:
    st.warning("No astro events found for this selection.")
    st.stop()

# ======== Summary Cards ========
col1, col2, col3 = st.columns(3)
bullish_count = len(df[df["Trend"] == "Bullish"])
bearish_count = len(df[df["Trend"] == "Bearish"])
neutral_count = len(df[df["Trend"] == "Neutral"])

col1.metric("📈 Bullish", bullish_count)
col2.metric("📉 Bearish", bearish_count)
col3.metric("⚖ Neutral", neutral_count)

# ======== Detailed Table ========
st.subheader(f"Astro Events for {selected_date} — {watchlist_name}")
st.dataframe(df, use_container_width=True)
