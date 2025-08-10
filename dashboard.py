import streamlit as st
import pandas as pd
import os
from datetime import datetime

DATA_PATH = "data"

st.set_page_config(page_title="Astro Market Dashboard", layout="wide")

# Tabs
tabs = ["Today Market", "Watchlist", "Upcoming Transit", "Intraday"]
page = st.sidebar.radio("Navigation", tabs)

def load_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        st.error(f"Missing file: {filename}")
        return pd.DataFrame()
    return pd.read_csv(filepath)

# === TODAY MARKET TAB ===
if page == "Today Market":
    st.title("📅 Today Market")
    today = datetime.now()
    year = today.year
    df = load_csv(f"ephemeris_hourly_{year}.csv")
    if not df.empty:
        df_today = df[df["DateTime"].str.startswith(today.strftime("%Y-%m-%d"))]
        st.dataframe(df_today)

# === WATCHLIST TAB ===
elif page == "Watchlist":
    st.title("👁 Watchlist Trends")
    year = st.selectbox("Select Year", list(range(2024, 2033)), index=0)
    df = load_csv(f"summary_daily_{year}.csv")
    if not df.empty:
        bullish = df[df["Trend"] == "🟢 Bullish"]
        bearish = df[df["Trend"] == "🔴 Bearish"]
        st.subheader("Bullish Symbols")
        st.dataframe(bullish)
        st.subheader("Bearish Symbols")
        st.dataframe(bearish)

# === UPCOMING TRANSIT TAB ===
elif page == "Upcoming Transit":
    st.title("🔮 Upcoming Transits")
    year = st.selectbox("Select Year", list(range(2024, 2033)), index=0)
    df = load_csv(f"ephemeris_daily_{year}.csv")
    if not df.empty:
        st.dataframe(df)

# === INTRADAY TAB ===
elif page == "Intraday":
    st.title("⏳ Intraday Astro Timeline")
    symbol = st.text_input("Enter Symbol", "TATASTEEL")
    year = datetime.now().year
    df = load_csv(f"ephemeris_hourly_{year}.csv")
    if not df.empty:
        df_symbol = df[df["Symbol"].str.upper() == symbol.upper()]
        st.dataframe(df_symbol)
