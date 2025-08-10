import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

DATA_PATH = "data"

st.set_page_config(page_title="Astro Market Dashboard", layout="wide")

# ---------- Load CSV ----------
def load_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        st.error(f"Missing file: {filename}")
        return pd.DataFrame()
    return pd.read_csv(filepath)

# ---------- Astro helpers ----------
def extract_astro_details(events_str):
    parts = events_str.split(", ")
    details = {}
    for p in parts:
        if ":" in p:
            planet, val = p.split(":")
            details[planet] = val
    return details

# ---------- Today Market ----------
def today_market():
    st.title("📅 Today Market with Astro Details")

    # User Inputs
    date_selected = st.date_input("Select Date", datetime.now().date(),
                                   min_value=datetime(2024, 8, 10).date(),
                                   max_value=datetime(2032, 12, 31).date())
    year = date_selected.year

    watchlists = {
        "EYE FUTURE WATCHLIST": ["TATASTEEL", "RELIANCE"],
        "BANKING WATCHLIST": ["HDFCBANK", "ICICIBANK"],
        "TECH WATCHLIST": ["INFY", "TCS"]
    }
    selected_watchlist = st.selectbox("Select Watchlist", list(watchlists.keys()))
    selected_symbols = watchlists[selected_watchlist]

    trend_filter = st.radio("Filter Trend", ["All", "🟢 Bullish", "🔴 Bearish"])

    # Load hourly CSV for that year
    df = load_csv(f"ephemeris_hourly_{year}.csv")
    if df.empty:
        return

    # Filter by date & watchlist
    df = df[df["DateTime"].str.startswith(str(date_selected))]
    df = df[df["Symbol"].isin(selected_symbols)]

    # Apply trend filter
    if trend_filter != "All":
        df = df[df["Trend"] == trend_filter]

    # Extract Astro details
    astro_details = df["Events"].apply(extract_astro_details)
    astro_df = pd.DataFrame(list(astro_details))
    df = pd.concat([df.reset_index(drop=True), astro_df.reset_index(drop=True)], axis=1)

    st.subheader(f"Market Data for {date_selected} — {selected_watchlist}")
    st.dataframe(df)

    # Timeline chart
    if not df.empty:
        fig = px.scatter(df, x="DateTime", y="Symbol", color="Trend",
                         hover_data=["Events"], title="Intraday Trend Timeline")
        st.plotly_chart(fig, use_container_width=True)

        # Symbol selector for detail view
        selected_symbol = st.selectbox("Select Symbol for Timeline", df["Symbol"].unique())
        df_symbol = df[df["Symbol"] == selected_symbol]

        fig2 = px.line(df_symbol, x="DateTime", y=df_symbol.index,
                       color="Trend", hover_data=["Events"],
                       title=f"Trend Timeline for {selected_symbol}")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Main ----------
tabs = ["Today Market", "Watchlist", "Upcoming Transit", "Intraday"]
page = st.sidebar.radio("Navigation", tabs)

if page == "Today Market":
    today_market()

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

elif page == "Upcoming Transit":
    st.title("🔮 Upcoming Transits")
    year = st.selectbox("Select Year", list(range(2024, 2033)), index=0)
    df = load_csv(f"ephemeris_daily_{year}.csv")
    if not df.empty:
        st.dataframe(df)

elif page == "Intraday":
    st.title("⏳ Intraday Astro Timeline")
    symbol = st.text_input("Enter Symbol", "TATASTEEL")
    year = datetime.now().year
    df = load_csv(f"ephemeris_hourly_{year}.csv")
    if not df.empty:
        df_symbol = df[df["Symbol"].str.upper() == symbol.upper()]
        st.dataframe(df_symbol)
