import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

DATA_PATH = "data"

st.set_page_config(page_title="Astro Market Dashboard", layout="wide")

# ----------- Load Watchlists from Uploaded Files -----------
def load_watchlist(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

WATCHLISTS = {
    "EYE FUTURE WATCHLIST": load_watchlist("Eye_d16ec.txt"),
    "WATCHLIST (2)": load_watchlist("Watchlist (2)_8e9c8.txt"),
}

# ----------- Load CSV Data -----------
def load_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        st.error(f"Missing file: {filename}")
        return pd.DataFrame()
    return pd.read_csv(filepath)

# ----------- Event Remark Calculation -----------
def classify_event(event_str):
    event_str_lower = event_str.lower()
    bullish_aspects = ["trine", "sextile", "conjunct jupiter", "moon in taurus", "moon in cancer"]
    bearish_aspects = ["square", "opposition", "conjunct saturn", "moon in scorpio", "moon in capricorn"]

    if any(term in event_str_lower for term in bullish_aspects):
        return "Bullish"
    elif any(term in event_str_lower for term in bearish_aspects):
        return "Bearish"
    else:
        return "Neutral"

# ----------- Today Market Tab -----------
def today_market():
    st.title("📅 Today Market — Astro Timeline")

    # Inputs
    date_selected = st.date_input("Select Date", datetime.now().date(),
                                   min_value=datetime(2025, 1, 1).date(),
                                   max_value=datetime(2032, 12, 31).date())
    year = date_selected.year

    selected_watchlist = st.selectbox("Select Watchlist", list(WATCHLISTS.keys()))
    selected_symbols = WATCHLISTS[selected_watchlist]

    trend_filter = st.radio("Filter Events", ["All", "Bullish", "Bearish"])

    # Load hourly data for year
    df = load_csv(f"ephemeris_hourly_{year}.csv")
    if df.empty:
        return

    # Filter for date and watchlist symbols
    df = df[df["DateTime"].str.startswith(str(date_selected))]
    df = df[df["Symbol"].isin(selected_symbols)]

    # Replace Events column with only astro transit names
    df["AstroEvent"] = df["Events"].apply(lambda x: x.split(",")[0] if "," in x else x)

    # Classify bullish/bearish
    df["Remark"] = df["AstroEvent"].apply(classify_event)

    # Apply filter
    if trend_filter != "All":
        df = df[df["Remark"] == trend_filter]

    # Highlight bullish rows in blue
    def highlight_bullish(row):
        color = 'background-color: lightblue' if row["Remark"] == "Bullish" else ''
        return [color] * len(row)

    st.subheader(f"Astro Events for {date_selected} — {selected_watchlist}")
    st.dataframe(df.style.apply(highlight_bullish, axis=1))

    # Timeline Chart
    if not df.empty:
        fig = px.scatter(df, x="DateTime", y="Symbol", color="Remark",
                         hover_data=["AstroEvent"],
                         title="Intraday Astro Event Impact Timeline",
                         color_discrete_map={"Bullish": "blue", "Bearish": "red", "Neutral": "gray"})
        st.plotly_chart(fig, use_container_width=True)

        # Symbol-specific view
        selected_symbol = st.selectbox("Select Symbol for Detailed Timeline", df["Symbol"].unique())
        df_symbol = df[df["Symbol"] == selected_symbol]
        fig2 = px.line(df_symbol, x="DateTime", y=df_symbol.index, color="Remark",
                       hover_data=["AstroEvent"],
                       title=f"Astro Event Timeline — {selected_symbol}",
                       color_discrete_map={"Bullish": "blue", "Bearish": "red", "Neutral": "gray"})
        st.plotly_chart(fig2, use_container_width=True)

# ----------- Main Tabs -----------
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
