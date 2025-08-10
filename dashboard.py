import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# ================================
# CONFIG
# ================================
DATA_PATH = "data"  # Folder containing pre-generated CSVs
WATCHLIST = ["TATASTEEL", "RELIANCE", "INFY", "HDFCBANK"]

# ================================
# LOAD FUNCTIONS
# ================================
@st.cache_data
def load_ephemeris(date):
    """Load ephemeris CSV for the year of the given date."""
    year = pd.to_datetime(date).year
    file_path = os.path.join(DATA_PATH, f"ephemeris_{year}.csv")
    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_path}. Please upload it to run this dashboard.")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df

@st.cache_data
def load_summary(year):
    """Load summary CSV for the given year."""
    file_path = os.path.join(DATA_PATH, f"summary_{year}.csv")
    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_path}. Please upload it to run this dashboard.")
        return pd.DataFrame()
    return pd.read_csv(file_path)

# ================================
# STREAMLIT DASHBOARD
# ================================
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("🔮 Astro Market Dashboard (CSV-Only Cloud Version)")

tabs = st.tabs(["📅 Today Market", "⏳ Intraday Astro Analysis", "📋 Watchlist", "🌙 Upcoming Transit"])

# ================================
# TAB 1: TODAY MARKET
# ================================
with tabs[0]:
    st.subheader("📅 Today Market Trends & Events")
    date = st.date_input("Select Date", datetime(2025, 8, 10),
                         min_value=datetime(2025, 1, 1),
                         max_value=datetime(2030, 12, 31))
    symbols = st.multiselect("Select Symbols", WATCHLIST, default=WATCHLIST)

    df = load_ephemeris(date)
    if not df.empty:
        today_df = df[df["DateTime"].dt.date == pd.to_datetime(date).date()]
        st.dataframe(today_df[today_df["Symbol"].isin(symbols)][["DateTime", "Symbol", "Trend", "Events"]])

# ================================
# TAB 2: INTRADAY ANALYSIS
# ================================
with tabs[1]:
    st.subheader("⏳ Intraday Astro Analysis")
    date = st.date_input("Select Date for Intraday", datetime(2025, 8, 10),
                         min_value=datetime(2025, 1, 1),
                         max_value=datetime(2030, 12, 31), key="intraday_date")
    symbol = st.selectbox("Select Symbol", WATCHLIST)
    start_time = st.time_input("Start Time", datetime.strptime("09:00", "%H:%M").time())
    end_time = st.time_input("End Time", datetime.strptime("15:30", "%H:%M").time())

    df = load_ephemeris(date)
    if not df.empty:
        intraday_df = df[(df["Symbol"] == symbol) & (df["DateTime"].dt.date == pd.to_datetime(date).date())]
        intraday_df = intraday_df[(intraday_df["DateTime"].dt.time >= start_time) &
                                  (intraday_df["DateTime"].dt.time <= end_time)]

        st.dataframe(intraday_df[["DateTime", "Symbol", "Trend", "Events"]])

        if not intraday_df.empty:
            fig = px.timeline(intraday_df, x_start="DateTime", x_end="DateTime",
                              y="Symbol", color="Trend",
                              color_discrete_map={"🟢 Bullish": "green", "🔴 Bearish": "red", "⚪ Neutral": "gray"},
                              hover_data=["Events"])
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)

# ================================
# TAB 3: WATCHLIST
# ================================
with tabs[2]:
    st.subheader("📋 Watchlist Monthly Summary")
    date = st.date_input("Select Date for Watchlist", datetime(2025, 8, 10),
                         min_value=datetime(2025, 1, 1),
                         max_value=datetime(2030, 12, 31), key="watchlist_date")
    symbols = st.multiselect("Select Symbols for Watchlist", WATCHLIST, default=WATCHLIST)

    summary_df = load_summary(pd.to_datetime(date).year)
    if not summary_df.empty:
        st.dataframe(summary_df[summary_df["Symbol"].isin(symbols)])

# ================================
# TAB 4: UPCOMING TRANSIT
# ================================
with tabs[3]:
    st.subheader("🌙 Upcoming Transit Events & Trend Changes")
    date = st.date_input("Select Date for Upcoming Transit", datetime(2025, 8, 10),
                         min_value=datetime(2025, 1, 1),
                         max_value=datetime(2030, 12, 31), key="upcoming_date")
    symbols = st.multiselect("Select Symbols for Transit", WATCHLIST, default=WATCHLIST)
    trend_filter = st.selectbox("Filter by Trend", ["All", "🟢 Bullish", "🔴 Bearish", "⚪ Neutral"])

    df = load_ephemeris(date)
    if not df.empty:
        future_df = df[df["DateTime"] > pd.to_datetime(date)]
        future_df = future_df[future_df["Symbol"].isin(symbols)]
        if trend_filter != "All":
            future_df = future_df[future_df["Trend"] == trend_filter]

        result = []
        for sym in symbols:
            sym_data = future_df[future_df["Symbol"] == sym]
            if not sym_data.empty:
                next_event = sym_data.iloc[0]
                result.append({
                    "Symbol": sym,
                    "NextDateTime": next_event["DateTime"],
                    "Trend": next_event["Trend"],
                    "Event": next_event["Events"]
                })
        if result:
            st.dataframe(pd.DataFrame(result))
