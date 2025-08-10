import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

# --------------------------------
# Mock Data Generator
# --------------------------------
def generate_mock_data(start_year=2025, end_year=2030):
    symbols = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NFLX"]
    transits = ["Moon Conjunct Sun", "Mars Trine Jupiter", "Venus Square Saturn", "Mercury Opposite Neptune"]
    impacts = ["Bullish", "Bearish"]
    planets = ["Sun", "Moon", "Mars", "Venus", "Jupiter", "Saturn"]

    data = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for _ in range(10):  # events per month
                date = datetime(year, month, random.randint(1, 28))
                start_time = datetime(year, month, date.day, random.randint(0, 23), 0)
                end_time = start_time + timedelta(hours=random.randint(1, 3))
                sym = random.choice(symbols)
                data.append({
                    "Date": date.date(),
                    "Start Time": start_time.strftime("%H:%M"),
                    "End Time": end_time.strftime("%H:%M"),
                    "Symbol": sym,
                    "Transit": random.choice(transits),
                    "Planet": random.choice(planets),
                    "Impact": random.choice(impacts),
                    "Strength": random.randint(60, 100)
                })
    return pd.DataFrame(data)

mock_df = generate_mock_data()

# --------------------------------
# Utility: Display Cards
# --------------------------------
def display_cards(df):
    for _, row in df.iterrows():
        color = "lightgreen" if row["Impact"] == "Bullish" else "#ff9999"
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px'>
            <b>{row['Date']} {row['Start Time']} - {row['End Time']}</b><br>
            <b>Symbol:</b> {row['Symbol']}<br>
            <b>Transit:</b> {row['Transit']} ({row['Planet']})<br>
            <b>Impact:</b> {row['Impact']}<br>
            <b>Strength:</b> {row['Strength']}%
            </div>
            """, unsafe_allow_html=True
        )

# --------------------------------
# Sidebar: Upload Watchlists
# --------------------------------
st.sidebar.header("Upload Watchlists")
wl1 = st.sidebar.file_uploader("Watchlist 1", type=["txt"])
wl2 = st.sidebar.file_uploader("Watchlist 2", type=["txt"])
wl3 = st.sidebar.file_uploader("Watchlist 3", type=["txt"])

watchlist_symbols = set()
for wl in [wl1, wl2, wl3]:
    if wl:
        watchlist_symbols.update(pd.read_csv(wl, header=None)[0].tolist())

# --------------------------------
# Tabs
# --------------------------------
tab1, tab2, tab3 = st.tabs(["📅 Today Market", "📊 Screener", "🌌 Upcoming Transit"])

# --------------------------------
# Tab 1: Today Market
# --------------------------------
with tab1:
    st.subheader("Today Market – Bullish & Bearish Symbols")
    year = st.selectbox("Select Year", sorted(mock_df["Date"].apply(lambda x: x.year).unique()))
    month = st.selectbox("Select Month", range(1, 13))
    date = st.selectbox("Select Date", sorted(mock_df[mock_df["Date"].apply(lambda x: x.year == year) &
                                                     (mock_df["Date"].apply(lambda x: x.month) == month)]["Date"].unique()))
    sentiment = st.radio("Select Sentiment", ["Bullish", "Bearish"])

    filtered_df = mock_df[(mock_df["Date"] == date) & (mock_df["Impact"] == sentiment)]
    if watchlist_symbols:
        filtered_df = filtered_df[filtered_df["Symbol"].isin(watchlist_symbols)]

    if filtered_df.empty:
        st.info("No data found for selection.")
    else:
        display_cards(filtered_df)

# --------------------------------
# Tab 2: Screener
# --------------------------------
with tab2:
    st.subheader("Screener – Weekly or Monthly Astro Events")
    mode = st.radio("View By", ["Week", "Month"])
    year = st.selectbox("Year", sorted(mock_df["Date"].apply(lambda x: x.year).unique()), key="scr_year")
    month = st.selectbox("Month", range(1, 13), key="scr_month")

    if mode == "Week":
        start_date = st.date_input("Select Start Date")
        end_date = start_date + timedelta(days=7)
        scr_df = mock_df[(mock_df["Date"] >= start_date) & (mock_df["Date"] <= end_date)]
    else:
        scr_df = mock_df[(mock_df["Date"].apply(lambda x: x.year) == year) &
                         (mock_df["Date"].apply(lambda x: x.month) == month)]

    if watchlist_symbols:
        scr_df = scr_df[scr_df["Symbol"].isin(watchlist_symbols)]

    if scr_df.empty:
        st.info("No events found.")
    else:
        display_cards(scr_df)

# --------------------------------
# Tab 3: Upcoming Transit
# --------------------------------
with tab3:
    st.subheader("Upcoming Transit Events")
    year = st.selectbox("Year", sorted(mock_df["Date"].apply(lambda x: x.year).unique()), key="up_year")
    month = st.selectbox("Month", range(1, 13), key="up_month")
    sentiment = st.radio("Sentiment", ["Bullish", "Bearish"], key="up_sentiment")

    up_df = mock_df[(mock_df["Date"].apply(lambda x: x.year) == year) &
                    (mock_df["Date"].apply(lambda x: x.month) == month) &
                    (mock_df["Impact"] == sentiment)]

    if watchlist_symbols:
        up_df = up_df[up_df["Symbol"].isin(watchlist_symbols)]

    if up_df.empty:
        st.info("No transit events found.")
    else:
        display_cards(up_df)
