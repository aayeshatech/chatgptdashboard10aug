import streamlit as st
import pandas as pd
from datetime import datetime
import emoji

# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="Market & Astro Screener", layout="wide")

# --------------------
# CSS for Card Styling
# --------------------
st.markdown("""
    <style>
    .card {
        border-radius: 12px;
        padding: 16px;
        margin: 8px;
        color: white;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
    }
    .bullish {
        background-color: #1E8449;
    }
    .bearish {
        background-color: #922B21;
    }
    .symbol {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .detail {
        font-size: 14px;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------
# Mock Data Loader (Replace with Astro Calculation)
# --------------------
def load_mock_data():
    data = [
        ["NIFTY", "Bullish", 85, "09:15", "Saturn – Structure", "Mars Energy Peak", "2h"],
        ["BANKNIFTY", "Bearish", 70, "09:15", "Venus – Value", "Venus Retrograde", "3h"],
        ["RELIANCE", "Bullish", 78, "10:30", "Jupiter – Expansion", "Moon Conj Jupiter", "1h"],
        ["INFY", "Bearish", 65, "10:30", "Mercury – Communication", "Mercury Square Saturn", "5h"],
        ["TCS", "Bullish", 90, "12:00", "Sun – Vitality", "Sun Sextile Venus", "4h"]
    ]
    return pd.DataFrame(data, columns=["Symbol", "Impact", "Strength", "Best Time", "Planetary Support", "Transit", "Next Change"])

# --------------------
# Group and Display as Cards
# --------------------
def display_cards(df, sentiment):
    df = df[df["Impact"] == sentiment].copy()
    df = df.sort_values(by=["Best Time", "Strength"], ascending=[True, False])

    if df.empty:
        st.info(f"No {sentiment} events found.")
        return

    grouped = df.groupby("Best Time")
    for time, group in grouped:
        st.markdown(f"### 🕒 {time}")
        cols = st.columns(3)
        for idx, row in group.reset_index(drop=True).iterrows():
            with cols[idx % 3]:
                color_class = "bullish" if sentiment == "Bullish" else "bearish"
                st.markdown(f"""
                    <div class="card {color_class}">
                        <div class="symbol">{row['Symbol']}</div>
                        <div class="detail"><b>Sentiment:</b> {emoji.emojize(':chart_increasing:') if sentiment == 'Bullish' else emoji.emojize(':chart_decreasing:')} {sentiment}</div>
                        <div class="detail"><b>Strength:</b> {row['Strength']}%</div>
                        <div class="detail"><b>Planetary Support:</b> {row['Planetary Support']}</div>
                        <div class="detail"><b>Transit:</b> {row['Transit']}</div>
                        <div class="detail"><b>Next Change:</b> {row['Next Change']}</div>
                    </div>
                """, unsafe_allow_html=True)

# --------------------
# UI Layout
# --------------------
st.title("📊 Market & Astro Screener")

period_type = st.selectbox("Select Period", ["Today", "Week", "Month"])
selected_date = st.date_input("Select Date", datetime.today())

# Load Data
df = load_mock_data()

# Display Tabs
tab1, tab2 = st.tabs(["🟢 Bullish", "🔴 Bearish"])
with tab1:
    display_cards(df, "Bullish")
with tab2:
    display_cards(df, "Bearish")
