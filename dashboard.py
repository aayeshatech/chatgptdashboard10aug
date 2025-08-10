import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time

# ===============================
# Load Watchlists from Uploaded Files
# ===============================
def load_watchlist(file_path):
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        return symbols
    except:
        return []

watchlists = {
    "EYE FUTURE WATCHLIST": load_watchlist("Eye_d16ec.txt"),
    "WATCHLIST (2)": load_watchlist("Watchlist (2)_8e9c8.txt"),
    "FUTURE": load_watchlist("FUTURE_e8298.txt")
}

# ===============================
# Fake Astro Event Generator (Replace with real ephemeris logic)
# ===============================
def generate_astro_data(date, symbols):
    rows = []
    for sym in symbols:
        for hour in range(9, 16):  # market hours
            event = "Moon conjunct Mars" if hour % 2 == 0 else "Sun trine Jupiter"
            trend = "Bullish" if "Moon" in event else "Bearish"
            strength = 70 if trend == "Bullish" else 40
            rows.append({
                "DateTime": datetime(date.year, date.month, date.day, hour, 0),
                "Symbol": sym,
                "Trend": trend,
                "Strength": strength,
                "AstroEvent": event,
                "Remark": trend
            })
    return pd.DataFrame(rows)

# ===============================
# Streamlit App Layout
# ===============================
st.set_page_config(layout="wide", page_title="Astro Market Dashboard")

st.title("🔮 Astro Market Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    selected_date = st.date_input("Analysis Date", datetime.now().date())
with col2:
    selected_time = st.time_input("Analysis Time", time(9, 15))
with col3:
    analysis_type = st.selectbox("Analysis Type", ["All", "Bullish", "Bearish"])

watchlist_name = st.selectbox("Select Watchlist", list(watchlists.keys()))
symbols_list = watchlists[watchlist_name]

# Load Data
df = generate_astro_data(selected_date, symbols_list)

# Filter by trend
if analysis_type != "All":
    df = df[df["Trend"] == analysis_type]

# ===============================
# Sentiment Cards
# ===============================
st.subheader("📊 Watchlist Analysis Results")
cards_col = st.columns(4)

for i, sym in enumerate(df["Symbol"].unique()):
    subdf = df[df["Symbol"] == sym]
    avg_strength = subdf["Strength"].mean()
    sentiment = subdf["Trend"].mode()[0]
    color = "green" if sentiment == "Bullish" else "red" if sentiment == "Bearish" else "gray"

    with cards_col[i % 4]:
        st.markdown(
            f"""
            <div style='background-color:{color};padding:15px;border-radius:10px;text-align:center;color:white;'>
                <h4>{sym}</h4>
                <p>{sentiment}</p>
                <b>Strength: {avg_strength:.0f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================
# Table View
# ===============================
st.subheader(f"Astro Events for {selected_date} — {watchlist_name}")
st.dataframe(df, use_container_width=True)

# ===============================
# Timeline Chart
# ===============================
st.subheader("📅 Intraday Trend Timeline")
fig = px.scatter(df, x="DateTime", y="Symbol", color="Trend",
                 hover_data=["AstroEvent", "Strength"], symbol="Trend",
                 color_discrete_map={"Bullish": "blue", "Bearish": "red", "Neutral": "gray"})
st.plotly_chart(fig, use_container_width=True)
