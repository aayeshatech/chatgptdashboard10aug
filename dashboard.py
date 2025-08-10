import streamlit as st
import pandas as pd
from datetime import datetime, time
import random

# ============= Helper Functions =============
def read_symbols_from_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8").strip()
        if not content:
            return []
        if "," in content:
            symbols = [s.strip() for s in content.split(",")]
        elif "\t" in content:
            symbols = [s.strip() for s in content.split("\t")]
        else:
            symbols = [line.strip() for line in content.split("\n")]
        return [s for s in symbols if s]
    except:
        return []

def generate_mock_astro_data(date, symbols):
    rows = []
    for sym in symbols:
        for hour in range(9, 16):
            trend = random.choice(["Bullish", "Bearish", "Neutral"])
            astro_event = random.choice([
                "Moon Conjunct Jupiter", "Sun Trine Saturn",
                "Venus Opposite Mars", "Mercury Sextile Venus",
                "Mars Square Neptune"
            ])
            rows.append({
                "DateTime": f"{date} {hour:02d}:00",
                "Symbol": sym,
                "AstroEvent": astro_event,
                "Impact": trend
            })
    return pd.DataFrame(rows)

def color_rows(row):
    if row["Impact"] == "Bullish":
        return ["background-color: lightblue"] * len(row)
    elif row["Impact"] == "Bearish":
        return ["background-color: lightcoral"] * len(row)
    else:
        return ["background-color: lightgrey"] * len(row)

# ============= Page Config =============
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("🔮 Astro Market Dashboard")

# ============= Sidebar - Watchlist Upload =============
st.sidebar.header("📂 Upload Watchlists")
wl1 = st.sidebar.file_uploader("Upload Watchlist 1", type=["txt", "csv"], key="wl1")
wl2 = st.sidebar.file_uploader("Upload Watchlist 2", type=["txt", "csv"], key="wl2")
wl3 = st.sidebar.file_uploader("Upload Watchlist 3", type=["txt", "csv"], key="wl3")

watchlists = {}
if wl1: watchlists["Watchlist 1"] = read_symbols_from_file(wl1)
if wl2: watchlists["Watchlist 2"] = read_symbols_from_file(wl2)
if wl3: watchlists["Watchlist 3"] = read_symbols_from_file(wl3)

if not watchlists:
    st.warning("Please upload at least one watchlist to continue.")
    st.stop()

selected_watchlist = st.sidebar.selectbox("Select Watchlist", list(watchlists.keys()), key="watchlist_select")
symbols_list = watchlists[selected_watchlist]
if not symbols_list:
    symbols_list = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]

# Common Month Dictionary
months_dict = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# ============= Tabs =============
tab1, tab2, tab3 = st.tabs(["Today Market", "Screener", "Upcoming Transit"])

# ===== TODAY MARKET TAB =====
with tab1:
    st.header("📅 Today Market Analysis")

    selected_year = st.selectbox("Select Year", list(range(2025, 2031)), key="today_market_year")
    selected_month = st.selectbox("Select Month", list(months_dict.values()), key="today_market_month")
    month_num = list(months_dict.keys())[list(months_dict.values()).index(selected_month)]
    selected_day = st.number_input("Select Date", min_value=1, max_value=31, value=datetime.now().day, key="today_market_day")
    start_time = st.time_input("Start Time", value=time(9, 0), key="today_market_start")
    end_time = st.time_input("End Time", value=time(15, 30), key="today_market_end")

    selected_date = f"{selected_year}-{month_num:02d}-{selected_day:02d}"
    df_today = generate_mock_astro_data(selected_date, symbols_list)
    df_today["TimeOnly"] = pd.to_datetime(df_today["DateTime"]).dt.time
    df_today = df_today[(df_today["TimeOnly"] >= start_time) & (df_today["TimeOnly"] <= end_time)]

    bull_df = df_today[df_today["Impact"] == "Bullish"]
    bear_df = df_today[df_today["Impact"] == "Bearish"]

    subtab1, subtab2 = st.tabs(["📈 Bullish", "📉 Bearish"])
    with subtab1:
        st.subheader("Bullish Events")
        st.dataframe(bull_df.style.apply(color_rows, axis=1), use_container_width=True)
    with subtab2:
        st.subheader("Bearish Events")
        st.dataframe(bear_df.style.apply(color_rows, axis=1), use_container_width=True)

# ===== SCREENER TAB =====
with tab2:
    st.header("📊 Screener")
    selected_year_sc = st.selectbox("Select Year", list(range(2025, 2031)), key="screener_year")
    selected_month_sc = st.selectbox("Select Month", list(months_dict.values()), key="screener_month")
    month_num_sc = list(months_dict.keys())[list(months_dict.values()).index(selected_month_sc)]
    selected_day_sc = st.number_input("Select Date", min_value=1, max_value=31, value=datetime.now().day, key="screener_day")
    selected_date_sc = f"{selected_year_sc}-{month_num_sc:02d}-{selected_day_sc:02d}"
    trend_filter = st.radio("Select Trend", ["All", "Bullish", "Bearish"], key="screener_trend")

    df_screen = generate_mock_astro_data(selected_date_sc, symbols_list)
    if trend_filter != "All":
        df_screen = df_screen[df_screen["Impact"] == trend_filter]

    st.dataframe(df_screen.style.apply(color_rows, axis=1), use_container_width=True)

# ===== UPCOMING TRANSIT TAB =====
with tab3:
    st.header("🔮 Upcoming Transit")
    selected_year_up = st.selectbox("Select Year", list(range(2025, 2031)), key="upcoming_year")
    selected_month_up = st.selectbox("Select Month", list(months_dict.values()), key="upcoming_month")
    month_num_up = list(months_dict.keys())[list(months_dict.values()).index(selected_month_up)]
    selected_day_up = st.number_input("Select Date", min_value=1, max_value=31, value=datetime.now().day, key="upcoming_day")
    selected_date_up = f"{selected_year_up}-{month_num_up:02d}-{selected_day_up:02d}"

    df_upcoming = generate_mock_astro_data(selected_date_up, symbols_list)
    upcoming_rows = []
    for sym in symbols_list:
        sym_df = df_upcoming[df_upcoming["Symbol"] == sym]
        if not sym_df.empty:
            next_event = sym_df.iloc[0]
            upcoming_rows.append({
                "Symbol": sym,
                "NextEvent": next_event["AstroEvent"],
                "Trend": next_event["Impact"],
                "Time": next_event["DateTime"]
            })
    upcoming_df = pd.DataFrame(upcoming_rows)
    st.dataframe(upcoming_df.style.apply(color_rows, axis=1), use_container_width=True)
