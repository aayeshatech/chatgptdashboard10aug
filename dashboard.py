import streamlit as st
import pandas as pd
from datetime import datetime, time, timedelta
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

def generate_mock_astro_data(date, symbols, hours_range=range(9, 16)):
    planets = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
    aspects = ["Conjunct", "Opposite", "Trine", "Square", "Sextile"]
    nakshatras = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu"]
    lords = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

    rows = []
    for sym in symbols:
        for hour in hours_range:
            planet1 = random.choice(planets)
            planet2 = random.choice([p for p in planets if p != planet1])
            aspect = random.choice(aspects)
            event = f"{planet1} {aspect} {planet2}"
            trend = random.choice(["Bullish", "Bearish"])
            rows.append({
                "DateTime": f"{date} {hour:02d}:00",
                "Symbol": sym,
                "Transit": f"{planet1} in {random.choice(['Aries','Taurus','Gemini','Cancer'])}",
                "Aspect": aspect,
                "Lord": random.choice(lords),
                "SubLord": random.choice(lords),
                "Nakshatra": random.choice(nakshatras),
                "AstroEvent": event,
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
symbols_list = watchlists[selected_watchlist] or ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]

months_dict = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# ============= Tabs =============
tab1, tab2, tab3 = st.tabs(["Today Market", "Advanced Screener", "Upcoming Transit"])

# ===== TODAY MARKET TAB =====
with tab1:
    st.header("📅 Today Market Analysis")
    selected_year = st.selectbox("Year", list(range(2025, 2031)), key="today_year")
    selected_month = st.selectbox("Month", list(months_dict.values()), key="today_month")
    month_num = list(months_dict.keys())[list(months_dict.values()).index(selected_month)]
    selected_day = st.number_input("Date", 1, 31, value=datetime.now().day, key="today_day")
    start_time = st.time_input("Start Time", value=time(9, 0))
    end_time = st.time_input("End Time", value=time(15, 30))
    selected_date = f"{selected_year}-{month_num:02d}-{selected_day:02d}"
    df_today = generate_mock_astro_data(selected_date, symbols_list)
    df_today["TimeOnly"] = pd.to_datetime(df_today["DateTime"]).dt.time
    df_today = df_today[(df_today["TimeOnly"] >= start_time) & (df_today["TimeOnly"] <= end_time)]
    bull_df = df_today[df_today["Impact"] == "Bullish"]
    bear_df = df_today[df_today["Impact"] == "Bearish"]
    btab, ntab = st.tabs(["📈 Bullish", "📉 Bearish"])
    with btab:
        st.dataframe(bull_df.style.apply(color_rows, axis=1), use_container_width=True)
    with ntab:
        st.dataframe(bear_df.style.apply(color_rows, axis=1), use_container_width=True)

# ===== ADVANCED SCREENER TAB =====
with tab2:
    st.header("📊 Advanced Astro Screener")
    period_type = st.radio("Select Period", ["Year", "Month", "Week"], horizontal=True)
    if period_type == "Year":
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_year")
        date_range = [f"{year_sc}-01-01", f"{year_sc}-12-31"]
    elif period_type == "Month":
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_month_year")
        month_sc = st.selectbox("Month", list(months_dict.values()), key="sc_month")
        month_num_sc = list(months_dict.keys())[list(months_dict.values()).index(month_sc)]
        start = datetime(year_sc, month_num_sc, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        date_range = [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]
    else:  # Week
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_week_year")
        week_num = st.number_input("Week Number", min_value=1, max_value=52, value=datetime.now().isocalendar()[1])
        first_day = datetime.strptime(f"{year_sc}-W{week_num}-1", "%Y-W%W-%w")
        last_day = first_day + timedelta(days=6)
        date_range = [first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")]

    astro_df = generate_mock_astro_data(date_range[0], symbols_list)
    st.subheader("Astro Events")
    st.dataframe(astro_df[["DateTime", "Transit", "Aspect", "Lord", "SubLord", "Nakshatra", "Impact"]]
                 .style.apply(color_rows, axis=1), use_container_width=True)

    st.subheader("Symbol Impact")
    bull_symbols = astro_df[astro_df["Impact"] == "Bullish"]["Symbol"].unique()
    bear_symbols = astro_df[astro_df["Impact"] == "Bearish"]["Symbol"].unique()
    btab, ntab = st.tabs(["📈 Bullish Symbols", "📉 Bearish Symbols"])
    with btab:
        st.write(bull_symbols)
    with ntab:
        st.write(bear_symbols)

# ===== UPCOMING TRANSIT TAB =====
with tab3:
    st.header("🔮 Upcoming Transit")
    selected_year_up = st.selectbox("Year", list(range(2025, 2031)), key="up_year")
    selected_month_up = st.selectbox("Month", list(months_dict.values()), key="up_month")
    month_num_up = list(months_dict.keys())[list(months_dict.values()).index(selected_month_up)]
    selected_day_up = st.number_input("Date", 1, 31, value=datetime.now().day, key="up_day")
    selected_date_up = f"{selected_year_up}-{month_num_up:02d}-{selected_day_up:02d}"
    df_upcoming = generate_mock_astro_data(selected_date_up, symbols_list)
    upcoming_rows = []
    for sym in symbols_list:
        sym_df = df_upcoming[df_upcoming["Symbol"] == sym]
        if not sym_df.empty:
            next_event = sym_df.iloc[0]
            upcoming_rows.append({
                "Symbol": sym,
                "NextEvent": next_event.get("AstroEvent", ""),
                "Impact": next_event.get("Impact", ""),
                "Time": next_event.get("DateTime", "")
            })
    upcoming_df = pd.DataFrame(upcoming_rows)
    if not upcoming_df.empty and "Impact" in upcoming_df.columns:
        st.dataframe(upcoming_df.style.apply(color_rows, axis=1), use_container_width=True)
    else:
        st.info("No upcoming events for the selected date.")
