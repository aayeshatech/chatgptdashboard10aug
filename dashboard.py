import streamlit as st
import pandas as pd
from datetime import datetime, time
import random

# ===== Mock Astro Data (Replace with real logic) =====
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

# ===== Dashboard Layout =====
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("🔮 Astro Market Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["Today Market", "Screener", "Upcoming Transit"])

# ===== TODAY MARKET TAB =====
with tab1:
    st.header("📅 Today Market Analysis")

    # Filters
    years = list(range(2025, 2031))
    months_dict = {1: "January", 2: "February", 3: "March", 4: "April",
                   5: "May", 6: "June", 7: "July", 8: "August",
                   9: "September", 10: "October", 11: "November", 12: "December"}
    selected_year = st.selectbox("Select Year", years)
    selected_month = st.selectbox("Select Month", list(months_dict.values()))
    month_num = list(months_dict.keys())[list(months_dict.values()).index(selected_month)]
    selected_day = st.number_input("Select Date", min_value=1, max_value=31, value=datetime.now().day)
    selected_date = f"{selected_year}-{month_num:02d}-{selected_day:02d}"

    start_time = st.time_input("Start Time", value=time(9, 0))
    end_time = st.time_input("End Time", value=time(15, 30))

    # Watchlist selection
    symbols_list = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]  # Replace with real watchlist
    df = generate_mock_astro_data(selected_date, symbols_list)

    # Filter by time range
    df["TimeOnly"] = pd.to_datetime(df["DateTime"]).dt.time
    df = df[(df["TimeOnly"] >= start_time) & (df["TimeOnly"] <= end_time)]

    # Split into bullish/bearish tabs
    bull_df = df[df["Impact"] == "Bullish"]
    bear_df = df[df["Impact"] == "Bearish"]

    subtab1, subtab2 = st.tabs(["📈 Bullish", "📉 Bearish"])

    with subtab1:
        st.subheader("Bullish Events")
        st.dataframe(bull_df.style.apply(lambda row: ["background-color: lightblue"]*len(row), axis=1), use_container_width=True)

    with subtab2:
        st.subheader("Bearish Events")
        st.dataframe(bear_df.style.apply(lambda row: ["background-color: lightcoral"]*len(row), axis=1), use_container_width=True)
import streamlit as st
import pandas as pd
from datetime import datetime
import random

# ===========================
# Safe Watchlist Loader
# ===========================
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

# ===========================
# Mock Astro Data Generator
# ===========================
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
            remark = trend
            rows.append({
                "DateTime": f"{date} {hour:02d}:00",
                "Symbol": sym,
                "Trend": trend,
                "AstroEvent": astro_event,
                "Remark": remark
            })
    return pd.DataFrame(rows)

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Astro Market Dashboard", layout="wide")
st.title("🔮 Astro Market Dashboard")

# Upload Watchlists
st.sidebar.header("📂 Upload Watchlists")
wl1 = st.sidebar.file_uploader("Upload Watchlist 1", type=["txt", "csv"])
wl2 = st.sidebar.file_uploader("Upload Watchlist 2", type=["txt", "csv"])
wl3 = st.sidebar.file_uploader("Upload Watchlist 3", type=["txt", "csv"])

watchlists = {}
if wl1: watchlists["Watchlist 1"] = read_symbols_from_file(wl1)
if wl2: watchlists["Watchlist 2"] = read_symbols_from_file(wl2)
if wl3: watchlists["Watchlist 3"] = read_symbols_from_file(wl3)

if not watchlists:
    st.warning("Please upload at least one watchlist to continue.")
    st.stop()

# Month & Date selection
months_dict = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
selected_month = st.selectbox("Select Month", list(months_dict.values()))
month_num = list(months_dict.keys())[list(months_dict.values()).index(selected_month)]
dates_in_month = list(range(1, 32))
selected_day = st.selectbox("Select Date", dates_in_month, index=datetime.now().day - 1)
selected_date = f"2025-{month_num:02d}-{selected_day:02d}"

# Watchlist selection
watchlist_name = st.selectbox("Select Watchlist", list(watchlists.keys()))
symbols_list = watchlists[watchlist_name]
if not symbols_list:
    st.warning(f"No symbols found in {watchlist_name}. Showing demo symbols.")
    symbols_list = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]

# Trend filter
analysis_type = st.radio("Filter Events", ["All", "Bullish", "Bearish"], horizontal=True)

# Generate Data
df = generate_mock_astro_data(selected_date, symbols_list)
if analysis_type != "All":
    df = df[df["Trend"] == analysis_type]
if df.empty:
    st.warning("No astro events found for this selection.")
    st.stop()

# ===========================
# Sentiment Cards
# ===========================
st.subheader("📊 Watchlist Analysis Results")
cols = st.columns(3)
bullish_count = len(df[df["Trend"] == "Bullish"])
bearish_count = len(df[df["Trend"] == "Bearish"])
neutral_count = len(df[df["Trend"] == "Neutral"])
total = len(df)

cols[0].metric("📈 Bullish", f"{bullish_count} ({(bullish_count/total*100):.0f}%)")
cols[1].metric("📉 Bearish", f"{bearish_count} ({(bearish_count/total*100):.0f}%)")
cols[2].metric("⚖ Neutral", f"{neutral_count} ({(neutral_count/total*100):.0f}%)")

# ===========================
# Astro Events Table
# ===========================
st.subheader(f"Astro Events for {selected_date} — {watchlist_name}")

def color_rows(row):
    if row["Remark"] == "Bullish":
        return ["background-color: lightblue"] * len(row)
    elif row["Remark"] == "Bearish":
        return ["background-color: lightcoral"] * len(row)
    else:
        return ["background-color: lightgrey"] * len(row)

styled_df = df.style.apply(color_rows, axis=1)
st.dataframe(styled_df, use_container_width=True)

# ===========================
# Upcoming Events Section
# ===========================
st.subheader("🔮 Upcoming Astro Events")
upcoming_rows = []
for sym in symbols_list:
    sym_df = df[df["Symbol"] == sym]
    if not sym_df.empty:
        next_event = sym_df.iloc[0]
        upcoming_rows.append({
            "Symbol": sym,
            "NextEvent": next_event["AstroEvent"],
            "Trend": next_event["Trend"],
            "Time": next_event["DateTime"]
        })
upcoming_df = pd.DataFrame(upcoming_rows)
st.table(upcoming_df)
