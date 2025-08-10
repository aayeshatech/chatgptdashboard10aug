import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# ===================== Mock Astro Data Generator =====================
def generate_mock_astro_data(date_str, symbols):
    """Generate realistic mock astro events for testing."""
    transits = ["Moon Conjunct Sun", "Mars Square Jupiter", "Venus Trine Saturn", "Mercury Sextile Uranus"]
    aspects = ["Conjunct", "Opposition", "Trine", "Square", "Sextile"]
    lords = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    sublords = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    nakshatras = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu"]
    impacts = ["Bullish", "Bearish"]

    data = []
    for symbol in symbols:
        for i in range(random.randint(1, 3)):  # multiple events per day
            dt_time = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            event = {
                "DateTime": dt_time,
                "Symbol": symbol,
                "Transit": random.choice(transits),
                "Aspect": random.choice(aspects),
                "Lord": random.choice(lords),
                "SubLord": random.choice(sublords),
                "Nakshatra": random.choice(nakshatras),
                "Impact": random.choice(impacts)
            }
            data.append(event)
    return pd.DataFrame(data)

# ===================== Styling Function =====================
def color_rows(row):
    if row["Impact"] == "Bullish":
        return ["background-color: lightblue"] * len(row)
    elif row["Impact"] == "Bearish":
        return ["background-color: pink"] * len(row)
    return [""] * len(row)

# ===================== Sidebar & Watchlist =====================
st.sidebar.header("📂 Upload Watchlists")
uploaded_files = st.sidebar.file_uploader("Upload up to 3 watchlist files", accept_multiple_files=True, type=["txt", "csv"])
symbols_list = []
if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode("utf-8").splitlines()
        symbols_list.extend(content)
    symbols_list = list(set(symbols_list))  # unique symbols
else:
    symbols_list = ["AAPL", "GOOG", "MSFT"]  # default for testing

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs(["📅 Today Market", "📊 Advanced Screener", "🔮 Upcoming Transit"])

months_dict = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}

# ===== TODAY MARKET TAB =====
with tab1:
    st.header("📅 Today Market")
    year = st.selectbox("Select Year", list(range(2025, 2031)), key="tm_year")
    month = st.selectbox("Select Month", list(months_dict.values()), key="tm_month")
    month_num = list(months_dict.keys())[list(months_dict.values()).index(month)]
    day = st.number_input("Select Day", min_value=1, max_value=31, value=datetime.now().day)
    start_time = st.time_input("Start Time", value=datetime.now().time())
    end_time = st.time_input("End Time", value=(datetime.now() + timedelta(hours=1)).time())

    date_str = f"{year}-{month_num:02d}-{day:02d}"
    df_today = generate_mock_astro_data(date_str, symbols_list)

    st.subheader("Bullish")
    st.dataframe(df_today[df_today["Impact"] == "Bullish"].style.apply(color_rows, axis=1), use_container_width=True)

    st.subheader("Bearish")
    st.dataframe(df_today[df_today["Impact"] == "Bearish"].style.apply(color_rows, axis=1), use_container_width=True)

# ===== ADVANCED SCREENER TAB =====
with tab2:
    st.header("📊 Advanced Astro Screener")
    period_type = st.radio("Select Period", ["Year", "Month", "Week"], horizontal=True)

    if period_type == "Year":
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_year")
        start = datetime(year_sc, 1, 1)
        end = datetime(year_sc, 12, 31)
    elif period_type == "Month":
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_month_year")
        month_sc = st.selectbox("Month", list(months_dict.values()), key="sc_month")
        month_num_sc = list(months_dict.keys())[list(months_dict.values()).index(month_sc)]
        start = datetime(year_sc, month_num_sc, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    else:  # Week
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="sc_week_year")
        week_num = st.number_input("Week Number", min_value=1, max_value=52, value=datetime.now().isocalendar()[1])
        start = datetime.strptime(f"{year_sc}-W{week_num}-1", "%Y-W%W-%w")
        end = start + timedelta(days=6)

    # Generate events for entire range
    all_events = []
    current_day = start
    while current_day <= end:
        df_day = generate_mock_astro_data(current_day.strftime("%Y-%m-%d"), symbols_list)
        all_events.append(df_day)
        current_day += timedelta(days=1)

    astro_df = pd.concat(all_events, ignore_index=True)
    astro_df["DateTime"] = pd.to_datetime(astro_df["DateTime"])
    astro_df = astro_df.sort_values(by="DateTime")

    # Filter by Impact
    impact_filter = st.selectbox("Filter Impact", ["All", "Bullish", "Bearish"])
    if impact_filter != "All":
        astro_df = astro_df[astro_df["Impact"] == impact_filter]

    st.subheader("Astro Events")
    st.dataframe(
        astro_df[["DateTime", "Symbol", "Transit", "Aspect", "Lord", "SubLord", "Nakshatra", "Impact"]]
        .style.apply(color_rows, axis=1),
        use_container_width=True
    )

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
    date_pick = st.date_input("Select Start Date", datetime.now())
    days_ahead = st.number_input("Days Ahead", min_value=1, max_value=365, value=7)

    all_events = []
    for i in range(days_ahead):
        df_day = generate_mock_astro_data((date_pick + timedelta(days=i)).strftime("%Y-%m-%d"), symbols_list)
        all_events.append(df_day)

    upcoming_df = pd.concat(all_events, ignore_index=True)
    if not upcoming_df.empty:
        st.dataframe(upcoming_df.style.apply(color_rows, axis=1), use_container_width=True)
    else:
        st.warning("No upcoming transit data available.")
