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
        for _ in range(random.randint(1, 3)):  # multiple events per day
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

# ===================== Sidebar Watchlist =====================
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
tab1, tab2 = st.tabs(["📊 Market & Astro Screener", "🔮 Upcoming Transit"])

months_dict = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}

# ===== MERGED MARKET & ASTRO SCREENER =====
with tab1:
    st.header("📊 Market & Astro Screener")
    period_type = st.radio("Select Period", ["Today", "Week", "Month"], horizontal=True)

    if period_type == "Today":
        selected_date = st.date_input("Select Date", datetime.now())
        start = selected_date
        end = selected_date

    elif period_type == "Week":
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="wk_year")
        week_num = st.number_input("Week Number", min_value=1, max_value=52, value=datetime.now().isocalendar()[1])
        start = datetime.strptime(f"{year_sc}-W{week_num}-1", "%Y-W%W-%w")
        end = start + timedelta(days=6)

    else:  # Month
        year_sc = st.selectbox("Year", list(range(2025, 2031)), key="mo_year")
        month_sc = st.selectbox("Month", list(months_dict.values()), key="mo_month")
        month_num_sc = list(months_dict.keys())[list(months_dict.values()).index(month_sc)]
        start = datetime(year_sc, month_num_sc, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    # Generate events for date range
    all_events = []
    current_day = start
    while current_day <= end:
        df_day = generate_mock_astro_data(current_day.strftime("%Y-%m-%d"), symbols_list)
        all_events.append(df_day)
        current_day += timedelta(days=1)

    astro_df = pd.concat(all_events, ignore_index=True)
    astro_df["DateTime"] = pd.to_datetime(astro_df["DateTime"])
    astro_df = astro_df.sort_values(by="DateTime")

    # Tabs for each date
    unique_dates = astro_df["DateTime"].dt.date.unique()
    date_tabs = st.tabs([d.strftime("%d %b %Y") for d in unique_dates])

    for i, date_val in enumerate(unique_dates):
        with date_tabs[i]:
            day_df = astro_df[astro_df["DateTime"].dt.date == date_val]

            bull_df = day_df[day_df["Impact"] == "Bullish"].copy()
            bear_df = day_df[day_df["Impact"] == "Bearish"].copy()

            # Count duration effect
            for df in [bull_df, bear_df]:
                df["Impact Duration (Days)"] = df.groupby("Symbol")["DateTime"].transform(lambda x: len(x.unique()))

            bull_tab, bear_tab = st.tabs(["📈 Bullish", "📉 Bearish"])

            with bull_tab:
                if not bull_df.empty:
                    st.write(f"### Bullish Events for {date_val}")
                    st.dataframe(bull_df.style.apply(color_rows, axis=1), use_container_width=True)
                else:
                    st.info("No Bullish events for this date.")

            with bear_tab:
                if not bear_df.empty:
                    st.write(f"### Bearish Events for {date_val}")
                    st.dataframe(bear_df.style.apply(color_rows, axis=1), use_container_width=True)
                else:
                    st.info("No Bearish events for this date.")

# ===== UPCOMING TRANSIT TAB =====
with tab2:
    st.header("🔮 Upcoming Transit")
    date_pick = st.date_input("Select Start Date", datetime.now())
    days_ahead = st.number_input("Days Ahead", min_value=1, max_value=365, value=7)

    all_events = []
    for i in range(days_ahead):
        df_day = generate_mock_astro_data((date_pick + timedelta(days=i)).strftime("%Y-%m-%d"), symbols_list)
        all_events.append(df_day)

    upcoming_df = pd.concat(all_events, ignore_index=True)
    upcoming_df["DateTime"] = pd.to_datetime(upcoming_df["DateTime"])
    upcoming_df = upcoming_df.sort_values(by="DateTime")

    if not upcoming_df.empty:
        st.dataframe(upcoming_df.style.apply(color_rows, axis=1), use_container_width=True)
    else:
        st.warning("No upcoming transit data available.")
