import pandas as pd
import datetime
import streamlit as st
import os

# ---------- Load the ephemeris and prepare data ----------

# Adjust this path to where your CSV actually resides
EPHEMERIS_PATH = os.path.join(os.path.dirname(__file__), 'vedic_sidereal_ephemeris_mock_2024_2032.csv')
astro_df = pd.read_csv(EPHEMERIS_PATH)
astro_df['Date'] = pd.to_datetime(astro_df['Date'])
astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]

# Rulership and exaltation lists
planet_home_signs = {
    'Sun': ['Leo'],
    'Moon': ['Cancer'],
    'Mars': ['Aries', 'Scorpio'],
    'Mercury': ['Gemini', 'Virgo'],
    'Jupiter': ['Sagittarius', 'Pisces'],
    'Venus': ['Taurus', 'Libra'],
    'Saturn': ['Capricorn', 'Aquarius'],
}
planet_exalt_sign = {
    'Sun': 'Aries',
    'Moon': 'Taurus',
    'Mars': 'Capricorn',
    'Mercury': 'Virgo',
    'Jupiter': 'Cancer',
    'Venus': 'Pisces',
    'Saturn': 'Libra',
}

# Add a status (Bullish/Bearish) based on rulership and exaltation
astro_df['Status'] = astro_df.apply(
    lambda row: 'Bullish'
    if (row['Rashi'] in planet_home_signs.get(row['Planet'], [])) or
       (row['Rashi'] == planet_exalt_sign.get(row['Planet'], None))
    else 'Bearish',
    axis=1
)

# ---------- Helper functions ----------

def compute_timeline(symbol, start_date, end_date):
    df = astro_df[(astro_df['Planet'] == symbol) &
                  (astro_df['Date'] >= start_date) &
                  (astro_df['Date'] <= end_date)].copy().sort_values('Date')
    timeline = []
    if df.empty:
        return timeline
    current_status = df.iloc[0]['Status']
    current_rashi = df.iloc[0]['Rashi']
    segment_start = df.iloc[0]['Date']
    for i in range(1, len(df)):
        row = df.iloc[i]
        if row['Rashi'] != current_rashi or row['Status'] != current_status:
            segment_end = df.iloc[i - 1]['Date']
            timeline.append({'start': segment_start,
                             'end': segment_end,
                             'rashi': current_rashi,
                             'status': current_status})
            current_status = row['Status']
            current_rashi = row['Rashi']
            segment_start = row['Date']
    timeline.append({'start': segment_start,
                     'end': df.iloc[-1]['Date'],
                     'rashi': current_rashi,
                     'status': current_status})
    return timeline

def astro_timeline_hourly(symbol, date_str, start_time_str, end_time_str):
    date = pd.to_datetime(date_str)
    start_dt = datetime.datetime.combine(date.date(), pd.to_datetime(start_time_str).time())
    end_dt = datetime.datetime.combine(date.date(), pd.to_datetime(end_time_str).time())
    # Look a day before and after for changes
    timeline = compute_timeline(symbol, date - pd.Timedelta(days=1), date + pd.Timedelta(days=1))
    intraday_segments = []
    for seg in timeline:
        seg_start = seg['start']
        seg_end = seg['end'] + pd.Timedelta(hours=23)
        if seg_end.date() < date.date() or seg_start.date() > date.date():
            continue
        seg_start_time = max(seg_start, start_dt)
        seg_end_time = min(seg_end, end_dt)
        if seg_start_time <= seg_end_time:
            intraday_segments.append({
                'from': seg_start_time,
                'to': seg_end_time,
                'rashi': seg['rashi'],
                'status': seg['status']
            })
    return intraday_segments

def build_watchlist(symbols, date_str):
    date = pd.to_datetime(date_str)
    watchlist = []
    for sym in symbols:
        df = astro_df[(astro_df['Planet'] == sym) & (astro_df['Date'] <= date)].copy().sort_values('Date')
        if df.empty:
            continue
        current_row = df.iloc[-1]
        status = current_row['Status']
        rashi = current_row['Rashi']
        df_future = astro_df[(astro_df['Planet'] == sym) & (astro_df['Date'] > date)].sort_values('Date')
        next_change_date, days_until_change = None, None
        if not df_future.empty:
            initial_status = df_future.iloc[0]['Status']
            for i in range(1, len(df_future)):
                if df_future.iloc[i]['Status'] != initial_status:
                    next_change_date = df_future.iloc[i]['Date']
                    days_until_change = (next_change_date - date).days
                    break
        watchlist.append({
            'symbol': sym,
            'status': status,
            'rashi': rashi,
            'days_until_change': days_until_change,
            'next_change_date': next_change_date
        })
    return watchlist

# ---------- Read stock/commodity watch‑lists ----------

def parse_watchlist(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    return [s.strip() for s in content.split(',') if s.strip()]

# Combine the two uploaded watchlists
stock_list = parse_watchlist('FUTURE_e8298.txt') + parse_watchlist('Eye_d16ec.txt')

# You must supply a mapping from each stock symbol to its ruling planet.
# Without this, the program cannot determine whether the stock is bullish or bearish.
# Example (you need to populate this with real mappings):
stock_planet_map = {
    # 'NSE:TRENT1!': 'Venus',
    # 'NSE:EICHERMOT1!': 'Mars',
    # ...
}

def stock_status_on_date(date_str):
    results = []
    for stock in stock_list:
        planet = stock_planet_map.get(stock)
        if planet:
            planet_info = build_watchlist([planet], date_str)[0]
            results.append({
                'stock': stock,
                'planet': planet,
                'status': planet_info['status'],
                'rashi': planet_info['rashi'],
                'days_until_change': planet_info['days_until_change'],
                'next_change_date': planet_info['next_change_date']
            })
        else:
            # No planet mapping defined
            results.append({
                'stock': stock,
                'planet': None,
                'status': None,
                'rashi': None,
                'days_until_change': None,
                'next_change_date': None
            })
    return pd.DataFrame(results)

# ---------- Streamlit UI ----------

st.title("Full Astro‑Based Report")

# Date selection (only need the date, no planet selection)
selected_date = st.date_input(
    "Select a date",
    value=datetime.date(2025, 8, 10),
    min_value=datetime.date(2024, 8, 10),
    max_value=datetime.date(2030, 12, 31)
)

if st.button("Generate Report"):
    date_str = selected_date.strftime("%Y-%m-%d")
    # 1. Show planetary watchlist (Sun–Saturn)
    planetary_watchlist = build_watchlist(['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'], date_str)
    st.subheader("Planetary Status on " + date_str)
    st.write(pd.DataFrame(planetary_watchlist))

    # 2. Show stock/commodity status if mapping is provided
    stock_df = stock_status_on_date(date_str)
    st.subheader("Watch‑List Symbols Status on " + date_str)
    st.write(stock_df)
