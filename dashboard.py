import pandas as pd
import datetime
import streamlit as st

st.set_page_config(page_title="Astro‑Based Market Dashboard", layout="wide")
st.title("Astro‑Based Market Dashboard")

# ---------- File Upload Section ----------

# Upload ephemeris CSV
ephemeris_file = st.file_uploader(
    "Upload the sidereal ephemeris CSV (vedic_sidereal_ephemeris_mock_2024_2032.csv)",
    type=["csv"]
)

# Upload watch‑list file (comma‑separated list of symbols)
watchlist_file = st.file_uploader(
    "Upload your watch‑list file (comma‑separated symbols) – optional",
    type=["txt", "csv"]
)

if ephemeris_file is not None:
    # Load ephemeris into DataFrame
    astro_df = pd.read_csv(ephemeris_file)
    astro_df['Date'] = pd.to_datetime(astro_df['Date'])
    astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]

    st.success("Ephemeris data loaded.")
else:
    st.warning("Please upload the ephemeris CSV to continue.")

# Parse watch‑list into a list of symbols
if watchlist_file is not None:
    content = watchlist_file.read().decode('utf-8')
    stock_list = [s.strip() for s in content.split(',') if s.strip()]
    st.success(f"Loaded {len(stock_list)} symbols from watch‑list.")
else:
    stock_list = []
    st.info("No watch‑list uploaded; only planetary statuses will be shown.")

# ---------- Only proceed if ephemeris is loaded ----------
if 'astro_df' in locals():
    # Define planet rulership and exaltation
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

    # Add status column: Bullish if in own sign or exalted sign, else Bearish
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
                timeline.append({
                    'start': segment_start,
                    'end': segment_end,
                    'rashi': current_rashi,
                    'status': current_status
                })
                current_status = row['Status']
                current_rashi = row['Rashi']
                segment_start = row['Date']
        timeline.append({
            'start': segment_start,
            'end': df.iloc[-1]['Date'],
            'rashi': current_rashi,
            'status': current_status
        })
        return timeline

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

    # Parse watch‑list into DataFrame with optional planet mapping
    def stock_status_on_date(date_str):
        rows = []
        for stock in stock_list:
            planet = stock_planet_map.get(stock)  # defined below
            if planet:
                planet_info = build_watchlist([planet], date_str)[0]
                rows.append({
                    'stock': stock,
                    'planet': planet,
                    'status': planet_info['status'],
                    'rashi': planet_info['rashi'],
                    'days_until_change': planet_info['days_until_change'],
                    'next_change_date': planet_info['next_change_date']
                })
            else:
                rows.append({
                    'stock': stock,
                    'planet': None,
                    'status': None,
                    'rashi': None,
                    'days_until_change': None,
                    'next_change_date': None
                })
        return pd.DataFrame(rows)

    # ---------- Planet mapping (user must fill this) ----------
    # Example: assign each stock ticker to a ruling planet.
    # Without mapping, statuses will be None for those symbols.
    stock_planet_map = {
        # 'NSE:TRENT1!': 'Venus',
        # 'NSE:EICHERMOT1!': 'Mars',
        # ...
    }

    # ---------- UI for selecting date ----------
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Select a date",
            value=datetime.date(2025, 8, 10),
            min_value=datetime.date(2024, 8, 10),
            max_value=datetime.date(2030, 12, 31)
        )

    with col2:
        # optional: allow user to override default mapping via text input
        mapping_text = st.text_area(
            "Optional: enter stock→planet mappings (one per line, format: symbol,planet)",
            placeholder="NSE:TRENT1!,Venus\nNSE:INFY1!,Mercury",
            height=100
        )
        if mapping_text:
            for line in mapping_text.strip().splitlines():
                try:
                    sym, pl = [part.strip() for part in line.split(',')]
                    stock_planet_map[sym] = pl
                except ValueError:
                    pass  # ignore malformed lines

    # ---------- Generate report ----------
    if st.button("Generate Report"):
        date_str = selected_date.strftime("%Y-%m-%d")

        # Planetary status report
        planetary_watchlist = build_watchlist(
            ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'],
            date_str
        )
        st.subheader("Planetary Status on " + date_str)
        st.write(pd.DataFrame(planetary_watchlist))

        # Stock/commodity status report
        if stock_list:
            stock_df = stock_status_on_date(date_str)
            st.subheader("Watch‑List Symbols Status on " + date_str)
            st.write(stock_df)
        else:
            st.info("No watch‑list symbols provided, so only planetary status is shown.")
