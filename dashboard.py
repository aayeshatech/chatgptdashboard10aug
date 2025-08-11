import pandas as pd
import datetime
import streamlit as st

st.set_page_config(page_title="Astro‑Based Market Dashboard", layout="wide")
st.title("Astro‑Based Market Dashboard")

# ------------------------------------------------------------------
# 1. File upload widgets
# ------------------------------------------------------------------

# Upload the ephemeris CSV file
ephemeris_file = st.file_uploader(
    "Upload the sidereal ephemeris CSV (vedic_sidereal_ephemeris_mock_2024_2032.csv)",
    type=["csv"]
)

# Upload the watch‑list file (comma‑separated symbols)
watchlist_file = st.file_uploader(
    "Upload your watch‑list file (comma‑separated symbols) – optional",
    type=["txt", "csv"]
)

# ------------------------------------------------------------------
# 2. Load ephemeris and watch‑list
# ------------------------------------------------------------------

if ephemeris_file is not None:
    # Read the ephemeris into a DataFrame
    astro_df = pd.read_csv(ephemeris_file)
    astro_df['Date'] = pd.to_datetime(astro_df['Date'])
    # Limit to the range 2024–2030
    astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]
    st.success("Ephemeris data loaded.")
else:
    astro_df = None
    st.warning("Please upload the ephemeris file to proceed.")

# Read watch‑list symbols into a list
if watchlist_file is not None:
    content = watchlist_file.read().decode('utf-8')
    stock_list = [s.strip() for s in content.split(',') if s.strip()]
    st.success(f"Loaded {len(stock_list)} watch‑list symbols.")
else:
    stock_list = []
    st.info("No watch‑list uploaded; only planetary status will be shown.")

# Proceed only if ephemeris is loaded
if astro_df is not None:

    # ------------------------------------------------------------------
    # 3. Define planet rulership and exaltation signs
    # ------------------------------------------------------------------
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

    # Add a 'Status' column: Bullish if in own or exalted sign, else Bearish
    astro_df['Status'] = astro_df.apply(
        lambda row: 'Bullish'
        if (row['Rashi'] in planet_home_signs.get(row['Planet'], [])) or
           (row['Rashi'] == planet_exalt_sign.get(row['Planet'], None))
        else 'Bearish',
        axis=1
    )

    # ------------------------------------------------------------------
    # 4. Helper functions
    # ------------------------------------------------------------------
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

    def stock_status_on_date(date_str, stock_planet_map):
        rows = []
        for stock in stock_list:
            planet = stock_planet_map.get(stock)
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

    # ------------------------------------------------------------------
    # 5. User inputs for date and symbol→planet mapping
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Select a date",
            value=datetime.date(2025, 8, 10),
            min_value=datetime.date(2024, 8, 10),
            max_value=datetime.date(2030, 12, 31)
        )

    with col2:
        mapping_text = st.text_area(
            "Optional: enter stock→planet mappings (one per line, format: symbol,planet)",
            placeholder="NSE:BANKNIFTY,Jupiter\nMCX:ALUMINIUM1!,Mars",
            height=120
        )

    # Convert mapping text into a dictionary
    stock_planet_map = {}
    if mapping_text.strip():
        for line in mapping_text.strip().splitlines():
            try:
                sym, pl = [part.strip() for part in line.split(',')]
                stock_planet_map[sym] = pl
            except ValueError:
                pass  # Ignore malformed lines

    # ------------------------------------------------------------------
    # 6. Generate report on button click
    # ------------------------------------------------------------------
    if st.button("Generate Report"):
        date_str = selected_date.strftime("%Y-%m-%d")

        # Planetary status report
        planetary_watchlist = build_watchlist(
            ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'],
            date_str
        )
        st.subheader(f"Planetary Status on {date_str}")
        st.write(pd.DataFrame(planetary_watchlist))

        # Watch‑list status report (only if symbols are loaded)
        if stock_list:
            stock_df = stock_status_on_date(date_str, stock_planet_map)
            st.subheader(f"Watch‑List Symbols Status on {date_str}")
            st.write(stock_df)
        else:
            st.info("No watch‑list symbols provided; only planetary status is shown.")
