import pandas as pd

# --------------------------------------------------------------------
# Load the sidereal ephemeris (make sure the CSV is in the same folder)
# --------------------------------------------------------------------
ephemeris_file = 'vedic_sidereal_ephemeris_mock_2024_2032.csv'
astro_df = pd.read_csv(ephemeris_file)
astro_df['Date'] = pd.to_datetime(astro_df['Date'])
# Restrict to 2024–2030 for this example
astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]

# --------------------------------------------------------------------
# Define rulership and exaltation signs for each classical planet
# --------------------------------------------------------------------
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

# Add a status column: Bullish if planet is in its own sign or exalted, else Bearish
astro_df['Status'] = astro_df.apply(
    lambda row: 'Bullish'
    if (row['Rashi'] in planet_home_signs.get(row['Planet'], [])) or
       (row['Rashi'] == planet_exalt_sign.get(row['Planet'], None))
    else 'Bearish',
    axis=1
)

# --------------------------------------------------------------------
# Helper function to compute a planet’s status and next change
# --------------------------------------------------------------------
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
        df_future = astro_df[(astro_df['Planet'] == sym) & (astro_df['Date'] > date)].copy().sort_values('Date')
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

# --------------------------------------------------------------------
# Define a mapping from watch‑list symbols to planets (demo only)
# You should replace these assignments with your own.
# --------------------------------------------------------------------
stock_planet_map = {
    'MCX:ALUMINIUM1!': 'Mars',
    'NSE:BANKNIFTY': 'Jupiter',
    'BITSTAMP:BTCUSD': 'Saturn',
    'NSE:CNX100': 'Sun',
    'NSE:CNX200': 'Moon',
    'NSE:CNXAUTO': 'Mars',
    'NSE:CNXCOMMODITIES': 'Mercury',
    'NSE:CNXCONSUMPTION': 'Venus',
    'NSE:CNXENERGY': 'Saturn',
    'NSE:CNXFINANCE': 'Jupiter'
}

# --------------------------------------------------------------------
# Compute the status for each mapped symbol on a given date
# --------------------------------------------------------------------
date_str = '2025-08-11'
results = []
for symbol, planet in stock_planet_map.items():
    planet_info = build_watchlist([planet], date_str)[0]
    results.append({
        'stock': symbol,
        'planet': planet,
        'status': planet_info['status'],
        'rashi': planet_info['rashi'],
        'days_until_change': planet_info['days_until_change'],
        'next_change_date': planet_info['next_change_date']
    })

# Convert to DataFrame for presentation
report_df = pd.DataFrame(results)
print(report_df)
