import pandas as pd
import re

# Load the ephemeris
astro_df = pd.read_csv('vedic_sidereal_ephemeris_mock_2024_2032.csv')
astro_df['Date'] = pd.to_datetime(astro_df['Date'])
astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]

# Define which signs make a planet Bullish (own sign or exaltation)
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

# Compute status for each planet on each day
astro_df['Status'] = astro_df.apply(
    lambda row: 'Bullish'
    if (row['Rashi'] in planet_home_signs.get(row['Planet'], [])) or
       (row['Rashi'] == planet_exalt_sign.get(row['Planet'], None))
    else 'Bearish',
    axis=1
)

# Function to get status and next change date for a planet on a given date
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

# Read your watch‑list files into a list of symbols
def parse_watchlist(path):
    with open(path, 'r') as f:
        return [s.strip() for s in f.read().split(',') if s.strip()]

symbols1 = parse_watchlist('FUTURE_e8298.txt')
symbols2 = parse_watchlist('Eye_552b8.txt')
all_symbols = symbols1 + symbols2

# Assign planets based on first letter of the symbol (A–D→Sun, E–H→Moon, I–L→Mars, M–Q→Mercury, R–U→Jupiter, V–Y→Venus, others→Saturn)
planet_order = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']
def assign_planet(symbol):
    m = re.search(r'[A-Za-z]', symbol)
    if m:
        ch = m.group(0).upper()
        idx = (ord(ch) - ord('A')) // 4  # each 4 letters of the alphabet
        idx = min(idx, len(planet_order) - 1)
        return planet_order[idx]
    return 'Saturn'

stock_planet_map = {sym: assign_planet(sym) for sym in all_symbols}

# Compute status for each symbol on 2025‑08‑11
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

# Create a DataFrame for display
res_df = pd.DataFrame(results)

# Summarise results
print("Total symbols:", len(res_df))
print("Bullish symbols:", (res_df['status'] == 'Bullish').sum())
print("Bearish symbols:", (res_df['status'] == 'Bearish').sum())
print("\nFirst few rows of the result:")
print(res_df.head(10))
