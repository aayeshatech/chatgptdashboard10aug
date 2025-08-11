import pandas as pd
import numpy as np
import datetime

# Load the CSV file (assuming it is in the same folder)
file_path = 'vedic_sidereal_ephemeris_mock_2024_2032.csv'
astro_df = pd.read_csv(file_path)

# Convert the Date column to datetime and filter the desired range (2024–2030)
astro_df['Date'] = pd.to_datetime(astro_df['Date'])
astro_df = astro_df[(astro_df['Date'] >= '2024-01-01') & (astro_df['Date'] <= '2030-12-31')]

# Define home (rulership) and exaltation signs for each classical planet
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

# Add a new column 'Status' indicating Bullish if planet is in its home or exalted sign, otherwise Bearish
astro_df['Status'] = astro_df.apply(
    lambda row: 'Bullish'
    if (row['Rashi'] in planet_home_signs.get(row['Planet'], [])) or
       (row['Rashi'] == planet_exalt_sign.get(row['Planet'], None))
    else 'Bearish',
    axis=1
)

def compute_timeline(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Return a list of segments where the planet stays in the same sign and status
    between start_date and end_date.
    """
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

    # Append the final segment
    timeline.append({
        'start': segment_start,
        'end': df.iloc[-1]['Date'],
        'rashi': current_rashi,
        'status': current_status
    })
    return timeline

def astro_timeline_hourly(symbol: str, date_str: str, start_time_str: str, end_time_str: str):
    """
    For a given date and intraday time range, return the planet’s status segments.
    Because the ephemeris is daily, the status remains constant within the day.
    """
    date = pd.to_datetime(date_str)
    start_dt = datetime.datetime.combine(date.date(), pd.to_datetime(start_time_str).time())
    end_dt = datetime.datetime.combine(date.date(), pd.to_datetime(end_time_str).time())

    # Look one day either side to catch sign changes at midnight
    timeline = compute_timeline(symbol, date - pd.Timedelta(days=1), date + pd.Timedelta(days=1))
    intraday_segments = []
    for seg in timeline:
        # Only keep segments that intersect our date
        seg_start = seg['start']
        seg_end = seg['end'] + pd.Timedelta(hours=23)
        if seg_end.date() < date.date() or seg_start.date() > date.date():
            continue
        # Intersection of our intraday period with this segment
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

def build_watchlist(symbols: list, date_str: str):
    """
    For each symbol in the list, return its status on the given date and the next change date.
    """
    date = pd.to_datetime(date_str)
    watchlist = []
    for sym in symbols:
        df = astro_df[(astro_df['Planet'] == sym) & (astro_df['Date'] <= date)].copy().sort_values('Date')
        if df.empty:
            continue
        current_row = df.iloc[-1]
        status = current_row['Status']
        rashi = current_row['Rashi']

        # Determine the next date when the status changes
        df_future = astro_df[(astro_df['Planet'] == sym) & (astro_df['Date'] > date)].sort_values('Date')
        next_change_date = None
        days_until_change = None
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

# Example usage:
# Watchlist for 10 August 2025
watchlist_data = build_watchlist(['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'], '2025-08-10')
print("Watchlist on 2025-08-10:", watchlist_data)

# Intraday timeline for Mars on 2025-08-10 (09:00 – 16:00)
timeline_data = astro_timeline_hourly('Mars', '2025-08-10', '09:00', '16:00')
print("Intraday timeline for Mars:", timeline_data)
