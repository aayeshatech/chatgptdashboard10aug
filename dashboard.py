import pandas as pd
import os
from datetime import datetime, timedelta

# =======================
# CONFIG
# =======================
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

WATCHLIST = ["TATASTEEL", "RELIANCE", "INFY", "HDFCBANK"]

# Base planetary positions on 10 Aug 2024 (sidereal, decimal degrees)
BASE_POSITIONS = {
    "Sun": 113.8333,       # Cancer 23°50’
    "Moon": 176.8667,      # Virgo 26°52’
    "Mercury": 128.8000,   # Leo 8°48’ (R)
    "Venus": 132.0333,     # Leo 12°02’
    "Mars": 49.5500,       # Taurus 19°33’
    "Jupiter": 51.7667,    # Taurus 21°46’
    "Saturn": 323.8833,    # Aquarius 23°53’ (R)
    "Uranus": 32.8333,     # Taurus 2°50’
    "Neptune": 335.3333,   # Pisces 5°20’ (R)
    "Pluto": 276.2500,     # Capricorn 6°15’ (R)
    "Rahu": 344.9000,      # Pisces 14°54’ (M)
    "TrueNode": 343.4500,  # Pisces 13°27’ (T)
    "Lilith": 160.4167,    # Virgo 10°25’
    "Chiron": 359.2333     # Pisces 29°14’
}

# Average sidereal motion in degrees/hour
MOTION_SPEEDS = {
    "Sun": 0.04107,
    "Moon": 0.54902,
    "Mercury": 0.05764,
    "Venus": 0.05000,
    "Mars": 0.02183,
    "Jupiter": 0.00346,
    "Saturn": 0.00140,
    "Uranus": 0.00050,
    "Neptune": 0.00025,
    "Pluto": 0.000166,
    "Rahu": -0.00417,      # Retrograde
    "TrueNode": -0.00417,
    "Lilith": 0.0,
    "Chiron": 0.0
}

# Nakshatra details
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

def get_rashi(deg):
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    return signs[int(deg // 30)]

def get_nakshatra(deg):
    segment = 13 + 1/3  # 13°20′
    index = int(deg // segment) % 27
    pada = int((deg % segment) // (segment / 4)) + 1
    return NAKSHATRAS[index], pada

def calculate_trend(planet_positions):
    # Simple rule: Jupiter + Moon in benefic rashis => bullish
    moon_rashi = get_rashi(planet_positions["Moon"])
    jup_rashi = get_rashi(planet_positions["Jupiter"])
    benefics = {"Taurus", "Cancer", "Sagittarius", "Pisces"}
    if moon_rashi in benefics and jup_rashi in benefics:
        return "🟢 Bullish"
    elif moon_rashi in {"Capricorn", "Scorpio"}:
        return "🔴 Bearish"
    return "⚪ Neutral"

# =======================
# GENERATE FUNCTION
# =======================
def generate_ephemeris(year):
    start_dt = datetime(2024, 8, 10, 0, 0)
    end_dt = datetime(year, 12, 31, 23, 0)
    positions = BASE_POSITIONS.copy()
    rows = []

    dt = start_dt
    while dt <= end_dt:
        trend = calculate_trend(positions)
        nakshatra_info = {p: get_nakshatra(pos) for p, pos in positions.items()}
        for sym in WATCHLIST:
            rows.append({
                "DateTime": dt.strftime("%Y-%m-%d %H:%M"),
                "Symbol": sym,
                "Trend": trend,
                "Events": ", ".join(f"{p}:{get_rashi(pos)}-{nakshatra_info[p][0]}P{nakshatra_info[p][1]}"
                                    for p, pos in positions.items())
            })
        # Increment positions
        for planet, speed in MOTION_SPEEDS.items():
            positions[planet] = (positions[planet] + speed) % 360
        dt += timedelta(hours=1)

    df = pd.DataFrame(rows)
    df_year = df[df["DateTime"].str.startswith(str(year))]
    df_year.to_csv(f"{DATA_PATH}/ephemeris_{year}.csv", index=False)

    # Daily summary
    df_daily = df_year.groupby(["Symbol"]).agg({"Trend": lambda x: x.mode()[0] if not x.mode().empty else "⚪ Neutral"}).reset_index()
    df_daily["Year"] = year
    df_daily.to_csv(f"{DATA_PATH}/summary_{year}.csv", index=False)

# =======================
# RUN FOR ALL YEARS
# =======================
for y in range(2024, 2033):
    print(f"Generating {y}...")
    generate_ephemeris(y)

print("✅ All CSV files generated in 'data/' folder (2024–2032).")
