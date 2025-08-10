import swisseph as swe
import pandas as pd
from datetime import datetime, timedelta
import os

# Set ephemeris path
swe.set_ephe_path("ephe")

DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
    "Rahu": swe.MEAN_NODE
}

WATCHLIST = ["TATASTEEL", "RELIANCE", "INFY", "HDFCBANK"]

def get_rashi(longitude):
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    return signs[int(longitude // 30)]

def generate_ephemeris(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23)
    rows = []
    dt = start
    while dt <= end:
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour)
        planet_positions = {}
        for pname, pcode in PLANETS.items():
            lon, lat, dist, speed = swe.calc_ut(jd, pcode)
            planet_positions[pname] = (get_rashi(lon), "R" if speed < 0 else "D")
        for sym in WATCHLIST:
            rows.append({
                "DateTime": dt.strftime("%Y-%m-%d %H:%M"),
                "Symbol": sym,
                "Trend": "🟢 Bullish",  # Placeholder for now
                "Events": ""  # Placeholder for now
            })
        dt += timedelta(hours=1)
    pd.DataFrame(rows).to_csv(f"{DATA_PATH}/ephemeris_{year}.csv", index=False)
    pd.DataFrame([{"Symbol": s, "Year": year, "MainTrend": "🟢 Bullish"} for s in WATCHLIST]) \
        .to_csv(f"{DATA_PATH}/summary_{year}.csv", index=False)

for y in range(2025, 2030 + 1):
    print(f"Generating {y}...")
    generate_ephemeris(y)

print("✅ All CSV files generated in 'data/' folder.")
