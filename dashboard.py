import pandas as pd
import os
from datetime import datetime, timedelta

DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

WATCHLIST = ["TATASTEEL", "RELIANCE", "INFY", "HDFCBANK"]

BASE_POSITIONS = {
    "Sun": 113.8333,
    "Moon": 176.8667,
    "Mercury": 128.8000,
    "Venus": 132.0333,
    "Mars": 49.5500,
    "Jupiter": 51.7667,
    "Saturn": 323.8833,
    "Uranus": 32.8333,
    "Neptune": 335.3333,
    "Pluto": 276.2500,
    "Rahu": 344.9000,
    "TrueNode": 343.4500,
    "Lilith": 160.4167,
    "Chiron": 359.2333
}

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
    "Rahu": -0.00417,
    "TrueNode": -0.00417,
    "Lilith": 0.0,
    "Chiron": 0.0
}

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
    segment = 13 + 1/3
    index = int(deg // segment) % 27
    pada = int((deg % segment) // (segment / 4)) + 1
    return NAKSHATRAS[index], pada

def calculate_trend(positions):
    moon_rashi = get_rashi(positions["Moon"])
    jup_rashi = get_rashi(positions["Jupiter"])
    benefics = {"Taurus", "Cancer", "Sagittarius", "Pisces"}
    if moon_rashi in benefics and jup_rashi in benefics:
        return "🟢 Bullish"
    elif moon_rashi in {"Capricorn", "Scorpio"}:
        return "🔴 Bearish"
    return "⚪ Neutral"

def generate_ephemeris(year, hourly=False):
    start_dt = datetime(2024, 8, 10, 0, 0)
    end_dt = datetime(year, 12, 31, 23, 0)

    step = timedelta(hours=1) if hourly else timedelta(days=1)
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
        for planet, speed in MOTION_SPEEDS.items():
            positions[planet] = (positions[planet] + speed * (1 if hourly else 24)) % 360
        dt += step

    df = pd.DataFrame(rows)
    df_year = df[df["DateTime"].str.startswith(str(year))]
    if hourly:
        df_year.to_csv(f"{DATA_PATH}/ephemeris_hourly_{year}.csv", index=False)
    else:
        df_year.to_csv(f"{DATA_PATH}/ephemeris_daily_{year}.csv", index=False)

    summary = df_year.groupby(["Symbol"]).agg({"Trend": lambda x: x.mode()[0]}).reset_index()
    summary["Year"] = year
    if hourly:
        summary.to_csv(f"{DATA_PATH}/summary_hourly_{year}.csv", index=False)
    else:
        summary.to_csv(f"{DATA_PATH}/summary_daily_{year}.csv", index=False)

# Run optimized generation
for y in range(2024, 2033):
    print(f"Generating daily data for {y}...")
    generate_ephemeris(y, hourly=False)

current_year = datetime.now().year
print(f"Generating hourly data for {current_year}...")
generate_ephemeris(current_year, hourly=True)

print("✅ All CSV files generated in 'data/' folder (daily for all years, hourly for current year).")
