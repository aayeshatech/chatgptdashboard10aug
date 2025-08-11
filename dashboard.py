
# astro_transit_app.py
# Streamlit app: Vedic Sidereal Astro Transit + Aspects + Moon Timeline
# Author: ChatGPT
#
# Requirements (install locally):
#   pip install streamlit pyswisseph pytz pandas
# Run:
#   streamlit run astro_transit_app.py
#
# Notes:
# - Uses Lahiri ayanamsa by default.
# - Finds aspects (0, 60, 90, 120, 180) with user-set orbs.
# - Computes Moon aspect *exact times* for the selected date using a fast search.
# - Times are shown in the chosen timezone (default: Asia/Kolkata).

import math
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd

import streamlit as st

# Try to import Swiss Ephemeris (preferred). If missing, show a helpful message.
SWISSEPH_AVAILABLE = True
try:
    import swisseph as swe
except Exception as e:
    SWISSEPH_AVAILABLE = False
    swe = None
    _import_err = e

# ------------- Constants -------------
PLANETS = [
    ('Sun', swe.SUN if SWISSEPH_AVAILABLE else 0),
    ('Moon', swe.MOON if SWISSEPH_AVAILABLE else 1),
    ('Mercury', swe.MERCURY if SWISSEPH_AVAILABLE else 2),
    ('Venus', swe.VENUS if SWISSEPH_AVAILABLE else 3),
    ('Mars', swe.MARS if SWISSEPH_AVAILABLE else 4),
    ('Jupiter', swe.JUPITER if SWISSEPH_AVAILABLE else 5),
    ('Saturn', swe.SATURN if SWISSEPH_AVAILABLE else 6),
    ('Rahu', swe.MEAN_NODE if SWISSEPH_AVAILABLE else 7),   # Mean Node (Rahu)
    ('Ketu', -1),  # derived = Rahu + 180
]

ZODIAC_SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu","Pushya","Ashlesha",
    "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha",
    "Jyeshtha","Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
    "Purva Bhadrapada","Uttara Bhadrapada","Revati"
]
NAK_DEG = 360.0 / 27.0  # 13°20' = 13.333...

ASPECTS = {
    0: "Conjunction",
    60: "Sextile",
    90: "Square",
    120: "Trine",
    180: "Opposition"
}

FAST_BODIES = {"Moon","Mercury","Venus","Sun","Mars"}

# ------------- Utilities -------------

def normalize_angle(a):
    a = a % 360.0
    if a < 0: a += 360.0
    return a

def min_angle_diff(a, b):
    """Return smallest angle difference between two longitudes (0-180)."""
    d = abs(normalize_angle(a) - normalize_angle(b))
    return d if d <= 180 else 360 - d

def ecl_to_sign_deg(longitude):
    """Return (sign, deg_in_sign)"""
    lon = normalize_angle(longitude)
    sign_index = int(lon // 30)
    deg_in_sign = lon - sign_index * 30
    return ZODIAC_SIGNS[sign_index], deg_in_sign

def deg_to_dms(deg):
    d = int(deg)
    m_float = abs(deg - d) * 60
    m = int(m_float)
    s = int(round((m_float - m) * 60))
    return f"{d:02d}°{m:02d}'{s:02d}\""

def nakshatra_for(longitude):
    lon = normalize_angle(longitude)
    idx = int(lon // NAK_DEG)
    pada = int(((lon % NAK_DEG) / NAK_DEG) * 4) + 1
    return NAKSHATRAS[idx], pada

def to_utc(dt_local, tzname):
    tz = pytz.timezone(tzname)
    return tz.localize(dt_local).astimezone(pytz.utc)

def to_local(dt_utc, tzname):
    tz = pytz.timezone(tzname)
    return dt_utc.astimezone(tz)

def julday_from_dt(dt_utc):
    """UTC datetime -> Julian Day using swe.julday"""
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)

def sidereal_longitude(body, jd_ut, ayanamsa):
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    lon, lat, dist, speed_long = 0.0, 0.0, 0.0, 0.0
    if body == -1:
        # Ketu = Rahu + 180
        ra = sidereal_longitude(swe.MEAN_NODE, jd_ut, ayanamsa)
        return normalize_angle(ra + 180.0)
    lon, lat, dist, speed = swe.calc_ut(jd_ut, body, flag)
    # convert to sidereal by subtracting ayanamsa
    ay = swe.get_ayanamsa_ut(jd_ut) if ayanamsa is None else swe.get_ayanamsa_ut(jd_ut)
    sid_lon = normalize_angle(lon - ay)
    return sid_lon

def planet_positions_for_date(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI):
    """Return DataFrame of positions and metadata for given local date (00:00 local)."""
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph not available. Please install with: pip install pyswisseph")

    # set sidereal mode
    swe.set_sid_mode(ayanamsa_mode, 0, 0)

    # use 12:00 local noon to reduce retro anomalies; display positions at local noon
    dt_noon_local = datetime(date_local.year, date_local.month, date_local.day, 12, 0, 0)
    dt_noon_utc = to_utc(dt_noon_local, tzname)
    jd = julday_from_dt(dt_noon_utc)

    rows = []
    for name, pid in PLANETS:
        lon = sidereal_longitude(pid, jd, ayanamsa_mode)
        sign, deg_in_sign = ecl_to_sign_deg(lon)
        nak, pada = nakshatra_for(lon)
        rows.append({
            "Planet": name,
            "Longitude": round(lon, 4),
            "Sign": sign,
            "Deg°": deg_to_dms(deg_in_sign),
            "Nakshatra": f"{nak}-{pada}"
        })
    return pd.DataFrame(rows)

def detect_aspects(positions, orb_major=3.0, orb_moon=6.0):
    """Return list of aspects present at the snapshot positions."""
    aspects = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            p1 = positions.iloc[i]
            p2 = positions.iloc[j]
            if p1["Planet"] == "Ketu" or p2["Planet"] == "Ketu":
                # Ketu handled via longitude already; proceed
                pass
            diff = min_angle_diff(p1["Longitude"], p2["Longitude"])
            for exact, name in ASPECTS.items():
                orb = orb_moon if ("Moon" in (p1["Planet"], p2["Planet"])) else orb_major
                if abs(diff - exact) <= orb:
                    aspects.append({
                        "Planet A": p1["Planet"],
                        "Planet B": p2["Planet"],
                        "Aspect": name,
                        "Exact°": exact,
                        "Deviation°": round(diff - exact, 3)
                    })
    return pd.DataFrame(aspects)

def refine_exact_time(body_fast, body_slow, target_angle, start_utc, tzname, ay_mode, tol_arcmin=1/60, max_iter=30):
    """Binary-search exact time when angle reaches target_angle.
       Returns local time.
    """
    # initial bracket: search +/- 12 hours around start_utc
    left = start_utc - timedelta(hours=12)
    right = start_utc + timedelta(hours=12)

    def angle_at(t):
        jd = julday_from_dt(t)
        l_fast = sidereal_longitude(body_fast, jd, ay_mode)
        l_slow = sidereal_longitude(body_slow, jd, ay_mode)
        d = normalize_angle(l_fast - l_slow)
        d = d if d <= 180 else 360 - d
        return d

    # Move left forward until angle crosses target or we exhaust
    a_left = angle_at(left)
    a_right = angle_at(right)

    # If no crossing, just do simple iterative approach from left to right
    for _ in range(max_iter):
        mid = left + (right - left)/2
        a_mid = angle_at(mid)
        # stop when within tolerance
        if abs(a_mid - target_angle) <= tol_arcmin:
            return to_local(mid, tzname)
        # Choose side that brings closer (not strictly monotonic; heuristic):
        if abs(a_left - target_angle) < abs(a_right - target_angle):
            right = mid; a_right = a_mid
        else:
            left = mid; a_left = a_mid
    return to_local(left + (right-left)/2, tzname)

def moon_aspect_timeline(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI,
                         orb_major=3.0, orb_moon=6.0, step_minutes=10):
    """Scan the selected day in steps and find Moon aspects + exact times."""
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph not available. Please install with: pip install pyswisseph")

    swe.set_sid_mode(ayanamsa_mode, 0, 0)

    # day start/end in local then convert to UTC
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname)
    end_utc = to_utc(end_local, tzname)

    events = []

    # targets: Moon with Sun, Mercury, Venus, Mars, Jupiter, Saturn, Rahu, Ketu
    targets = [
        ('Sun', swe.SUN), ('Mercury', swe.MERCURY), ('Venus', swe.VENUS),
        ('Mars', swe.MARS), ('Jupiter', swe.JUPITER), ('Saturn', swe.SATURN),
        ('Rahu', swe.MEAN_NODE), ('Ketu', -1)
    ]

    cur = t_utc
    prev_diffs = {}

    while cur < end_utc:
        jd = julday_from_dt(cur)
        lon_moon = sidereal_longitude(swe.MOON, jd, ayanamsa_mode)

        for name, pid in targets:
            lon_other = sidereal_longitude(pid, jd, ayanamsa_mode)
            diff = min_angle_diff(lon_moon, lon_other)
            for exact, asp_name in ASPECTS.items():
                orb = orb_moon  # Moon aspects use moon orb
                if abs(diff - exact) <= orb:
                    # attempt refine exact time around 'cur'
                    local_guess = to_local(cur, tzname)
                    exact_local = refine_exact_time(swe.MOON, pid, exact, cur, tzname, ayanamsa_mode)
                    events.append({
                        "Time": exact_local.strftime("%Y-%m-%d %H:%M"),
                        "Moon Aspect": f"Moon {asp_name} {name}",
                        "Exact°": exact
                    })
        cur += timedelta(minutes=step_minutes)

    # Deduplicate near-duplicates (same aspect within ~45 min)
    events_sorted = sorted(events, key=lambda x: x["Time"])
    deduped = []
    last_key = {}
    for e in events_sorted:
        key = e["Moon Aspect"]
        if key not in last_key:
            deduped.append(e); last_key[key] = e
        else:
            # keep the first occurrence that day
            continue

    return pd.DataFrame(deduped)

# ------------- UI -------------

st.set_page_config(page_title="Vedic Sidereal Transits & Moon Timeline", layout="wide")

st.title("🪐 Vedic Sidereal Transit Explorer (Date Input + Aspects + Moon Timeline)")

if not SWISSEPH_AVAILABLE:
    st.error(
        "pyswisseph (Swiss Ephemeris) is not installed in this environment.\n\n"
        f"Technical details: {_import_err}\n\n"
        "To run locally: `pip install pyswisseph streamlit pytz pandas` and then `streamlit run astro_transit_app.py`."
    )
    st.stop()

colA, colB, colC = st.columns(3)
with colA:
    date_in = st.date_input("Select Date", value=pd.Timestamp.today().date())
with colB:
    tz_in = st.text_input("Time Zone (IANA)", value="Asia/Kolkata")
with colC:
    ay_choice = st.selectbox("Ayanamsa", ["Lahiri (default)","Raman","Krishnamurti","True Citra"])
ayanamsa_map = {
    "Lahiri (default)": swe.SIDM_LAHIRI,
    "Raman": swe.SIDM_RAMAN,
    "Krishnamurti": swe.SIDM_KRISHNAMURTI,
    "True Citra": swe.SIDM_TRUE_CITRA
}
ay_mode = ayanamsa_map[ay_choice]

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    orb_major = st.slider("Orb for aspects (most planets) [°]", 1.0, 6.0, 3.0, 0.5)
with col2:
    orb_moon = st.slider("Orb for Moon aspects [°]", 2.0, 10.0, 6.0, 0.5)

# Positions
st.subheader("Planetary Positions (Sidereal)")
pos_df = planet_positions_for_date(date_in, tzname=tz_in, ayanamsa_mode=ay_mode)
st.dataframe(pos_df, use_container_width=True)

# Aspects (snapshot around local noon)
st.subheader("Planetary Aspects (snapshot)")
asp_df = detect_aspects(pos_df, orb_major=orb_major, orb_moon=orb_moon)
if asp_df.empty:
    st.info("No major aspects within selected orbs at the snapshot time.")
else:
    st.dataframe(asp_df.sort_values(by=["Aspect","Planet A"]), use_container_width=True)

# Moon Timeline
st.subheader("🌓 Moon Aspect Timeline (Exact times for the selected day)")
with st.spinner("Calculating Moon aspects across the day..."):
    moon_df = moon_aspect_timeline(date_in, tzname=tz_in, ayanamsa_mode=ay_mode,
                                   orb_major=orb_major, orb_moon=orb_moon, step_minutes=15)
if moon_df.empty:
    st.info("No major Moon aspects found for the day within the chosen orb.")
else:
    st.dataframe(moon_df, use_container_width=True)

# Download buttons
csv1 = pos_df.to_csv(index=False).encode()
csv2 = asp_df.to_csv(index=False).encode()
csv3 = moon_df.to_csv(index=False).encode() if not moon_df.empty else b""

st.download_button("Download Positions CSV", csv1, file_name=f"positions_{date_in}.csv")
st.download_button("Download Aspects CSV", csv2, file_name=f"aspects_{date_in}.csv")
if csv3:
    st.download_button("Download Moon Timeline CSV", csv3, file_name=f"moon_timeline_{date_in}.csv")

st.caption("Tip: Use the CSVs to feed your Telegram alert formatter or TradingView webhook logic.")
