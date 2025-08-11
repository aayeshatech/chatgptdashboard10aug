
# astro_transit_app.py
# Vedic Sidereal Transits – Defensive version (no Telegram).

import math
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
import streamlit as st

SWISSEPH_AVAILABLE = True
try:
    import swisseph as swe
except Exception as e:
    SWISSEPH_AVAILABLE = False
    swe = None
    _import_err = e

PLANETS = [
    ('Sun', swe.SUN if SWISSEPH_AVAILABLE else 0),
    ('Moon', swe.MOON if SWISSEPH_AVAILABLE else 1),
    ('Mercury', swe.MERCURY if SWISSEPH_AVAILABLE else 2),
    ('Venus', swe.VENUS if SWISSEPH_AVAILABLE else 3),
    ('Mars', swe.MARS if SWISSEPH_AVAILABLE else 4),
    ('Jupiter', swe.JUPITER if SWISSEPH_AVAILABLE else 5),
    ('Saturn', swe.SATURN if SWISSEPH_AVAILABLE else 6),
    ('Rahu', swe.MEAN_NODE if SWISSEPH_AVAILABLE else 7),
    ('Ketu', -1),
]

ZODIAC_SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu","Pushya","Ashlesha",
    "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha",
    "Jyeshtha","Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
    "Purva Bhadrapada","Uttara Bhadrapada","Revati"
]
NAK_DEG = 360.0 / 27.0
ASPECTS = {0:"Conjunction",60:"Sextile",90:"Square",120:"Trine",180:"Opposition"}
USE_MOSEPH = False

# ---------- helpers ----------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def normalize_angle(a):
    a = a % 360.0
    if a < 0: a += 360.0
    return a

def min_angle_diff(a, b):
    if a is None or b is None:
        return None
    d = abs(normalize_angle(a) - normalize_angle(b))
    return d if d <= 180 else 360 - d

def ecl_to_sign_deg(longitude):
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
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)

def _extract_calc_array(out):
    """Handle forms like: (xx,), (xx, ret), (ret, xx), or flat sequences. Return a list."""
    if isinstance(out, (list, tuple)):
        if len(out) == 2 and isinstance(out[0], (list, tuple)) and not isinstance(out[1], (list, tuple)):
            # (xx, ret/err)
            return list(out[0])
        if len(out) == 2 and isinstance(out[1], (list, tuple)) and not isinstance(out[0], (list, tuple)):
            # (ret/err, xx)
            return list(out[1])
        if len(out) == 1 and isinstance(out[0], (list, tuple)):
            return list(out[0])
        return list(out)
    # unknown type; try attributes
    possible = []
    for k in ("longitude","latitude","distance","longitude_speed","lat_speed","dist_speed"):
        if hasattr(out, k):
            possible.append(getattr(out, k))
    return possible if possible else [None, None, None, None]

def _calc_ut_standardized(jd_ut, body, use_moseph=False):
    flag_base = swe.FLG_MOSEPH if use_moseph else swe.FLG_SWIEPH
    flag = flag_base | swe.FLG_SPEED
    out = swe.calc_ut(jd_ut, body, flag)
    arr = _extract_calc_array(out)
    # grab first 4 numeric-like values
    lon = safe_float(arr[0], None) if len(arr) > 0 else None
    lat = safe_float(arr[1], None) if len(arr) > 1 else None
    dist = safe_float(arr[2], None) if len(arr) > 2 else None
    speed_lon = safe_float(arr[3], 0.0) if len(arr) > 3 else 0.0
    return lon, lat, dist, speed_lon

def _try_calc_ut(jd_ut, body):
    global USE_MOSEPH
    lon, lat, dist, speed = _calc_ut_standardized(jd_ut, body, use_moseph=USE_MOSEPH)
    # if lon is None (bad parse), switch to MOSEPH and retry
    if lon is None:
        USE_MOSEPH = True
        lon, lat, dist, speed = _calc_ut_standardized(jd_ut, body, use_moseph=True)
    return lon, lat, dist, speed

def sidereal_longitude(body, jd_ut, ayanamsa):
    if body == -1:
        ra = sidereal_longitude(swe.MEAN_NODE, jd_ut, ayanamsa)
        return normalize_angle(ra + 180.0)
    lon, lat, dist, speed = _try_calc_ut(jd_ut, body)
    if lon is None:
        return None
    ay = swe.get_ayanamsa_ut(jd_ut)
    return normalize_angle(lon - ay)

# --- core computations ---
def planet_positions_for_date(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI):
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph not available")
    swe.set_sid_mode(ayanamsa_mode, 0, 0)
    try:
        swe.set_ephe_path("/usr/share/ephe")
    except Exception:
        pass
    dt_noon_local = datetime(date_local.year, date_local.month, date_local.day, 12, 0, 0)
    dt_noon_utc = to_utc(dt_noon_local, tzname)
    jd = julday_from_dt(dt_noon_utc)
    rows = []
    for name, pid in PLANETS:
        lon = sidereal_longitude(pid, jd, ayanamsa_mode)
        if lon is None:
            rows.append({"Planet": name, "Longitude": None, "Sign": "N/A", "Deg°": "N/A", "Nakshatra": "N/A"})
            continue
        sign, deg_in_sign = ecl_to_sign_deg(lon)
        nak, pada = nakshatra_for(lon)
        rows.append({"Planet": name, "Longitude": round(lon, 4), "Sign": sign,
                     "Deg°": deg_to_dms(deg_in_sign), "Nakshatra": f"{nak}-{pada}"})
    return pd.DataFrame(rows)

def detect_aspects(positions, orb_major=3.0, orb_moon=6.0):
    # work only with rows that have numeric longitude
    pos = positions[pd.to_numeric(positions["Longitude"], errors="coerce").notna()].reset_index(drop=True)
    aspects = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            p1 = pos.iloc[i]; p2 = pos.iloc[j]
            diff = min_angle_diff(p1["Longitude"], p2["Longitude"])
            if diff is None: 
                continue
            for exact, name in ASPECTS.items():
                orb = orb_moon if ("Moon" in (p1["Planet"], p2["Planet"])) else orb_major
                if abs(diff - exact) <= orb:
                    aspects.append({"Planet A": p1["Planet"], "Planet B": p2["Planet"],
                                    "Aspect": name, "Exact°": exact, "Deviation°": round(diff-exact,3)})
    return pd.DataFrame(aspects)

# --- UI ---
st.set_page_config(page_title="Vedic Sidereal Transits – Defensive", layout="wide")
st.title("🪐 Vedic Sidereal Transit Explorer — Safe Mode")

if not SWISSEPH_AVAILABLE:
    st.error("pyswisseph not installed here. Install locally: pip install pyswisseph streamlit pytz pandas")
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
swe.set_sid_mode(ay_mode, 0, 0)

st.subheader("Planetary Positions (Sidereal)")
pos_df = planet_positions_for_date(date_in, tzname=tz_in, ayanamsa_mode=ay_mode)
st.dataframe(pos_df, use_container_width=True)

st.subheader("Planetary Aspects (snapshot)")
asp_df = detect_aspects(pos_df, orb_major=3.0, orb_moon=6.0)
if asp_df.empty:
    st.info("No aspects found within default orbs (or some longitudes unavailable).")
else:
    st.dataframe(asp_df, use_container_width=True)

if USE_MOSEPH:
    st.caption("Note: MOSEPH fallback is active. For best precision, configure Swiss ephemeris files via swe.set_ephe_path().")
