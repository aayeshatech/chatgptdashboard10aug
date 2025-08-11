
# astro_transit_app.py
# Streamlit app: Vedic Sidereal Astro Transit + Aspects + Moon Timeline (+ Ingress + Telegram format)
# Robust ephemeris handling (fallback to MOSEPH if Swiss ephemeris files are missing) + calc_ut return-shape fix.

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

# ---- Robust Swiss Ephemeris flags ----
USE_MOSEPH = False  # will be set True if we detect missing ephemeris files

def _calc_ut_standardized(jd_ut, body, use_moseph=False):
    """Call swe.calc_ut and normalize return shape across versions.
    Returns (lon, lat, dist, speed_lon).
    """
    flag_base = swe.FLG_MOSEPH if use_moseph else swe.FLG_SWIEPH
    flag = flag_base | swe.FLG_SPEED
    out = swe.calc_ut(jd_ut, body, flag)
    # pyswisseph versions may return 3, 4, or 6 numbers; normalize to 4.
    if isinstance(out, (list, tuple)):
        if len(out) >= 4:
            lon, lat, dist, speed_lon = out[0], out[1], out[2], out[3]
        elif len(out) == 3:
            lon, lat, dist = out
            speed_lon = 0.0
        else:
            # unexpected; pad
            vals = list(out) + [0.0, 0.0, 0.0, 0.0]
            lon, lat, dist, speed_lon = vals[0], vals[1], vals[2], vals[3]
    else:
        # very old wrapper returning object; try attributes
        lon = getattr(out, 'longitude', 0.0)
        lat = getattr(out, 'latitude', 0.0)
        dist = getattr(out, 'distance', 1.0)
        speed_lon = getattr(out, 'longitude_speed', 0.0)
    return float(lon), float(lat), float(dist), float(speed_lon)

def _try_calc_ut(jd_ut, body):
    """Try SWIEPH, then fallback to MOSEPH (built-in), to avoid file path issues
    and normalize output to 4-tuple.
    """
    global USE_MOSEPH
    try:
        return _calc_ut_standardized(jd_ut, body, use_moseph=USE_MOSEPH)
    except Exception:
        # Switch to MOSEPH and try again
        USE_MOSEPH = True
        return _calc_ut_standardized(jd_ut, body, use_moseph=True)

def normalize_angle(a):
    a = a % 360.0
    if a < 0: a += 360.0
    return a

def min_angle_diff(a, b):
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

def sidereal_longitude(body, jd_ut, ayanamsa):
    if body == -1:
        ra = sidereal_longitude(swe.MEAN_NODE, jd_ut, ayanamsa)
        return normalize_angle(ra + 180.0)
    lon, lat, dist, speed = _try_calc_ut(jd_ut, body)
    # sidereal conversion
    ay = swe.get_ayanamsa_ut(jd_ut)
    return normalize_angle(lon - ay)

def planet_positions_for_date(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI):
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
        sign, deg_in_sign = ecl_to_sign_deg(lon)
        nak, pada = nakshatra_for(lon)
        rows.append({"Planet": name, "Longitude": round(lon, 4), "Sign": sign,
                     "Deg°": deg_to_dms(deg_in_sign), "Nakshatra": f"{nak}-{pada}"})
    return pd.DataFrame(rows)

def detect_aspects(positions, orb_major=3.0, orb_moon=6.0):
    aspects = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            p1 = positions.iloc[i]; p2 = positions.iloc[j]
            diff = min_angle_diff(p1["Longitude"], p2["Longitude"])
            for exact, name in ASPECTS.items():
                orb = orb_moon if ("Moon" in (p1["Planet"], p2["Planet"])) else orb_major
                if abs(diff - exact) <= orb:
                    aspects.append({"Planet A": p1["Planet"], "Planet B": p2["Planet"],
                                    "Aspect": name, "Exact°": exact, "Deviation°": round(diff-exact,3)})
    return pd.DataFrame(aspects)

def refine_exact_time(body_fast, body_slow, target_angle, start_utc, tzname, ay_mode, tol_arcmin=1/60, max_iter=30):
    left = start_utc - timedelta(hours=12)
    right = start_utc + timedelta(hours=12)

    def angle_at(t):
        jd = julday_from_dt(t)
        l_fast = sidereal_longitude(body_fast, jd, ay_mode)
        l_slow = sidereal_longitude(body_slow, jd, ay_mode)
        d = normalize_angle(l_fast - l_slow)
        d = d if d <= 180 else 360 - d
        return d

    a_left = angle_at(left); a_right = angle_at(right)
    for _ in range(max_iter):
        mid = left + (right - left)/2
        a_mid = angle_at(mid)
        if abs(a_mid - target_angle) <= tol_arcmin:
            return to_local(mid, tzname)
        if abs(a_left - target_angle) < abs(a_right - target_angle):
            right = mid; a_right = a_mid
        else:
            left = mid; a_left = a_mid
    return to_local(left + (right-left)/2, tzname)

def moon_aspect_timeline(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI,
                         orb_moon=6.0, step_minutes=15):
    swe.set_sid_mode(ayanamsa_mode, 0, 0)
    try:
        swe.set_ephe_path("/usr/share/ephe")
    except Exception:
        pass
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname); end_utc = to_utc(end_local, tzname)

    targets = [('Sun', swe.SUN), ('Mercury', swe.MERCURY), ('Venus', swe.VENUS),
               ('Mars', swe.MARS), ('Jupiter', swe.JUPITER), ('Saturn', swe.SATURN),
               ('Rahu', swe.MEAN_NODE), ('Ketu', -1)]
    events = []

    cur = t_utc
    seen = set()
    while cur < end_utc:
        jd = julday_from_dt(cur)
        lon_moon = sidereal_longitude(swe.MOON, jd, ayanamsa_mode)
        for name, pid in targets:
            lon_other = sidereal_longitude(pid, jd, ayanamsa_mode)
            diff = min_angle_diff(lon_moon, lon_other)
            for exact, asp_name in ASPECTS.items():
                if abs(diff - exact) <= orb_moon:
                    key = (asp_name, name)
                    if key not in seen:
                        exact_local = refine_exact_time(swe.MOON, pid, exact, cur, tzname, ayanamsa_mode)
                        events.append({"Time": exact_local.strftime("%Y-%m-%d %H:%M"),
                                       "Event": f"Moon {asp_name} {name}", "Exact°": exact})
                        seen.add(key)
        cur += timedelta(minutes=step_minutes)
    return pd.DataFrame(sorted(events, key=lambda x: x["Time"]))

def scan_ingress(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI, step_minutes=10):
    swe.set_sid_mode(ayanamsa_mode, 0, 0)
    try:
        swe.set_ephe_path("/usr/share/ephe")
    except Exception:
        pass
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname); end_utc = to_utc(end_local, tzname)

    def moon_lon(t): return sidereal_longitude(swe.MOON, julday_from_dt(t), ayanamsa_mode)

    events = []
    cur = t_utc
    lon0 = moon_lon(cur)
    sign0, _ = ecl_to_sign_deg(lon0)
    nak0, pada0 = nakshatra_for(lon0)

    while cur < end_utc:
        nxt = cur + timedelta(minutes=step_minutes)
        lon = moon_lon(nxt)
        sign, _ = ecl_to_sign_deg(lon)
        nak, pada = nakshatra_for(lon)

        # Sign change refine
        if sign != sign0:
            lo, hi = cur, nxt
            for _ in range(24):
                mid = lo + (hi - lo)/2
                s_mid, _ = ecl_to_sign_deg(moon_lon(mid))
                if s_mid == sign0: lo = mid
                else: hi = mid
            events.append({"Time": to_local(hi, tzname).strftime("%Y-%m-%d %H:%M"),
                           "Event": f"Moon enters {sign}"})
            sign0 = sign

        # Nakshatra change refine
        if nak != nak0:
            lo, hi = cur, nxt
            for _ in range(24):
                mid = lo + (hi - lo)/2
                n_mid, _ = nakshatra_for(moon_lon(mid))
                if n_mid == nak0: lo = mid
                else: hi = mid
            events.append({"Time": to_local(hi, tzname).strftime("%Y-%m-%d %H:%M"),
                           "Event": f"Moon enters {nak}"})
            nak0 = nak

        cur = nxt

    return pd.DataFrame(sorted(events, key=lambda x: x["Time"]))

def telegram_format(date_str, tzname, pos_df, asp_df, moon_df, ingress_df):
    lines = []
    lines.append(f"✨ **Astro Transit** ({date_str}, {tzname})")
    lines.append("")
    lines.append("🪐 *Positions* (Sidereal, noon):")
    for _, r in pos_df.iterrows():
        lines.append(f"• {r['Planet']}: {r['Sign']} {r['Deg°']} ({r['Nakshatra']})")

    if not asp_df.empty:
        lines.append("")
        lines.append("🔭 *Aspects*:")
        for _, r in asp_df.iterrows():
            lines.append(f"• {r['Planet A']} {r['Aspect']} {r['Planet B']} (±{abs(r['Deviation°']):.2f}°)")

    if not ingress_df.empty:
        lines.append("")
        lines.append("🌙 *Moon Ingress*:")
        for _, r in ingress_df.iterrows():
            lines.append(f"• {r['Time']} – {r['Event']}")

    if not moon_df.empty:
        lines.append("")
        lines.append("🕒 *Moon Aspects Timeline*:")
        for _, r in moon_df.iterrows():
            lines.append(f"• {r['Time']} – {r['Event']} ({r['Exact°']}°)")

    return "\n".join(lines)

# -------- UI --------
st.set_page_config(page_title="Vedic Sidereal Transits – Date, Aspects, Moon Timeline", layout="wide")
st.title("🪐 Vedic Sidereal Transit Explorer")
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
swe.set_sid_mode(ay_mode, 0, 0)

col1, col2 = st.columns(2)
with col1:
    orb_major = st.slider("Orb for aspects (most planets) [°]", 1.0, 6.0, 3.0, 0.5)
with col2:
    orb_moon = st.slider("Orb for Moon aspects [°]", 2.0, 10.0, 6.0, 0.5)

st.markdown("---")
st.subheader("Planetary Positions (Sidereal)")
pos_df = planet_positions_for_date(date_in, tzname=tz_in, ayanamsa_mode=ay_mode)
st.dataframe(pos_df, use_container_width=True)

st.subheader("Planetary Aspects (snapshot around local noon)")
asp_df = detect_aspects(pos_df, orb_major=orb_major, orb_moon=orb_moon)
if asp_df.empty:
    st.info("No major aspects within selected orbs at the snapshot time.")
else:
    st.dataframe(asp_df.sort_values(by=["Aspect","Planet A"]), use_container_width=True)

st.subheader("🌙 Moon Aspects Timeline (exact times)")
with st.spinner("Computing Moon aspects across the day..."):
    moon_df = moon_aspect_timeline(date_in, tzname=tz_in, ayanamsa_mode=ay_mode, orb_moon=orb_moon, step_minutes=15)
st.dataframe(moon_df, use_container_width=True)

st.subheader("🌗 Moon Ingress (Sign & Nakshatra)")
with st.spinner("Scanning ingress events..."):
    ingress_df = scan_ingress(date_in, tzname=tz_in, ayanamsa_mode=ay_mode, step_minutes=10)
st.dataframe(ingress_df, use_container_width=True)

# Downloads
csv1 = pos_df.to_csv(index=False).encode()
csv2 = asp_df.to_csv(index=False).encode()
csv3 = moon_df.to_csv(index=False).encode()
csv4 = ingress_df.to_csv(index=False).encode()
colD, colE, colF, colG = st.columns(4)
with colD: st.download_button("Download Positions CSV", csv1, file_name=f"positions_{date_in}.csv")
with colE: st.download_button("Download Aspects CSV", csv2, file_name=f"aspects_{date_in}.csv")
with colF: st.download_button("Download Moon Timeline CSV", csv3, file_name=f"moon_timeline_{date_in}.csv")
with colG: st.download_button("Download Moon Ingress CSV", csv4, file_name=f"moon_ingress_{date_in}.csv")

# Telegram formatter
st.markdown("---")
st.subheader("📨 Telegram Alert Preview")
msg = telegram_format(str(date_in), tz_in, pos_df, asp_df, moon_df, ingress_df)
st.text_area("Formatted message (copy/paste to your bot):", value=msg, height=300)

if USE_MOSEPH:
    st.info("Running with MOSEPH (built-in ephemeris). For higher precision, download Swiss ephemeris files and set swe.set_ephe_path(path).")
