
# astro_transit_app.py
# Vedic Sidereal Transits — Robust + Full Aspect Timelines + Moon KP details + Data Analysis (on-screen only)

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

# KP
DASHA_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
TOTAL_YEARS = 120.0
NAK_LORD = {
    "Ashwini":"Ketu","Bharani":"Venus","Krittika":"Sun","Rohini":"Moon","Mrigashira":"Mars","Ardra":"Rahu",
    "Punarvasu":"Jupiter","Pushya":"Saturn","Ashlesha":"Mercury",
    "Magha":"Ketu","Purva Phalguni":"Venus","Uttara Phalguni":"Sun",
    "Hasta":"Moon","Chitra":"Mars","Swati":"Rahu","Vishakha":"Jupiter","Anuradha":"Saturn","Jyeshtha":"Mercury",
    "Mula":"Ketu","Purva Ashadha":"Venus","Uttara Ashadha":"Sun",
    "Shravana":"Moon","Dhanishta":"Mars","Shatabhisha":"Rahu",
    "Purva Bhadrapada":"Jupiter","Uttara Bhadrapada":"Saturn","Revati":"Mercury"
}

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

def kp_subsequence(start_lord):
    idx = DASHA_ORDER.index(start_lord)
    return DASHA_ORDER[idx:] + DASHA_ORDER[:idx]

def kp_sublord_of_longitude(lon):
    nak, pada = nakshatra_for(lon)
    star_lord = NAK_LORD[nak]
    seq = kp_subsequence(star_lord)
    within = (normalize_angle(lon) % NAK_DEG) / NAK_DEG  # 0..1 within the star
    cum = 0.0
    for lord in seq:
        frac = DASHA_YEARS[lord] / TOTAL_YEARS
        if within < cum + frac:
            return nak, pada, star_lord, lord
        cum += frac
    return nak, pada, star_lord, seq[-1]

def to_utc(dt_local, tzname):
    tz = pytz.timezone(tzname)
    return tz.localize(dt_local).astimezone(pytz.utc)

def to_local(dt_utc, tzname):
    tz = pytz.timezone(tzname)
    return dt_utc.astimezone(tz)

def julday_from_dt(dt_utc):
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)

def _extract_calc_array(out):
    if isinstance(out, (list, tuple)):
        if len(out) == 2 and isinstance(out[0], (list, tuple)) and not isinstance(out[1], (list, tuple)):
            return list(out[0])
        if len(out) == 2 and isinstance(out[1], (list, tuple)) and not isinstance(out[0], (list, tuple)):
            return list(out[1])
        if len(out) == 1 and isinstance(out[0], (list, tuple)):
            return list(out[0])
        return list(out)
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
    lon = safe_float(arr[0], None) if len(arr) > 0 else None
    lat = safe_float(arr[1], None) if len(arr) > 1 else None
    dist = safe_float(arr[2], None) if len(arr) > 2 else None
    speed_lon = safe_float(arr[3], 0.0) if len(arr) > 3 else 0.0
    return lon, lat, dist, speed_lon

def _try_calc_ut(jd_ut, body):
    global USE_MOSEPH
    lon, lat, dist, speed = _calc_ut_standardized(jd_ut, body, use_moseph=USE_MOSEPH)
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

# ---- Timelines ----
def refine_exact_time(body_a, body_b, target_angle, start_utc, tzname, ay_mode, tol_deg=1/60, max_iter=28):
    left = start_utc - timedelta(hours=6)
    right = start_utc + timedelta(hours=6)
    swe.set_sid_mode(ay_mode, 0, 0)

    def angle_at(t):
        jd = julday_from_dt(t)
        la = sidereal_longitude(body_a, jd, ay_mode)
        lb = sidereal_longitude(body_b, jd, ay_mode)
        if la is None or lb is None:
            return None
        d = normalize_angle(la - lb)
        d = d if d <= 180 else 360 - d
        return d

    a_left = angle_at(left)
    a_right = angle_at(right)
    if a_left is None or a_right is None:
        return to_local(start_utc, tzname)

    for _ in range(max_iter):
        mid = left + (right - left)/2
        a_mid = angle_at(mid)
        if a_mid is None:
            mid += timedelta(minutes=2)
            a_mid = angle_at(mid)
            if a_mid is None:
                break
        if abs(a_mid - target_angle) <= tol_deg:
            return to_local(mid, tzname)
        if abs(a_left - target_angle) < abs(a_right - target_angle):
            right = mid; a_right = a_mid
        else:
            left = mid; a_left = a_mid
    return to_local(left + (right-left)/2, tzname)

def planetary_aspect_timeline(date_local, tzname="Asia/Kolkata", ay_mode=swe.SIDM_LAHIRI,
                              orb_major=3.0, orb_moon=6.0, step_minutes=20):
    swe.set_sid_mode(ay_mode, 0, 0)
    try:
        swe.set_ephe_path("/usr/share/ephe")
    except Exception:
        pass
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname); end_utc = to_utc(end_local, tzname)

    planet_ids = {name: pid for name, pid in PLANETS}
    names = list(planet_ids.keys())

    events = []
    cur = t_utc
    seen = set()
    while cur < end_utc:
        jd = julday_from_dt(cur)
        longs = {nm: sidereal_longitude(pid, jd, ay_mode) for nm, pid in planet_ids.items()}

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                A, B = names[i], names[j]
                la, lb = longs.get(A), longs.get(B)
                if la is None or lb is None:
                    continue
                diff = min_angle_diff(la, lb)
                for exact, a_name in ASPECTS.items():
                    orb = orb_moon if ("Moon" in (A, B)) else orb_major
                    if diff is not None and abs(diff - exact) <= orb:
                        key = (a_name, A, B)
                        if key in seen:
                            continue
                        exact_local = refine_exact_time(planet_ids[A], planet_ids[B], exact, cur, tzname, ay_mode)
                        # Moon KP info
                        kp_info = {"Nakshatra": "", "Star Lord": "", "Sub-Lord": ""}
                        if A == "Moon" or B == "Moon":
                            jd_exact = julday_from_dt(exact_local.astimezone(pytz.utc))
                            lon_moon = sidereal_longitude(swe.MOON, jd_exact, ay_mode)
                            if lon_moon is not None:
                                nak, pada = nakshatra_for(lon_moon)
                                star, sub = NAK_LORD[nak], kp_sublord_of_longitude(lon_moon)[3]
                                kp_info = {"Nakshatra": f"{nak}-{pada}", "Star Lord": star, "Sub-Lord": sub}
                        events.append({
                            "Time": exact_local.strftime("%Y-%m-%d %H:%M"),
                            "Aspect": a_name,
                            "Exact°": exact,
                            "Planet A": A,
                            "Planet B": B,
                            "Moon Nakshatra@Exact": kp_info["Nakshatra"],
                            "Moon Star Lord@Exact": kp_info["Star Lord"],
                            "Moon Sub-Lord@Exact": kp_info["Sub-Lord"],
                        })
                        seen.add(key)
        cur += timedelta(minutes=step_minutes)

    return pd.DataFrame(sorted(events, key=lambda x: x["Time"]))

def moon_kp_sublord_timeline(date_local, tzname="Asia/Kolkata", ayanamsa_mode=swe.SIDM_LAHIRI, step_minutes=5):
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
    if lon0 is None:
        return pd.DataFrame([])
    nak0, pada0 = nakshatra_for(lon0)
    _, _, _, sub0 = kp_sublord_of_longitude(lon0)

    while cur < end_utc:
        nxt = cur + timedelta(minutes=step_minutes)
        lon = moon_lon(nxt)
        if lon is None:
            cur = nxt
            continue
        _, _, _, sub = kp_sublord_of_longitude(lon)
        if sub != sub0:
            lo, hi = cur, nxt
            for _ in range(24):
                mid = lo + (hi - lo)/2
                lon_mid = moon_lon(mid)
                if lon_mid is None:
                    break
                _, _, _, s_mid = kp_sublord_of_longitude(lon_mid)
                if s_mid == sub0: lo = mid
                else: hi = mid
            local_time = to_local(hi, tzname).strftime("%Y-%m-%d %H:%M")
            nak_mid, pada_mid = nakshatra_for(moon_lon(hi))
            star = NAK_LORD[nak_mid]
            events.append({"Time": local_time, "Nakshatra": f"{nak_mid}-{pada_mid}", "Star Lord": star, "Sub-Lord": sub})
            sub0 = sub
        cur = nxt

    return pd.DataFrame(events)

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
    if lon0 is None:
        return pd.DataFrame([])
    sign0, _ = ecl_to_sign_deg(lon0)
    nak0, _ = nakshatra_for(lon0)

    while cur < end_utc:
        nxt = cur + timedelta(minutes=step_minutes)
        lon = moon_lon(nxt)
        if lon is None:
            cur = nxt
            continue
        sign, _ = ecl_to_sign_deg(lon)
        nak, _ = nakshatra_for(lon)

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

# ========== Data Analysis (Symbol-wise Bullish/Bearish) ==========
DEFAULT_RULES = {
    "weights": {
        "benefics": {"Jupiter": 2.0, "Venus": 1.5, "Moon": 1.0, "Mercury": 0.8},
        "malefics": {"Saturn": -2.0, "Mars": -1.5, "Rahu": -1.5, "Ketu": -1.2},
        "sun": 0.5  # Sun treated mildly positive
    },
    "aspect_multipliers": {
        "Trine": 1.0, "Sextile": 0.8, "Conjunction": 0.6, "Opposition": -0.9, "Square": -1.0
    },
    "asset_bias": {
        "NIFTY": {"Jupiter": +0.5, "Saturn": -0.3, "Mercury": +0.2},
        "BANKNIFTY": {"Jupiter": +0.6, "Saturn": -0.5, "Mercury": +0.3},
        "GOLD": {"Saturn": -0.6, "Jupiter": +0.4, "Venus": +0.2, "Rahu": +0.3},  # flight-to-safety + Venus demand
        "CRUDE": {"Mars": +0.6, "Saturn": -0.2, "Jupiter": +0.2},
        "BTC": {"Rahu": +0.6, "Saturn": -0.4, "Jupiter": +0.2},
        "DOW": {"Jupiter": +0.4, "Saturn": -0.3, "Mercury": +0.2}
    },
    "thresholds": {"bullish": 1.0, "bearish": -1.0}
}

def score_event(row, asset, rules=DEFAULT_RULES):
    A, B, aspect = row["Planet A"], row["Planet B"], row["Aspect"]
    w = rules["weights"]; mult = rules["aspect_multipliers"]; bias_map = rules["asset_bias"]
    # planet base weights
    def p_weight(p):
        if p in ("Sun",): return w["sun"]
        if p in w["benefics"]: return w["benefics"][p]
        if p in w["malefics"]: return w["malefics"][p]
        return 0.0
    s = (p_weight(A) + p_weight(B)) * mult.get(aspect, 0.0)
    # asset bias tweaks
    asset = asset.upper()
    if asset in bias_map:
        s += bias_map[asset].get(A, 0.0) + bias_map[asset].get(B, 0.0)
    return s

def classify_score(score, rules=DEFAULT_RULES):
    if score >= rules["thresholds"]["bullish"]: return "Bullish"
    if score <= rules["thresholds"]["bearish"]: return "Bearish"
    return "Neutral/Volatile"

# --- UI ---
st.set_page_config(page_title="Vedic Sidereal Transits — Timelines + Analysis", layout="wide")
st.title("🪐 Vedic Sidereal Transit Explorer — Full Timelines + Data Analysis")

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
ay_mode = ayansma = ayanamsa_map[ay_choice]  # typo-proof alias
swe.set_sid_mode(ay_mode, 0, 0)

col1, col2 = st.columns(2)
with col1:
    orb_major = st.slider("Orb for aspects (most planets) [°]", 1.0, 6.0, 3.0, 0.5)
with col2:
    orb_moon = st.slider("Orb for Moon aspects [°]", 2.0, 10.0, 6.0, 0.5)

st.markdown("---")
tabs = st.tabs([
    "Positions", "Aspects (snapshot)", "Moon Aspects Timeline", "Moon Ingress",
    "Moon KP Sub-lord", "All Planet Aspect Timeline", "Data Analysis"
])

with tabs[0]:
    st.subheader("Planetary Positions (Sidereal)")
    pos_df = planet_positions_for_date(date_in, tzname=tz_in, ayanamsa_mode=ay_mode)
    st.dataframe(pos_df, use_container_width=True)

with tabs[1]:
    st.subheader("Planetary Aspects (snapshot around local noon)")
    asp_df = detect_aspects(pos_df, orb_major=orb_major, orb_moon=orb_moon)
    if asp_df.empty:
        st.info("No major aspects within selected orbs at the snapshot time.")
    else:
        st.dataframe(asp_df.sort_values(by=["Aspect","Planet A"]), use_container_width=True)

with tabs[2]:
    st.subheader("🌙 Moon Aspects Timeline (exact times + KP at exact moment)")
    with st.spinner("Computing Moon aspects across the day..."):
        day_all = planetary_aspect_timeline(date_in, tzname=tz_in, ay_mode=ay_mode, orb_major=orb_major, orb_moon=orb_moon, step_minutes=20)
        moon_df = day_all[(day_all["Planet A"]=="Moon") | (day_all["Planet B"]=="Moon")].reset_index(drop=True)
    st.dataframe(moon_df, use_container_width=True)

with tabs[3]:
    st.subheader("🌗 Moon Ingress (Sign & Nakshatra)")
    with st.spinner("Scanning ingress events..."):
        ingress_df = scan_ingress(date_in, tzname=tz_in, ayanamsa_mode=ay_mode, step_minutes=10)
    st.dataframe(ingress_df, use_container_width=True)

with tabs[4]:
    st.subheader("🧭 KP Sub-lord Timeline (Moon)")
    with st.spinner("Calculating KP sub-lord changes..."):
        kp_df = moon_kp_sublord_timeline(date_in, tzname=tz_in, ayanamsa_mode=ay_mode, step_minutes=5)
    st.dataframe(kp_df, use_container_width=True)

with tabs[5]:
    st.subheader("🕒 All Planet Aspect Timeline (exact times for all pairs)")
    with st.spinner("Computing aspect timeline for all planets..."):
        asp_timeline_df = planetary_aspect_timeline(date_in, tzname=tz_in, ay_mode=ay_mode, orb_major=orb_major, orb_moon=orb_moon, step_minutes=20)
    st.dataframe(asp_timeline_df, use_container_width=True)

with tabs[6]:
    st.subheader("📊 Data Analysis — Symbol-wise Bullish/Bearish Timeline")
    c1, c2 = st.columns([2,1])
    with c1:
        symbol = st.text_input("Symbol", value="NIFTY")
    with c2:
        asset_class = st.selectbox("Asset Class", ["NIFTY","BANKNIFTY","GOLD","CRUDE","BTC","DOW","OTHER"], index=0)

    with st.spinner("Scoring events for the day..."):
        # ensure we have timeline
        if 'asp_timeline_df' not in locals():
            asp_timeline_df = planetary_aspect_timeline(date_in, tzname=tz_in, ay_mode=ay_mode, orb_major=orb_major, orb_moon=orb_moon, step_minutes=20)
        df = asp_timeline_df.copy()

        # score each row
        scores = []
        for _, r in df.iterrows():
            s = score_event(r, asset_class, DEFAULT_RULES)
            scores.append(s)
        df["Score"] = scores
        df["Signal"] = [classify_score(s, DEFAULT_RULES) for s in scores]
        df["Symbol"] = symbol.upper()

        # Only keep the core view
        view_cols = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B",
                     "Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
        df_view = df[view_cols].sort_values("Time")

    # Show summary counts
    colx, coly, colz = st.columns(3)
    with colx:
        st.metric("Bullish windows", int((df_view["Signal"]=="Bullish").sum()))
    with coly:
        st.metric("Bearish windows", int((df_view["Signal"]=="Bearish").sum()))
    with colz:
        st.metric("Neutral/Volatile", int((df_view["Signal"]=="Neutral/Volatile").sum()))

    st.dataframe(df_view, use_container_width=True)

    st.markdown("**Rules (editable JSON):**")
    rules_json = st.text_area("Tweak weights/aspects/bias/thresholds then press Apply", value=str(DEFAULT_RULES), height=220)
    if st.button("Apply Custom Rules"):
        try:
            import ast
            custom = ast.literal_eval(rules_json)
            scores2 = []
            for _, r in df.iterrows():
                s2 = score_event(r, asset_class, custom)
                scores2.append(s2)
            df["Score"] = scores2
            df["Signal"] = [classify_score(s2, custom) for s2 in scores2]
            df_view2 = df[view_cols].sort_values("Time")
            st.success("Applied custom rules.")
            st.dataframe(df_view2, use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse rules: {e}")

# Downloads
st.markdown("---")
colD, colE, colF, colG, colH, colI, colJ = st.columns(7)
with colD:
    st.download_button("Download Positions CSV", pos_df.to_csv(index=False).encode(), file_name=f"positions_{date_in}.csv")
with colE:
    st.download_button("Download Aspects (snapshot) CSV", asp_df.to_csv(index=False).encode(), file_name=f"aspects_snapshot_{date_in}.csv")
with colF:
    st.download_button("Download Moon Timeline CSV", moon_df.to_csv(index=False).encode(), file_name=f"moon_timeline_{date_in}.csv")
with colG:
    st.download_button("Download Moon Ingress CSV", ingress_df.to_csv(index=False).encode(), file_name=f"moon_ingress_{date_in}.csv")
with colH:
    st.download_button("Download KP Sub-lord CSV", kp_df.to_csv(index=False).encode(), file_name=f"kp_sublord_{date_in}.csv")
with colI:
    st.download_button("Download All Aspects Timeline CSV", asp_timeline_df.to_csv(index=False).encode(), file_name=f"aspects_timeline_{date_in}.csv")
with colJ:
    st.download_button("Download Data Analysis CSV", df_view.to_csv(index=False).encode(), file_name=f"signal_timeline_{symbol}_{date_in}.csv")

if USE_MOSEPH:
    st.caption("Note: MOSEPH fallback is active. For best precision, configure Swiss ephemeris files via swe.set_ephe_path().")
