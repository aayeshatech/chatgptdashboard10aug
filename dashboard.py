
# astro_transit_app.py
# Vedic Sidereal Transits — Strict KP + Highlights + Sector Scanner

import math
from datetime import datetime, timedelta, time as dtime
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

# ---------- Constants ----------
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

SIGN_LORDS = {
    "Aries":"Mars","Taurus":"Venus","Gemini":"Mercury","Cancer":"Moon","Leo":"Sun","Virgo":"Mercury",
    "Libra":"Venus","Scorpio":"Mars","Sagittarius":"Jupiter","Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter"
}

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

# ---------- Helpers ----------
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
    if a is None or b is None: return None
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
    within = (normalize_angle(lon) % NAK_DEG) / NAK_DEG
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
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day,
                      dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)

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

def _calc_ut_standardized(jd_ut, body, flag_extra=0, use_moseph=False):
    flag_base = swe.FLG_MOSEPH if use_moseph else swe.FLG_SWIEPH
    flag = flag_base | swe.FLG_SPEED | flag_extra
    out = swe.calc_ut(jd_ut, body, flag)
    arr = _extract_calc_array(out)
    lon = safe_float(arr[0], None) if len(arr) > 0 else None
    lat = safe_float(arr[1], None) if len(arr) > 1 else None
    dist = safe_float(arr[2], None) if len(arr) > 2 else None
    speed_lon = safe_float(arr[3], 0.0) if len(arr) > 3 else 0.0
    return lon, lat, dist, speed_lon

def _try_calc_ut(jd_ut, body, flag_extra=0):
    global USE_MOSEPH
    lon, lat, dist, speed = _calc_ut_standardized(jd_ut, body, flag_extra=flag_extra, use_moseph=USE_MOSEPH)
    if lon is None:
        USE_MOSEPH = True
        lon, lat, dist, speed = _calc_ut_standardized(jd_ut, body, flag_extra=flag_extra, use_moseph=True)
    return lon, lat, dist, speed

def sidereal_longitude(body, jd_ut, ayanamsa):
    if body == -1:
        ra = sidereal_longitude(swe.MEAN_NODE, jd_ut, ayanamsa)
        return normalize_angle(ra + 180.0)
    lon, lat, dist, speed = _try_calc_ut(jd_ut, body, flag_extra=0)
    if lon is None:
        return None
    ay = swe.get_ayanamsa_ut(jd_ut)
    return normalize_angle(lon - ay)

def declination(body, jd_ut):
    lon, lat, dist, speed = _try_calc_ut(jd_ut, body, flag_extra=swe.FLG_EQUATORIAL)
    return lat

# ---------- Core computations (aspect timeline & KP table) ----------
def refine_exact_time(body_a, body_b, target_angle, start_utc, tzname, ay_mode, tol_deg=1/60, max_iter=28):
    left = start_utc - timedelta(hours=6)
    right = start_utc + timedelta(hours=6)
    swe.set_sid_mode(ay_mode, 0, 0)
    def angle_at(t):
        jd = julday_from_dt(t)
        la = sidereal_longitude(body_a, jd, ay_mode)
        lb = sidereal_longitude(body_b, jd, ay_mode)
        if la is None or lb is None: return None
        d = normalize_angle(la - lb); d = d if d <= 180 else 360 - d
        return d
    a_left = angle_at(left); a_right = angle_at(right)
    if a_left is None or a_right is None: return to_local(start_utc, tzname)
    for _ in range(max_iter):
        mid = left + (right-left)/2
        a_mid = angle_at(mid)
        if a_mid is None:
            mid += timedelta(minutes=2); a_mid = angle_at(mid)
            if a_mid is None: break
        if abs(a_mid - target_angle) <= tol_deg: return to_local(mid, tzname)
        if abs(a_left - target_angle) < abs(a_right - target_angle): right = mid; a_right = a_mid
        else: left = mid; a_left = a_mid
    return to_local(left + (right-left)/2, tzname)

def planetary_aspect_timeline(date_local, tzname="Asia/Kolkata", ay_mode=swe.SIDM_LAHIRI,
                              orb_major=3.0, orb_moon=6.0, step_minutes=20):
    swe.set_sid_mode(ay_mode, 0, 0)
    try: swe.set_ephe_path("/usr/share/ephe")
    except Exception: pass
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname); end_utc = to_utc(end_local, tzname)
    planet_ids = {name: pid for name, pid in PLANETS}; names = list(planet_ids.keys())
    events = []; cur = t_utc; seen = set()
    while cur < end_utc:
        jd = julday_from_dt(cur)
        longs = {nm: sidereal_longitude(pid, jd, ay_mode) for nm, pid in planet_ids.items()}
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                A,B = names[i], names[j]
                la,lb = longs.get(A), longs.get(B)
                if la is None or lb is None: continue
                diff = min_angle_diff(la, lb)
                for exact,a_name in ASPECTS.items():
                    orb = orb_moon if ("Moon" in (A,B)) else orb_major
                    if diff is not None and abs(diff-exact) <= orb:
                        key = (a_name,A,B)
                        if key in seen: continue
                        exact_local = refine_exact_time(planet_ids[A], planet_ids[B], exact, cur, tzname, ay_mode)
                        kp_info = {"Nakshatra":"","Star Lord":"","Sub-Lord":""}
                        if A=="Moon" or B=="Moon":
                            jd_exact = julday_from_dt(exact_local.astimezone(pytz.utc))
                            lon_moon = sidereal_longitude(swe.MOON, jd_exact, ay_mode)
                            if lon_moon is not None:
                                nak,pada = nakshatra_for(lon_moon)
                                star, sub = NAK_LORD[nak], kp_sublord_of_longitude(lon_moon)[3]
                                kp_info = {"Nakshatra":f"{nak}-{pada}","Star Lord":star,"Sub-Lord":sub}
                        events.append({
                            "Time": exact_local.strftime("%Y-%m-%d %H:%M"),
                            "Aspect": a_name, "Exact°": exact,
                            "Planet A": A, "Planet B": B,
                            "Moon Nakshatra@Exact": kp_info["Nakshatra"],
                            "Moon Star Lord@Exact": kp_info["Star Lord"],
                            "Moon Sub-Lord@Exact": kp_info["Sub-Lord"],
                        })
                        seen.add(key)
        cur += timedelta(minutes=step_minutes)
    return pd.DataFrame(sorted(events, key=lambda x: x["Time"]))

def intraday_kp_table(date_local, tzname="Asia/Kolkata", ay_mode=swe.SIDM_KRISHNAMURTI,
                      planets=("Moon","Mercury","Venus","Sun","Mars","Jupiter","Saturn","Rahu","Ketu"),
                      step_minutes=1):
    """Strict KP by default: Krishnamurti ayanamsa + 1-min Moon-friendly scan."""
    swe.set_sid_mode(ay_mode, 0, 0)
    try: swe.set_ephe_path("/usr/share/ephe")
    except Exception: pass
    start_local = datetime(date_local.year, date_local.month, date_local.day, 0, 0, 0)
    end_local = start_local + timedelta(days=1)
    t_utc = to_utc(start_local, tzname); end_utc = to_utc(end_local, tzname)
    pid_map = dict(PLANETS)
    rows = []
    for pname in planets:
        body = pid_map[pname]
        cur = t_utc
        jd0 = julday_from_dt(cur)
        lon0 = sidereal_longitude(body, jd0, ay_mode)
        if lon0 is None: continue
        nak0, _, star0, sub0 = kp_sublord_of_longitude(lon0)
        while cur < end_utc:
            nxt = cur + timedelta(minutes=step_minutes)
            jd = julday_from_dt(nxt)
            lon = sidereal_longitude(body, jd, ay_mode)
            if lon is None: cur = nxt; continue
            nak, pada, star, sub = kp_sublord_of_longitude(lon)
            if (star != star0) or (sub != sub0):
                lo, hi = cur, nxt
                for _ in range(24):
                    mid = lo + (hi-lo)/2
                    jd_mid = julday_from_dt(mid)
                    lon_mid = sidereal_longitude(body, jd_mid, ay_mode)
                    if lon_mid is None: break
                    _, _, star_mid, sub_mid = kp_sublord_of_longitude(lon_mid)
                    if (star_mid == star0) and (sub_mid == sub0): lo = mid
                    else: hi = mid
                jd_exact = julday_from_dt(hi)
                lon_exact = sidereal_longitude(body, jd_exact, ay_mode)
                sign_ex, deg_ex = ecl_to_sign_deg(lon_exact)
                nak_ex, pada_ex, star_ex, sub_ex = kp_sublord_of_longitude(lon_exact)
                motion = "D"  # simplify
                rows.append({
                    "Planet": pname[:2],
                    "Date": to_local(hi, tzname).strftime("%Y-%m-%d"),
                    "Time": to_local(hi, tzname).strftime("%H:%M:%S"),
                    "Motion": motion,
                    "Sign Lord": SIGN_LORDS[sign_ex],
                    "Star Lord": NAK_LORD[nak_ex],
                    "Sub Lord": sub_ex,
                    "Zodiac": sign_ex,
                    "Nakshatra": nak_ex,
                    "Pada": pada_ex,
                    "Pos in Zodiac": deg_to_dms(deg_ex),
                    "Declination": ""
                })
                star0, sub0 = star_ex, sub_ex
            cur = nxt
    return pd.DataFrame(rows).sort_values(["Date","Time","Planet"])

# ----- Scoring rules for analysis -----
DEFAULT_RULES = {
    "weights": {
        "benefics": {"Jupiter": 2.0, "Venus": 1.5, "Moon": 1.0, "Mercury": 0.8},
        "malefics": {"Saturn": -2.0, "Mars": -1.5, "Rahu": -1.5, "Ketu": -1.2},
        "sun": 0.5
    },
    "aspect_multipliers": {"Trine": 1.0, "Sextile": 0.8, "Conjunction": 0.6, "Opposition": -0.9, "Square": -1.0},
    "asset_bias": {
        "NIFTY": {"Jupiter": +0.5, "Saturn": -0.3, "Mercury": +0.2},
        "BANKNIFTY": {"Jupiter": +0.6, "Saturn": -0.5, "Mercury": +0.3},
        "GOLD": {"Saturn": -0.6, "Jupiter": +0.4, "Venus": +0.2, "Rahu": +0.3},
        "CRUDE": {"Mars": +0.6, "Saturn": -0.2, "Jupiter": +0.2},
        "BTC": {"Rahu": +0.6, "Saturn": -0.4, "Jupiter": +0.2},
        "DOW": {"Jupiter": +0.4, "Saturn": -0.3, "Mercury": +0.2}
    },
    "kp_weights": {
        "star": {"Jupiter": +0.6, "Venus": +0.4, "Mercury": +0.2, "Moon": +0.2, "Sun": +0.1,
                 "Saturn": -0.6, "Mars": -0.4, "Rahu": -0.5, "Ketu": -0.4},
        "sub":  {"Jupiter": +0.8, "Venus": +0.5, "Mercury": +0.3, "Moon": +0.3, "Sun": +0.1,
                 "Saturn": -0.8, "Mars": -0.6, "Rahu": -0.6, "Ketu": -0.5}
    },
    "thresholds": {"bullish": 1.0, "bearish": -1.0}
}

def score_event(row, asset, rules=DEFAULT_RULES):
    A, B, aspect = row["Planet A"], row["Planet B"], row["Aspect"]
    w = rules["weights"]; mult = rules["aspect_multipliers"]; bias_map = rules["asset_bias"]
    def p_weight(p):
        if p in ("Sun",): return w["sun"]
        if p in w["benefics"]: return w["benefics"][p]
        if p in w["malefics"]: return w["malefics"][p]
        return 0.0
    s = (p_weight(A) + p_weight(B)) * mult.get(aspect, 0.0)
    asset = asset.upper()
    if asset in bias_map: s += bias_map[asset].get(A,0.0) + bias_map[asset].get(B,0.0)
    if "Moon Star Lord@Exact" in row and row["Moon Star Lord@Exact"]:
        s += rules["kp_weights"]["star"].get(row["Moon Star Lord@Exact"], 0.0)
    if "Moon Sub-Lord@Exact" in row and row["Moon Sub-Lord@Exact"]:
        s += rules["kp_weights"]["sub"].get(row["Moon Sub-Lord@Exact"], 0.0)
    return s

def score_kp_only(row, asset, rules=DEFAULT_RULES):
    s = 0.0
    s += rules["kp_weights"]["star"].get(row["Star Lord"], 0.0)
    s += rules["kp_weights"]["sub"].get(row["Sub Lord"], 0.0)
    sl = row.get("Sign Lord")
    if sl: s += 0.2 if sl in ("Jupiter","Venus","Mercury","Moon","Sun") else -0.2
    bias_map = rules["asset_bias"]; asset = asset.upper()
    if asset in bias_map:
        s += bias_map[asset].get("Jupiter",0)/10.0 - abs(bias_map[asset].get("Saturn",0))/10.0
    return s

def classify_score(score, rules=DEFAULT_RULES):
    if score >= rules["thresholds"]["bullish"]: return "Bullish"
    if score <= rules["thresholds"]["bearish"]: return "Bearish"
    return "Neutral"

# ---- Styling helpers ----
def style_signal_table(df):
    if df.empty: return df
    def color_row(row):
        sig = row.get("Signal","")
        if sig == "Bullish": return ['background-color: #cfe8ff' for _ in row]  # blue
        if sig == "Bearish": return ['background-color: #ffd6d6' for _ in row]  # red
        return ['']*len(row)
    return df.style.apply(color_row, axis=1)

# ---------- UI ----------
st.set_page_config(page_title="Vedic Sidereal — KP Strict + Sector Scanner", layout="wide")
st.title("🪐 Vedic Sidereal Transit Explorer — KP Strict Mode + Sector Scanner")

if not SWISSEPH_AVAILABLE:
    st.error("pyswisseph not installed here. Install locally: pip install pyswisseph streamlit pytz pandas")
    st.stop()

# Global controls
colA, colB, colC = st.columns(3)
with colA:
    date_in = st.date_input("Select Date", value=pd.Timestamp.today().date())
with colB:
    tz_in = st.text_input("Time Zone (IANA)", value="Asia/Kolkata")
with colC:
    strict_kp = st.checkbox("KP strict mode (Krishnamurti + 1-min Moon scan)", value=True)

ay_mode = swe.SIDM_KRISHNAMURTI if strict_kp else swe.SIDM_LAHIRI
swe.set_sid_mode(ay_mode, 0, 0)

# Build base timelines once
with st.spinner("Computing base timelines..."):
    asp_timeline_df = planetary_aspect_timeline(date_in, tzname=tz_in, ay_mode=ay_mode, orb_major=3.0, orb_moon=6.0, step_minutes=20)
    kp_moon_df = intraday_kp_table(date_in, tzname=tz_in, ay_mode=ay_mode, planets=("Moon",), step_minutes=1 if strict_kp else 5)

# Tabs
tabs = st.tabs(["Data Analysis", "Intraday KP Table", "Sector Scanner"])

with tabs[0]:
    st.subheader("📊 Data Analysis — Symbol-wise Bullish/Bearish Timeline")
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        symbol = st.text_input("Symbol", value="NIFTY")
    with c2:
        asset_class = st.selectbox("Asset Class", ["NIFTY","BANKNIFTY","GOLD","CRUDE","BTC","DOW","OTHER"], index=0)
    with c3:
        start_t = st.time_input("Start Time", value=dtime(9,15))
    with c4:
        end_t = st.time_input("End Time", value=dtime(15,30))

    tz = pytz.timezone(tz_in)
    dfA = asp_timeline_df.copy()
    dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
    start_local = tz.localize(datetime.combine(date_in, start_t))
    end_local = tz.localize(datetime.combine(date_in, end_t))
    if end_local <= start_local: end_local = end_local + timedelta(days=1)
    dfA = dfA[(dfA["TimeLocal"] >= start_local) & (dfA["TimeLocal"] < end_local)].copy()

    dfK = kp_moon_df.copy()
    dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
    dfK = dfK[(dfK["DT"] >= start_local) & (dfK["DT"] < end_local)].copy()

    scoresA = [score_event(r, asset_class, DEFAULT_RULES) for _, r in dfA.iterrows()]
    dfA["Score"] = scoresA
    dfA["Signal"] = [classify_score(s, DEFAULT_RULES) for s in scoresA]
    dfA["Symbol"] = symbol.upper()
    view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B",
                  "Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
    dfA_view = dfA[view_colsA]

    if not dfK.empty:
        dfK["Score"] = [score_kp_only(r, asset_class, DEFAULT_RULES) for _, r in dfK.iterrows()]
        dfK["Signal"] = [classify_score(s, DEFAULT_RULES) for s in dfK["Score"]]
        dfK["Time"] = dfK["DT"].dt.strftime("%Y-%m-%d %H:%M")
        dfK_view = dfK.rename(columns={
            "Star Lord":"Moon Star Lord@Exact",
            "Sub Lord":"Moon Sub-Lord@Exact",
            "Nakshatra":"Moon Nakshatra@Exact"
        })
        dfK_view["Aspect"] = "KP (Moon Star/Sub change)"
        dfK_view["Exact°"] = ""
        dfK_view["Planet A"] = "Moon"; dfK_view["Planet B"] = "-"
        dfK_view["Symbol"] = symbol.upper()
        dfK_view = dfK_view[view_colsA]
        combined = pd.concat([dfA_view, dfK_view], ignore_index=True)
    else:
        combined = dfA_view.copy()

    combined = combined.sort_values("Time")
    st.dataframe(style_signal_table(combined), use_container_width=True)

with tabs[1]:
    st.subheader("📄 Intraday KP Table (Moon, Strict Mode friendly)")
    st.dataframe(kp_moon_df, use_container_width=True)

with tabs[2]:
    st.subheader("🏭 Sector Scanner — choose sector & symbol")
    DEFAULT_SECTORS = {
        "NIFTY50": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","BHARTIARTL","ITC","HINDUNILVR","LT","SBIN"],
        "BANKNIFTY": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","PNB","BANDHANBNK","FEDERALBNK"],
        "PHARMA": ["SUNPHARMA","CIPLA","DRREDDY","DIVISLAB","AUROPHARMA"],
        "AUTO": ["TATAMOTORS","MARUTI","M&M","EICHERMOT","HEROMOTOCO"],
        "FMCG": ["ITC","HINDUNILVR","NESTLEIND","BRITANNIA","DABUR"],
        "METAL": ["TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA","SAIL"],
        "OIL & GAS": ["RELIANCE","ONGC","BPCL","IOC","GAIL"],
        "SUGAR": ["BALRAMCHIN","EIDPARRY","DHAMPURSUG","DWARKESH"],
        "TEA": ["TATACONSUM","MCLEODRUSS","GOODRICKE"],
        "TELECOM": ["BHARTIARTL","IDEA"]
    }

    left, right = st.columns([2,1])
    with left:
        sector = st.selectbox("Sector", list(DEFAULT_SECTORS.keys()), index=0)
    with right:
        edit = st.checkbox("Edit sector list", value=False)

    sectors_json = st.text_area("Edit sector mapping (Python dict)", value=str(DEFAULT_SECTORS), height=160, disabled=not edit)
    try:
        import ast
        sectors = ast.literal_eval(sectors_json) if edit else DEFAULT_SECTORS
    except Exception:
        sectors = DEFAULT_SECTORS
        st.warning("Sector JSON parse failed. Using defaults.")

    symbols = sectors.get(sector, [])
    symbol = st.selectbox("Symbol", symbols, index=0 if symbols else None, disabled=(len(symbols)==0))

    s1, s2 = st.columns(2)
    with s1:
        start_t2 = st.time_input("Start Time", value=dtime(9,15))
    with s2:
        end_t2 = st.time_input("End Time", value=dtime(15,30))

    # Overview for all symbols in sector
    tz = pytz.timezone(tz_in)
    start_local2 = tz.localize(datetime.combine(date_in, start_t2))
    end_local2 = tz.localize(datetime.combine(date_in, end_t2))
    if end_local2 <= start_local2: end_local2 = end_local2 + timedelta(days=1)

    def signals_for(sym):
        # Use NIFTY bias for most, BANKNIFTY for bank sector
        acl = "BANKNIFTY" if sector == "BANKNIFTY" else "NIFTY"
        dfA = asp_timeline_df.copy()
        dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
        dfA = dfA[(dfA["TimeLocal"] >= start_local2) & (dfA["TimeLocal"] < end_local2)].copy()
        dfK = kp_moon_df.copy()
        dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
        dfK = dfK[(dfK["DT"] >= start_local2) & (dfK["DT"] < end_local2)].copy()

        scoresA = [score_event(r, acl, DEFAULT_RULES) for _, r in dfA.iterrows()]
        dfA["Score"] = scoresA
        dfA["Signal"] = [classify_score(s, DEFAULT_RULES) for s in scoresA]
        dfA["Symbol"] = sym

        if not dfK.empty:
            dfK["Score"] = [score_kp_only(r, acl, DEFAULT_RULES) for _, r in dfK.iterrows()]
            dfK["Signal"] = [classify_score(s, DEFAULT_RULES) for s in dfK["Score"]]
            dfK["Time"] = dfK["DT"].dt.strftime("%Y-%m-%d %H:%M")
            dfK_view = dfK.rename(columns={
                "Star Lord":"Moon Star Lord@Exact",
                "Sub Lord":"Moon Sub-Lord@Exact",
                "Nakshatra":"Moon Nakshatra@Exact"
            })
            dfK_view["Aspect"] = "KP (Moon Star/Sub change)"
            dfK_view["Exact°"] = ""
            dfK_view["Planet A"] = "Moon"; dfK_view["Planet B"] = "-"
            dfK_view["Symbol"] = sym
            view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B",
                          "Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
            combined = pd.concat([dfA[view_colsA], dfK_view[view_colsA]], ignore_index=True)
        else:
            view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B",
                          "Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
            combined = dfA[view_colsA]

        combined = combined.sort_values("Time")
        bull = int((combined["Signal"]=="Bullish").sum())
        bear = int((combined["Signal"]=="Bearish").sum())
        neu  = int((combined["Signal"]=="Neutral").sum())
        return combined, bull, bear, neu

    if symbols:
        # Overview table counts
        rows = []
        for sym in symbols:
            _, b, s, n = signals_for(sym)
            rows.append({"Symbol": sym, "Bullish": b, "Bearish": s, "Neutral": n})
        overview = pd.DataFrame(rows).sort_values(["Bullish","Bearish"], ascending=[False,True])
        st.markdown("**Sector overview (counts in selected time window):**")
        st.dataframe(overview, use_container_width=True)

        st.markdown("**Detailed signals for selected symbol:**")
        detail, bcnt, scnt, ncnt = signals_for(symbol)
        st.dataframe(style_signal_table(detail), use_container_width=True)
    else:
        st.info("No symbols configured for this sector.")
