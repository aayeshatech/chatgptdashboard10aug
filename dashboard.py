
import math
from datetime import datetime, timedelta, time as dtime
import pytz
import pandas as pd
import streamlit as st
import calendar as _cal

# ---------------- Setup ----------------
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

SIGN_LORDS = {"Aries":"Mars","Taurus":"Venus","Gemini":"Mercury","Cancer":"Moon","Leo":"Sun","Virgo":"Mercury",
    "Libra":"Venus","Scorpio":"Mars","Sagittarius":"Jupiter","Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter"}
ZODIAC_SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
NAKSHATRAS = ["Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu","Pushya","Ashlesha",
    "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha",
    "Jyeshtha","Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
    "Purva Bhadrapada","Uttara Bhadrapada","Revati"]
NAK_DEG = 360.0 / 27.0
ASPECTS = {0:"Conjunction",60:"Sextile",90:"Square",120:"Trine",180:"Opposition"}
USE_MOSEPH = False

DASHA_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
TOTAL_YEARS = 120.0
NAK_LORD = {"Ashwini":"Ketu","Bharani":"Venus","Krittika":"Sun","Rohini":"Moon","Mrigashira":"Mars","Ardra":"Rahu",
    "Punarvasu":"Jupiter","Pushya":"Saturn","Ashlesha":"Mercury","Magha":"Ketu","Purva Phalguni":"Venus","Uttara Phalguni":"Sun",
    "Hasta":"Moon","Chitra":"Mars","Swati":"Rahu","Vishakha":"Jupiter","Anuradha":"Saturn","Jyeshtha":"Mercury",
    "Mula":"Ketu","Purva Ashadha":"Venus","Uttara Ashadha":"Sun","Shravana":"Moon","Dhanishta":"Mars","Shatabhisha":"Rahu",
    "Purva Bhadrapada":"Jupiter","Uttara Bhadrapada":"Saturn","Revati":"Mercury"}

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

# Safe default rules reference used throughout (avoids NameError on first render)
RULES_CURRENT = DEFAULT_RULES
def get_rules():
    global RULES_CURRENT
    try:
        return RULES_CURRENT
    except Exception:
        return DEFAULT_RULES


# -------------- Helpers --------------
def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

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
    d = int(deg); m_float = abs(deg - d) * 60; m = int(m_float); s = int(round((m_float - m) * 60))
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
        if hasattr(out, k): possible.append(getattr(out, k))
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
    if lon is None: return None
    ay = swe.get_ayanamsa_ut(jd_ut)
    return normalize_angle(lon - ay)

def refine_exact_time(body_a, body_b, target_angle, start_utc, tzname, ay_mode, tol_deg=1/60, max_iter=28):
    left = start_utc - timedelta(hours=6); right = start_utc + timedelta(hours=6)
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
                      planets=("Moon",), step_minutes=1):
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
                rows.append({
                    "Planet": pname[:2],
                    "Date": to_local(hi, tzname).strftime("%Y-%m-%d"),
                    "Time": to_local(hi, tzname).strftime("%H:%M:%S"),
                    "Motion": "D",
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

# -------- Scoring & styling --------
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

def style_signal_table(df):
    if df.empty: return df
    def color_row(row):
        sig = row.get("Signal","")
        if sig == "Bullish": return ['background-color: #cfe8ff' for _ in row]  # blue
        if sig == "Bearish": return ['background-color: #ffd6d6' for _ in row]  # red
        return ['']*len(row)
    return df.style.apply(color_row, axis=1)

def classify_net(bull, bear):
    net = bull - bear
    if net > 0: return "Bullish"
    if net < 0: return "Bearish"
    return "Neutral"

def style_sector_table(df, current_sector=None):
    if df.empty: return df
    def color_row(row):
        sig = row.get("Trend","")
        base = ''
        if sig == "Bullish": base = '#cfe8ff'
        elif sig == "Bearish": base = '#ffd6d6'
        if current_sector and row.get("Sector","") == current_sector:
            base = '#b3e6cc'
        return [f'background-color: {base}' if base else '' for _ in row]
    return df.style.apply(color_row, axis=1)

def make_rules_with_aspects(default_rules, trine, sextile, conj, opp, square):
    r = {k: (v.copy() if isinstance(v, dict) else v) for k, v in default_rules.items()}
    r['weights'] = default_rules['weights'].copy()
    r['aspect_multipliers'] = default_rules['aspect_multipliers'].copy()
    r['asset_bias'] = {k:v.copy() for k,v in default_rules['asset_bias'].items()}
    r['kp_weights'] = {k:v.copy() for k,v in default_rules['kp_weights'].items()}
    r['thresholds'] = default_rules['thresholds'].copy()
    r['aspect_multipliers'].update({
        'Trine': trine, 'Sextile': sextile, 'Conjunction': conj, 'Opposition': opp, 'Square': square
    })
    return r

# --- Score-based sector engine ---
def sector_scores_for_window(sector_syms, asp_df, kp_df, tz_in, date_in, start_t, end_t, asset_class, kp_premium=1.2, rules=None):
    rules = rules or DEFAULT_RULES
    tz = pytz.timezone(tz_in)
    start_local = tz.localize(datetime.combine(date_in, start_t))
    end_local = tz.localize(datetime.combine(date_in, end_t))
    if end_local <= start_local: end_local = end_local + timedelta(days=1)

    dfA = asp_df.copy()
    dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
    dfA = dfA[(dfA["TimeLocal"] >= start_local) & (dfA["TimeLocal"] < end_local)].copy()
    dfK = kp_df.copy()
    dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
    dfK = dfK[(dfK["DT"] >= start_local) & (dfK["DT"] < end_local)].copy()

    if not dfA.empty:
        dfA["Score"] = [score_event(r, asset_class, rules) for _, r in dfA.iterrows()]
        dfA["w"] = 1.0
    else:
        dfA["Score"] = []; dfA["w"] = []
    if not dfK.empty:
        dfK["Score"] = [score_kp_only(r, asset_class, rules) for _, r in dfK.iterrows()]
        dfK["w"] = kp_premium
        dfK["Time"] = dfK["DT"].dt.strftime("%Y-%m-%d %H:%M")
    else:
        dfK["Score"] = []; dfK["w"] = []

    combined_scores = list(dfA["Score"]*dfA["w"]) + list(dfK["Score"]*dfK["w"])
    if len(sector_syms) == 0:
        return 0.0, 0.0, 0.0
    total_score = float(sum(combined_scores))
    avg_per_symbol = total_score / float(len(sector_syms))
    abs_total = float(sum(abs(s) for s in combined_scores)) or 1.0
    confidence = min(1.0, abs(total_score) / abs_total)
    return total_score, avg_per_symbol, confidence

def build_sector_overview(sectors, asp_timeline_df, kp_moon_df, tz_in, date_in, start_t, end_t, kp_premium=1.2, net_threshold=0.25, rules=None):
    rows = []
    for sec, syms in sectors.items():
        acl = "BANKNIFTY" if sec == "BANKNIFTY" else "NIFTY"
        total, avg_ps, conf = sector_scores_for_window(syms, asp_timeline_df, kp_moon_df, tz_in, date_in, start_t, end_t, acl, kp_premium=kp_premium, rules=rules)
        trend = "Bullish" if total > net_threshold else ("Bearish" if total < -net_threshold else "Neutral")
        rows.append({"Sector": sec, "NetScore": round(total,2), "Avg/Stock": round(avg_ps,2), "Confidence": round(conf,2), "Trend": trend})
    df = pd.DataFrame(rows).sort_values(["NetScore","Avg/Stock","Confidence"], ascending=[False,False,False]).reset_index(drop=True)
    return df



def build_calendar_table(month_days, month_df):
    """Return (display_df, styled_df) for a month calendar using pandas Styler.
    Colors: blue for positive NetScore, red for negative, white/grey for ~0.
    """
    import pandas as _pd
    import calendar as _cal
    # Map date->score
    score_map = {row['Date']: float(row['NetScore']) for _, row in month_df.iterrows()}
    first = month_days[0]
    first_weekday = _cal.monthrange(first.year, first.month)[0]  # 0=Mon
    # Build 6x7 grids
    labels = [["" for _ in range(7)] for __ in range(6)]
    values = [[None for _ in range(7)] for __ in range(6)]
    r = 0; c = first_weekday
    for d in month_days:
        key = str(d)
        labels[r][c] = str(d.day)
        values[r][c] = score_map.get(key, 0.0)
        c += 1
        if c == 7:
            c = 0; r += 1
    cols = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    df_disp = _pd.DataFrame(labels, index=[f'W{i+1}' for i in range(6)], columns=cols)
    df_vals = _pd.DataFrame(values, index=df_disp.index, columns=df_disp.columns)

    def color_fn(row):
        out = []
        for j, cell in enumerate(row.index):
            v = row[cell]
            # If NaN/None display stays as is
            try:
                x = float(df_vals.loc[row.name, cell]) if df_vals.loc[row.name, cell] is not None else None
            except Exception:
                x = None
            if x is None:
                out.append('background-color: #f3f3f3')
            elif x > 0.2:
                out.append('background-color: #cfe8ff')  # light blue
            elif x < -0.2:
                out.append('background-color: #ffd6d6')  # light red
            else:
                out.append('background-color: #f9f9f9')  # near zero
        return out

    styled = df_disp.style.apply(color_fn, axis=1)
    return df_disp, styled


def _aspect_strength_for_sort(aspect_name):
    # heuristic strength for ranking daily "major" events
    base = {"Conjunction": 1.0, "Opposition": 0.9, "Square": 0.8, "Trine": 0.7, "Sextile": 0.6}
    return base.get(aspect_name, 0.5)

def _duration_hint(aspect_name):
    # rough duration text for swing persistence
    if aspect_name in ("Conjunction","Opposition"):
        return "FOR 5–7 DAYS"
    if aspect_name in ("Square",):
        return "FOR 4–6 DAYS"
    if aspect_name in ("Trine","Sextile"):
        return "FOR 3–5 DAYS"
    return "FOR 2–4 DAYS"

def build_transit_cards_for_range(start_date, days, tz_in, ay_mode, strict_kp, sectors, start_t, end_t, rules, kp_premium, net_threshold, planets_filter=None, aspects_filter=None, per_day_limit=3):
    cards = []
    # Build per-day rank + extract strongest 1–2 non-Moon aspects
    for i in range(days):
        d = (pd.Timestamp(start_date) + pd.Timedelta(days=i)).date()
        asp, kp = cached_streams_for_date(d, tz_in, ay_mode, strict_kp)
        # strongest aspects (exclude Moon to avoid clutter)
        if asp.empty:
            daily_events = []
        else:
            df = asp.copy()
            df = df[(df["Planet A"]!="Moon") & (df["Planet B"]!="Moon")]
            if df.empty:
                daily_events = []
            else:
                # crude strength: aspect weight + benefic/malefic sum
                df["rough"] = df.apply(lambda r: _aspect_strength_for_sort(r["Aspect"]) +                     (1 if r["Planet A"] in ("Jupiter","Venus") else 0) +                     (1 if r["Planet B"] in ("Jupiter","Venus") else 0) +                     (-0.6 if r["Planet A"] in ("Saturn","Mars") else 0) +                     (-0.6 if r["Planet B"] in ("Saturn","Mars") else 0), axis=1)
                df = df.sort_values(["rough","Time"], ascending=[False, True]).head(2)
                daily_events = df.to_dict("records")
        # sector rank for that day
        rdf = build_sector_overview(sectors, asp, kp, tz_in, d, start_t, end_t, kp_premium=float(kp_premium), net_threshold=float(net_threshold), rules=rules)
        if rdf.empty:
            top_sector = "-"; trend = "NEUTRAL"; netscore = 0
        else:
            top = rdf.iloc[0]
            top_sector, netscore = top["Sector"], float(top["NetScore"])
            trend = "BULLISH" if netscore > net_threshold else ("BEARISH" if netscore < -net_threshold else "NEUTRAL")
        # build cards
        for ev in (daily_events if daily_events else [{}]):
            if daily_events:
                title = f"{pd.to_datetime(ev['Time']).strftime('%a, %b %d')} — {ev['Planet A']} {ev['Aspect'].lower()} {ev['Planet B']}"
                impact = f"{trend} {_duration_hint(ev['Aspect'])}"
                event_text = f"{ev['Planet A']} {ev['Aspect'].lower()} {ev['Planet B']}"
            else:
                title = f"{pd.Timestamp(d).strftime('%a, %b %d')} — Sector Bias"
                impact = f"{trend} FOR 1 DAY"
                event_text = "Sector bias from combined astro factors"
            cards.append({
                "date": str(d),
                "title": title,
                "event": event_text,
                "impact": impact,
                "sector": top_sector,
                "netscore": round(netscore,2)
            })
    return cards

def render_cards(cards, header):
    st.markdown(f"### {header}")
    if not cards:
        st.info("No transits found.")
        return
    for c in cards:
        color = "#19c37d" if "BULLISH" in c["impact"] else ("#f7766d" if "BEARISH" in c["impact"] else "#f5a623")
        st.markdown(f"""
<div style='border:1px solid #e5e7eb;padding:12px;border-radius:8px;background:#f7faff;margin-bottom:8px;'>
  <div style='font-weight:600;color:#1d4ed8'>{c['title']}</div>
  <div><strong>Event:</strong> {c['event']}</div>
  <div><strong>Impact:</strong> <span style='color:{color};font-weight:700'>{c['impact']}</span></div>
  <div><strong>Affected Sector:</strong> {c['sector']} &nbsp; <span style='opacity:.7'>(NetScore {c['netscore']})</span></div>
</div>
""", unsafe_allow_html=True)


def _duration_days(aspect_name):
    # mid-point days for analysis window
    if aspect_name in ("Conjunction","Opposition"): return 6
    if aspect_name in ("Square",): return 5
    if aspect_name in ("Trine","Sextile"): return 4
    return 3

def analyze_transit_window(start_date, days, sectors, tz_in, start_t, end_t, rules, kp_premium, net_threshold, ay_mode, strict_kp):
    # Aggregate sector NetScore over [start_date, start_date+days-1]
    dates = [(pd.Timestamp(start_date) + pd.Timedelta(days=i)).date() for i in range(days)]
    agg = None
    for d in dates:
        asp, kp = cached_streams_for_date(d, tz_in, ay_mode, strict_kp)
        rdf = build_sector_overview(sectors, asp, kp, tz_in, d, start_t, end_t, kp_premium=float(kp_premium), net_threshold=float(net_threshold), rules=rules)
        rdf = rdf[["Sector","NetScore","Avg/Stock","Confidence"]]
        rdf = rdf.rename(columns={"NetScore": f"NetScore_{d}"})
        agg = rdf if agg is None else agg.merge(rdf, on=["Sector","Avg/Stock","Confidence"], how="outer")
    # Sum netscore across days
    if agg is None or agg.empty:
        return pd.DataFrame(), {}
    score_cols = [c for c in agg.columns if c.startswith("NetScore_")]
    agg["TotalNet"] = agg[score_cols].sum(axis=1, skipna=True)
    agg = agg.sort_values("TotalNet", ascending=False).reset_index(drop=True)
    meta = {"window": f"{start_date} → {(pd.Timestamp(start_date)+pd.Timedelta(days=days-1)).date()}", "days": days}
    return agg, meta

def stock_breakdown_for_sector_over_window(sector, sectors_map, start_date, days, tz_in, start_t, end_t, rules, kp_premium, ay_mode, strict_kp):
    dates = [(pd.Timestamp(start_date) + pd.Timedelta(days=i)).date() for i in range(days)]
    syms = sectors_map.get(sector, [])
    if not syms: return pd.DataFrame()
    acl = "BANKNIFTY" if sector == "BANKNIFTY" else "NIFTY"
    rows_accum = {}
    tz = pytz.timezone(tz_in)
    for d in dates:
        asp, kp = cached_streams_for_date(d, tz_in, ay_mode, strict_kp)
        # time filter
        start_local = tz.localize(datetime.combine(d, start_t))
        end_local = tz.localize(datetime.combine(d, end_t))
        if end_local <= start_local: end_local = end_local + timedelta(days=1)
        dfA = asp.copy()
        dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
        dfA = dfA[(dfA["TimeLocal"] >= start_local) & (dfA["TimeLocal"] < end_local)].copy()
        dfK = kp.copy()
        dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
        dfK = dfK[(dfK["DT"] >= start_local) & (dfK["DT"] < end_local)].copy()
        for sym in syms:
            sA = [score_event(r, acl, rules) for _, r in dfA.iterrows()]
            sK = [score_kp_only(r, acl, rules) for _, r in dfK.iterrows()] if not dfK.empty else []
            sigA = [classify_score(s, rules) for s in sA]
            sigK = [classify_score(s, rules) for s in sK]
            signals = sigA + sigK
            b,bear,n = signals.count("Bullish"), signals.count("Bearish"), signals.count("Neutral")
            if sym not in rows_accum:
                rows_accum[sym] = {"Symbol": sym, "Bullish":0, "Bearish":0, "Neutral":0}
            rows_accum[sym]["Bullish"] += b; rows_accum[sym]["Bearish"] += bear; rows_accum[sym]["Neutral"] += n
    out = pd.DataFrame(list(rows_accum.values())).sort_values(["Bullish","Bearish"], ascending=[False,True])
    return out


def _duration_days_for_bodies(aspect_name, A, B):
    # Heuristic durations by fastest involved planet
    fast = set([A, B])
    if "Moon" in fast: return 2  # 1–2 days
    if "Mercury" in fast: return 3
    if "Sun" in fast: return 4
    if "Venus" in fast or "Mars" in fast: return 5
    if "Jupiter" in fast or "Saturn" in fast or "Rahu" in fast or "Ketu" in fast: return 6
    return _duration_days(aspect_name)

def _duration_hint_by_bodies(aspect_name, A, B):
    d = _duration_days_for_bodies(aspect_name, A, B)
    lo = max(1, d-1); hi = d+1
    return f"FOR {lo}\u2013{hi} DAYS"

# ---------------- UI ----------------
st.set_page_config(page_title="Vedic Sidereal — KP Strict + Sector Ranking", layout="wide")
st.title("🪐 Vedic Sidereal — Sector Ranking + KP Strict")

if not SWISSEPH_AVAILABLE:
    st.error("pyswisseph not installed here. Install locally: pip install pyswisseph streamlit pytz pandas")
    st.stop()

# Global controls
colA, colB, colC = st.columns(3)
with colA:
    date_in = st.date_input("Select Date", value=pd.Timestamp.today().date(), key="date_global")
with colB:
    tz_in = st.text_input("Time Zone (IANA)", value="Asia/Kolkata", key="tz_global")
with colC:
    strict_kp = st.checkbox("KP strict mode (Krishnamurti + 1-min Moon scan)", value=True, key="strict_global")
ay_mode = swe.SIDM_KRISHNAMURTI if strict_kp else swe.SIDM_LAHIRI
swe.set_sid_mode(ay_mode, 0, 0)

# Defaults in session_state for settings
if "kp_premium" not in st.session_state: st.session_state.kp_premium = 1.2
if "net_threshold" not in st.session_state: st.session_state.net_threshold = 0.25
if "aspect_weights" not in st.session_state:
    st.session_state.aspect_weights = {
        "Trine": DEFAULT_RULES["aspect_multipliers"]["Trine"],
        "Sextile": DEFAULT_RULES["aspect_multipliers"]["Sextile"],
        "Conjunction": DEFAULT_RULES["aspect_multipliers"]["Conjunction"],
        "Opposition": DEFAULT_RULES["aspect_multipliers"]["Opposition"],
        "Square": DEFAULT_RULES["aspect_multipliers"]["Square"],
    }

with st.spinner("Computing base timelines..."):
    asp_timeline_df = planetary_aspect_timeline(date_in, tzname=tz_in, ay_mode=ay_mode, step_minutes=20)
    kp_moon_df = intraday_kp_table(date_in, tzname=tz_in, ay_mode=ay_mode, planets=("Moon",), step_minutes=1 if strict_kp else 5)

tabs = st.tabs(["Settings", "Sector Scanner", "Data Analysis", "Intraday KP Table", "Weekly Outlook", "Monthly Outlook"])

# -------- Settings Tab --------
with tabs[0]:
    st.subheader("⚙️ Settings")
    s1, s2 = st.columns(2)
    with s1:
        st.session_state.kp_premium = st.slider("KP weight", min_value=0.5, max_value=2.0, value=float(st.session_state.kp_premium), step=0.1)
    with s2:
        st.session_state.net_threshold = st.slider("NetScore threshold (bull/bear)", min_value=0.0, max_value=2.0, value=float(st.session_state.net_threshold), step=0.05)

    with st.expander("⚖️ Weights — Aspect multipliers", expanded=False):
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.session_state.aspect_weights["Trine"] = st.slider("Trine", -2.0, 2.0, float(st.session_state.aspect_weights["Trine"]), 0.1)
            st.session_state.aspect_weights["Sextile"] = st.slider("Sextile", -2.0, 2.0, float(st.session_state.aspect_weights["Sextile"]), 0.1)
        with col_w2:
            st.session_state.aspect_weights["Conjunction"] = st.slider("Conjunction", -2.0, 2.0, float(st.session_state.aspect_weights["Conjunction"]), 0.1)
            st.session_state.aspect_weights["Opposition"] = st.slider("Opposition", -2.0, 2.0, float(st.session_state.aspect_weights["Opposition"]), 0.1)
        with col_w3:
            st.session_state.aspect_weights["Square"] = st.slider("Square", -2.0, 2.0, float(st.session_state.aspect_weights["Square"]), 0.1)

    st.success("Settings updated. Switch to Sector Scanner / Analysis to see impact.")

# Build rules_current from session state
rules_current = make_rules_with_aspects(
    DEFAULT_RULES,
    st.session_state.aspect_weights["Trine"],
    st.session_state.aspect_weights["Sextile"],
    st.session_state.aspect_weights["Conjunction"],
    st.session_state.aspect_weights["Opposition"],
    st.session_state.aspect_weights["Square"],
)

# Safe alias
try:
    RULES_CURRENT = rules_current
except Exception:
    RULES_CURRENT = DEFAULT_RULES

# -------- Sector Scanner --------
with tabs[1]:
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
        sector = st.selectbox("Sector", list(DEFAULT_SECTORS.keys()), index=0, key="sector_select")
    with right:
        edit = st.checkbox("Edit sector list", value=False, key="sector_edit_toggle")
    sectors_json = st.text_area("Edit sector mapping (Python dict)", value=str(DEFAULT_SECTORS), height=160, disabled=not edit, key="sector_json")
    try:
        import ast
        sectors = ast.literal_eval(sectors_json) if edit else DEFAULT_SECTORS
    except Exception:
        sectors = DEFAULT_SECTORS
        st.warning("Sector mapping parse failed; using defaults.")

    s1, s2 = st.columns(2)
    with s1:
        start_t2 = st.time_input("Start Time", value=dtime(9,15), key="start_time_sector")
    with s2:
        end_t2 = st.time_input("End Time", value=dtime(15,30), key="end_time_sector")

    rank_df = build_sector_overview(
        sectors, asp_timeline_df, kp_moon_df, tz_in, date_in, start_t2, end_t2,
        kp_premium=float(st.session_state.kp_premium),
        net_threshold=float(st.session_state.net_threshold),
        rules=rules_current
    )
    st.markdown("**All sectors ranking (by NetScore)**")
    st.dataframe(style_sector_table(rank_df, current_sector=sector), use_container_width=True)
    if not rank_df.empty:
        c_bull, c_bear = st.columns(2)
        with c_bull:
            top_bullish = rank_df.iloc[0]
            st.metric("Top Bullish Sector", f"{top_bullish['Sector']} (NetScore {int(top_bullish['NetScore'])})")
        with c_bear:
            worst = rank_df.sort_values("NetScore", ascending=True).iloc[0]
            st.metric("Top Bearish Sector", f"{worst['Sector']} (NetScore {int(worst['NetScore'])})")
    else:
        st.metric("Top Bullish Sector", "No data")
        st.metric("Top Bearish Sector", "No data")

    symbols = sectors.get(sector, [])
    symbol_sec = st.selectbox("Symbol", symbols, index=0 if symbols else None, disabled=(len(symbols)==0), key="sector_symbol")

    tz = pytz.timezone(tz_in)
    start_local2 = tz.localize(datetime.combine(date_in, start_t2))
    end_local2 = tz.localize(datetime.combine(date_in, end_t2))
    if end_local2 <= start_local2: end_local2 = end_local2 + timedelta(days=1)

    def score_event_local(dfA, dfK, asset_class, sym, rules):
        scoresA = [score_event(r, asset_class, rules) for _, r in dfA.iterrows()]
        dfA["Score"] = scoresA
        dfA["Signal"] = [classify_score(s, rules) for s in scoresA]
        dfA["Symbol"] = sym
        if not dfK.empty:
            dfK["Score"] = [score_kp_only(r, asset_class, rules) for _, r in dfK.iterrows()]
            dfK["Signal"] = [classify_score(s, rules) for s in dfK["Score"]]
            dfK["Time"] = dfK["DT"].dt.strftime("%Y-%m-%d %H:%M")
            dfK_view = dfK.rename(columns={"Star Lord":"Moon Star Lord@Exact","Sub Lord":"Moon Sub-Lord@Exact","Nakshatra":"Moon Nakshatra@Exact"})
            dfK_view["Aspect"] = "KP (Moon Star/Sub change)"
            dfK_view["Exact°"] = ""
            dfK_view["Planet A"] = "Moon"; dfK_view["Planet B"] = "-"
            dfK_view["Symbol"] = sym
            view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B","Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
            return pd.concat([dfA[view_colsA], dfK_view[view_colsA]], ignore_index=True)
        else:
            view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B","Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
            return dfA[view_colsA]

    dfA = asp_timeline_df.copy()
    dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
    dfA = dfA[(dfA["TimeLocal"] >= start_local2) & (dfA["TimeLocal"] < end_local2)].copy()
    dfK = kp_moon_df.copy()
    dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
    dfK = dfK[(dfK["DT"] >= start_local2) & (dfK["DT"] < end_local2)].copy()

    if symbols:
        rows = []
        acl = "BANKNIFTY" if sector == "BANKNIFTY" else "NIFTY"
        for sym in symbols:
            combined = score_event_local(dfA.copy(), dfK.copy(), acl, sym, get_rules())
            rows.append({"Symbol": sym, "Bullish": int((combined['Signal']=='Bullish').sum()),
                         "Bearish": int((combined['Signal']=='Bearish').sum()),
                         "Neutral": int((combined['Signal']=='Neutral').sum())})
        overview = pd.DataFrame(rows).sort_values(["Bullish","Bearish"], ascending=[False,True])
        st.markdown("**Sector overview (counts in selected time window):**")
        st.dataframe(overview, use_container_width=True)
        st.markdown("**Detailed signals for selected symbol:**")
        detail = score_event_local(dfA.copy(), dfK.copy(), acl, symbol_sec, get_rules()).sort_values("Time")
        st.dataframe(style_signal_table(detail), use_container_width=True)
    else:
        st.info("No symbols configured for this sector.")

# -------- Data Analysis --------
with tabs[2]:
    st.subheader("📊 Data Analysis — Symbol-wise Bullish/Bearish Timeline")
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        symbol = st.text_input("Symbol", value="NIFTY", key="symbol_analysis")
    with c2:
        asset_class = st.selectbox("Asset Class", ["NIFTY","BANKNIFTY","GOLD","CRUDE","BTC","DOW","OTHER"], index=0, key="asset_class_analysis")
    with c3:
        start_t = st.time_input("Start Time", value=dtime(9,15), key="start_time_analysis")
    with c4:
        end_t = st.time_input("End Time", value=dtime(15,30), key="end_time_analysis")

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

    scoresA = [score_event(r, asset_class, get_rules()) for _, r in dfA.iterrows()]
    dfA["Score"] = scoresA
    dfA["Signal"] = [classify_score(s, get_rules()) for s in scoresA]
    dfA["Symbol"] = symbol.upper()
    view_colsA = ["Time","Symbol","Signal","Score","Aspect","Exact°","Planet A","Planet B","Moon Nakshatra@Exact","Moon Star Lord@Exact","Moon Sub-Lord@Exact"]
    dfA_view = dfA[view_colsA]

    if not dfK.empty:
        dfK["Score"] = [score_kp_only(r, asset_class, get_rules()) for _, r in dfK.iterrows()]
        dfK["Signal"] = [classify_score(s, get_rules()) for s in dfK["Score"]]
        dfK["Time"] = dfK["DT"].dt.strftime("%Y-%m-%d %H:%M")
        dfK_view = dfK.rename(columns={"Star Lord":"Moon Star Lord@Exact","Sub Lord":"Moon Sub-Lord@Exact","Nakshatra":"Moon Nakshatra@Exact"})
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

# -------- Intraday KP --------
with tabs[3]:
    st.subheader("📄 Intraday KP Table (Moon)")
    st.dataframe(kp_moon_df, use_container_width=True)

# -------- Weekly Outlook --------
@st.cache_data(show_spinner=False)
def cached_streams_for_date(date_local, tzname, ay_mode, strict_kp):
    asp = planetary_aspect_timeline(date_local, tzname=tzname, ay_mode=ay_mode, step_minutes=20)
    kp  = intraday_kp_table(date_local, tzname=tzname, ay_mode=ay_mode, planets=("Moon",), step_minutes=1 if strict_kp else 5)
    return asp, kp

def rank_for_single_date(date_local, sectors, tz_in, start_t, end_t, kp_premium, net_threshold, rules, ay_mode, strict_kp):
    asp, kp = cached_streams_for_date(date_local, tz_in, ay_mode, strict_kp)
    rank_df = build_sector_overview(
        sectors, asp, kp, tz_in, date_local, start_t, end_t,
        kp_premium=float(kp_premium), net_threshold=float(net_threshold), rules=rules
    )
    return rank_df

with tabs[4]:
    st.subheader("📅 Weekly Outlook — sector bias by day")
    s1, s2, s3 = st.columns([1,1,1])
    with s1:
        week_start = st.selectbox("Week starts on", ["Monday","Sunday"], index=0, key="week_start")
    with s2:
        w_start_t = st.time_input("Start Time", value=dtime(9,15), key="week_start_time")
    with s3:
        w_end_t = st.time_input("End Time", value=dtime(15,30), key="week_end_time")

    anchor = pd.Timestamp(date_in)
    if week_start == "Monday":
        start_day = anchor - pd.Timedelta(days=anchor.weekday())
    else:
        start_day = anchor - pd.Timedelta(days=(anchor.weekday()+1) % 7)
    days = [start_day + pd.Timedelta(days=i) for i in range(7)]
    days_py = [d.date() for d in days]

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
    with st.expander("Edit sectors (optional)"):
        sectors_json_w = st.text_area("Sectors dict", value=str(DEFAULT_SECTORS), height=140, key="sectors_weekly_json")
        try:
            import ast
            sectors_w = ast.literal_eval(sectors_json_w)
        except Exception:
            sectors_w = DEFAULT_SECTORS
            st.warning("Sector mapping parse failed; using defaults.")
    rows = []
    with st.spinner("Computing weekly sector ranks..."):
        for d in days_py:
            rdf = rank_for_single_date(d, sectors_w, tz_in, w_start_t, w_end_t,
                                       st.session_state.kp_premium, st.session_state.net_threshold,
                                       RULES_CURRENT, ay_mode, strict_kp)
            if rdf.empty:
                rows.append({"Date": str(d), "Top Bullish": "-", "NetScore": 0, "Top Bearish": "-", "BearScore": 0})
            else:
                top_bull = rdf.iloc[0]; bot_bear = rdf.sort_values("NetScore", ascending=True).iloc[0]
                rows.append({"Date": str(d), "Top Bullish": top_bull["Sector"], "NetScore": top_bull["NetScore"],
                             "Top Bearish": bot_bear["Sector"], "BearScore": bot_bear["NetScore"]})
    week_df = pd.DataFrame(rows)
    st.dataframe(week_df, use_container_width=True)

    with st.expander("🔭 Next 7 Days — Major Transits", expanded=False):
        cards = build_transit_cards_for_range(days_py[0], 7, tz_in, ay_mode, strict_kp, sectors_w, w_start_t, w_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold)
        
st.markdown("**Upcoming planetary movements affecting sectors:**")
filt1, filt2, filt3 = st.columns([2,2,1])
with filt1:
    planets_pick = st.multiselect("Planets", ['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], default=['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], key="weekly_planets_filter")
with filt2:
    aspects_pick = st.multiselect("Aspects", ['Conjunction','Opposition','Square','Trine','Sextile'], default=['Conjunction','Opposition','Square','Trine','Sextile'], key="weekly_aspects_filter")
with filt3:
    per_day = st.slider("Max/day", 1, 5, 3, key="weekly_cards_per_day")
cards = build_transit_cards_for_range(days_py[0], 7, tz_in, ay_mode, strict_kp, sectors_w, w_start_t, w_end_t, get_rules(), st.session_state.kp_premium, st.session_state.net_threshold, planets_filter=planets_pick, aspects_filter=aspects_pick, per_day_limit=per_day)

st.markdown("**Upcoming planetary movements affecting sectors:**")
filt1m, filt2m, filt3m = st.columns([2,2,1])
with filt1m:
    planets_pick_m = st.multiselect("Planets", ['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], default=['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], key="monthly_planets_filter")
with filt2m:
    aspects_pick_m = st.multiselect("Aspects", ['Conjunction','Opposition','Square','Trine','Sextile'], default=['Conjunction','Opposition','Square','Trine','Sextile'], key="monthly_aspects_filter")
with filt3m:
    per_day_m = st.slider("Max/day", 1, 5, 3, key="monthly_cards_per_day")
cards = build_transit_cards_for_range(month_days[0], len(month_days), tz_in, ay_mode, strict_kp, sectors_m, m_start_t, m_end_t, get_rules(), st.session_state.kp_premium, st.session_state.net_threshold, planets_filter=planets_pick_m, aspects_filter=aspects_pick_m, per_day_limit=per_day_m)
render_cards(cards, "Upcoming planetary movements affecting sectors:")


        if cards:
            options_m = [f"{c['date']} — {c['event']}" for c in cards]
            pick_m = st.selectbox("Select a transit", options_m, key="monthly_transit_pick")
            sel_m = cards[options_m.index(pick_m)]
            parts = sel_m['event'].split()
            asp_type = parts[1].capitalize() if len(parts) > 1 else "Conjunction"
            days_m = _duration_days(asp_type)
            st.caption(f"Window: {sel_m['date']} for ~{days_m} days")
            rank_win_m, meta_m = analyze_transit_window(sel_m['date'], days_m, sectors_m, tz_in, m_start_t, m_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold, ay_mode, strict_kp)
            if not rank_win_m.empty:
                st.markdown("**Sector ranking during transit window**")
                st.dataframe(rank_win_m[['Sector','TotalNet','Avg/Stock','Confidence']], use_container_width=True)
                sec_choice_m = st.selectbox("Sector for stock breakdown", rank_win_m['Sector'].tolist(), key="monthly_transit_sector_choice")
                stock_df_m = stock_breakdown_for_sector_over_window(sec_choice_m, sectors_m, sel_m['date'], days_m, tz_in, m_start_t, m_end_t, RULES_CURRENT, st.session_state.kp_premium, ay_mode, strict_kp)
                st.markdown(f"**Stocks in {sec_choice_m} over transit window**")
                st.dataframe(stock_df_m, use_container_width=True)
            else:
                st.info("No data for this window.")

        # Select a transit to analyze
        if cards:
            options = [f"{c['date']} — {c['event']}" for c in cards]
            pick = st.selectbox("Select a transit", options, key="weekly_transit_pick")
            sel = cards[options.index(pick)]
            # Estimate days from aspect type in text (simple parse)
            parts = sel['event'].split()
            asp_type = parts[1].capitalize() if len(parts) > 1 else "Conjunction"
            days = _duration_days(asp_type)
            st.caption(f"Window: {sel['date']} for ~{days} days")
            # Sector ranking for the transit window
            rank_win, meta = analyze_transit_window(sel['date'], days, sectors_w, tz_in, w_start_t, w_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold, ay_mode, strict_kp)
            if not rank_win.empty:
                st.markdown("**Sector ranking during transit window**")
                st.dataframe(rank_win[['Sector','TotalNet','Avg/Stock','Confidence']], use_container_width=True)
                # Drilldown sector -> stocks
                sec_choice = st.selectbox("Sector for stock breakdown", rank_win['Sector'].tolist(), key="weekly_transit_sector_choice")
                stock_df = stock_breakdown_for_sector_over_window(sec_choice, sectors_w, sel['date'], days, tz_in, w_start_t, w_end_t, RULES_CURRENT, st.session_state.kp_premium, ay_mode, strict_kp)
                st.markdown(f"**Stocks in {sec_choice} over transit window**")
                st.dataframe(stock_df, use_container_width=True)
            else:
                st.info("No data for this window.")



    c1, c2 = st.columns(2)
    with c1:
        day_pick = st.selectbox("Pick a day", [str(d) for d in days_py], key="weekly_day_pick")
    with c2:
        sector_pick = st.selectbox("Pick a sector", list(sectors_w.keys()), key="weekly_sector_pick")

    dsel = pd.to_datetime(day_pick).date()
    asp_sel, kp_sel = cached_streams_for_date(dsel, tz_in, ay_mode, strict_kp)
    tz = pytz.timezone(tz_in)
    start_local = tz.localize(datetime.combine(dsel, w_start_t))
    end_local = tz.localize(datetime.combine(dsel, w_end_t))
    if end_local <= start_local: end_local = end_local + pd.Timedelta(days=1)

    dfA = asp_sel.copy()
    dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
    dfA = dfA[(dfA["TimeLocal"] >= start_local) & (dfA["TimeLocal"] < end_local)].copy()
    dfK = kp_sel.copy()
    dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
    dfK = dfK[(dfK["DT"] >= start_local) & (dfK["DT"] < end_local)].copy()

    symbols_w = sectors_w.get(sector_pick, [])
    if symbols_w:
        acl = "BANKNIFTY" if sector_pick == "BANKNIFTY" else "NIFTY"
        rows = []
        for sym in symbols_w:
            sA = [score_event(r, acl, get_rules()) for _, r in dfA.iterrows()]
            sK = [score_kp_only(r, acl, get_rules()) for _, r in dfK.iterrows()] if not dfK.empty else []
            sigA = [classify_score(s, get_rules()) for s in sA]
            sigK = [classify_score(s, get_rules()) for s in sK]
            signals = sigA + sigK
            rows.append({"Symbol": sym, "Bullish": signals.count("Bullish"), "Bearish": signals.count("Bearish"), "Neutral": signals.count("Neutral")})
        sym_df = pd.DataFrame(rows).sort_values(["Bullish","Bearish"], ascending=[False,True])
        st.markdown(f"**{sector_pick} — {day_pick} (window {w_start_t}–{w_end_t})**")
        st.dataframe(sym_df, use_container_width=True)
    else:
        st.info("No symbols configured for this sector.")

# -------- Monthly Outlook --------
with tabs[5]:
    st.subheader("🗓️ Monthly Outlook — sector bias by date")
    m1, m2, m3 = st.columns([1,1,1])
    with m1:
        m_start_t = st.time_input("Start Time", value=dtime(9,15), key="month_start_time")
    with m2:
        m_end_t = st.time_input("End Time", value=dtime(15,30), key="month_end_time")
    with m3:
        month_anchor = st.date_input("Month of", value=date_in, key="month_anchor")

    first_day = pd.Timestamp(month_anchor).replace(day=1)
    next_month = (first_day + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
    last_day = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
    day = first_day.date()
    month_days = []
    while day <= last_day:
        month_days.append(day)
        day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()

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
    with st.expander("Edit sectors (optional)"):
        sectors_json_m = st.text_area("Sectors dict", value=str(DEFAULT_SECTORS), height=140, key="sectors_monthly_json")
        try:
            import ast
            sectors_m = ast.literal_eval(sectors_json_m)
        except Exception:
            sectors_m = DEFAULT_SECTORS
            st.warning("Sector mapping parse failed; using defaults.")

    rows = []
    with st.spinner("Computing monthly sector ranks..."):
        for d in month_days:
            rdf = rank_for_single_date(d, sectors_m, tz_in, m_start_t, m_end_t,
                                       st.session_state.kp_premium, st.session_state.net_threshold,
                                       RULES_CURRENT, ay_mode, strict_kp)
            if rdf.empty:
                rows.append({"Date": str(d), "Top Bullish": "-", "NetScore": 0, "Top Bearish": "-", "BearScore": 0})
            else:
                top_bull = rdf.iloc[0]; bot_bear = rdf.sort_values("NetScore", ascending=True).iloc[0]
                rows.append({"Date": str(d), "Top Bullish": top_bull["Sector"], "NetScore": top_bull["NetScore"],
                             "Top Bearish": bot_bear["Sector"], "BearScore": bot_bear["NetScore"]})
    month_df = pd.DataFrame(rows)
    st.dataframe(month_df, use_container_width=True)

    with st.expander("🔭 This Month — Major Transits", expanded=False):
        cards = build_transit_cards_for_range(month_days[0], len(month_days), tz_in, ay_mode, strict_kp, sectors_m, m_start_t, m_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold)
        
st.markdown("**Upcoming planetary movements affecting sectors:**")
filt1, filt2, filt3 = st.columns([2,2,1])
with filt1:
    planets_pick = st.multiselect("Planets", ['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], default=['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Rahu','Ketu'], key="weekly_planets_filter")
with filt2:
    aspects_pick = st.multiselect("Aspects", ['Conjunction','Opposition','Square','Trine','Sextile'], default=['Conjunction','Opposition','Square','Trine','Sextile'], key="weekly_aspects_filter")
with filt3:
    per_day = st.slider("Max/day", 1, 5, 3, key="weekly_cards_per_day")
cards = build_transit_cards_for_range(days_py[0], 7, tz_in, ay_mode, strict_kp, sectors_w, w_start_t, w_end_t, get_rules(), st.session_state.kp_premium, st.session_state.net_threshold, planets_filter=planets_pick, aspects_filter=aspects_pick, per_day_limit=per_day)
render_cards(cards, "Upcoming planetary movements affecting sectors:")

        if cards:
            options_m = [f"{c['date']} — {c['event']}" for c in cards]
            pick_m = st.selectbox("Select a transit", options_m, key="monthly_transit_pick")
            sel_m = cards[options_m.index(pick_m)]
            parts = sel_m['event'].split()
            asp_type = parts[1].capitalize() if len(parts) > 1 else "Conjunction"
            days_m = _duration_days(asp_type)
            st.caption(f"Window: {sel_m['date']} for ~{days_m} days")
            rank_win_m, meta_m = analyze_transit_window(sel_m['date'], days_m, sectors_m, tz_in, m_start_t, m_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold, ay_mode, strict_kp)
            if not rank_win_m.empty:
                st.markdown("**Sector ranking during transit window**")
                st.dataframe(rank_win_m[['Sector','TotalNet','Avg/Stock','Confidence']], use_container_width=True)
                sec_choice_m = st.selectbox("Sector for stock breakdown", rank_win_m['Sector'].tolist(), key="monthly_transit_sector_choice")
                stock_df_m = stock_breakdown_for_sector_over_window(sec_choice_m, sectors_m, sel_m['date'], days_m, tz_in, m_start_t, m_end_t, RULES_CURRENT, st.session_state.kp_premium, ay_mode, strict_kp)
                st.markdown(f"**Stocks in {sec_choice_m} over transit window**")
                st.dataframe(stock_df_m, use_container_width=True)
            else:
                st.info("No data for this window.")

        # Select a transit to analyze
        if cards:
            options = [f"{c['date']} — {c['event']}" for c in cards]
            pick = st.selectbox("Select a transit", options, key="weekly_transit_pick")
            sel = cards[options.index(pick)]
            # Estimate days from aspect type in text (simple parse)
            parts = sel['event'].split()
            asp_type = parts[1].capitalize() if len(parts) > 1 else "Conjunction"
            days = _duration_days(asp_type)
            st.caption(f"Window: {sel['date']} for ~{days} days")
            # Sector ranking for the transit window
            rank_win, meta = analyze_transit_window(sel['date'], days, sectors_w, tz_in, w_start_t, w_end_t, RULES_CURRENT, st.session_state.kp_premium, st.session_state.net_threshold, ay_mode, strict_kp)
            if not rank_win.empty:
                st.markdown("**Sector ranking during transit window**")
                st.dataframe(rank_win[['Sector','TotalNet','Avg/Stock','Confidence']], use_container_width=True)
                # Drilldown sector -> stocks
                sec_choice = st.selectbox("Sector for stock breakdown", rank_win['Sector'].tolist(), key="weekly_transit_sector_choice")
                stock_df = stock_breakdown_for_sector_over_window(sec_choice, sectors_w, sel['date'], days, tz_in, w_start_t, w_end_t, RULES_CURRENT, st.session_state.kp_premium, ay_mode, strict_kp)
                st.markdown(f"**Stocks in {sec_choice} over transit window**")
                st.dataframe(stock_df, use_container_width=True)
            else:
                st.info("No data for this window.")



    # Calendar heatmap by top-sector NetScore (pandas Styler; no extra installs)
    disp_df, styled = build_calendar_table(month_days, month_df[['Date','NetScore']])
    st.dataframe(styled, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        day_pick_m = st.selectbox("Pick a date", [str(d) for d in month_days], key="monthly_day_pick")
    with c2:
        sector_pick_m = st.selectbox("Pick a sector", list(sectors_m.keys()), key="monthly_sector_pick")

    dsel = pd.to_datetime(day_pick_m).date()
    asp_sel, kp_sel = cached_streams_for_date(dsel, tz_in, ay_mode, strict_kp)
    tz = pytz.timezone(tz_in)
    start_local = tz.localize(datetime.combine(dsel, m_start_t))
    end_local = tz.localize(datetime.combine(dsel, m_end_t))
    if end_local <= start_local: end_local = end_local + pd.Timedelta(days=1)

    dfA = asp_sel.copy()
    dfA["TimeLocal"] = pd.to_datetime(dfA["Time"], format="%Y-%m-%d %H:%M").apply(lambda x: tz.localize(x))
    dfA = dfA[(dfA["TimeLocal"] >= start_local) & (dfA["TimeLocal"] < end_local)].copy()
    dfK = kp_sel.copy()
    dfK["DT"] = pd.to_datetime(dfK["Date"] + " " + dfK["Time"]).apply(lambda x: tz.localize(x))
    dfK = dfK[(dfK["DT"] >= start_local) & (dfK["DT"] < end_local)].copy()

    symbols_m = sectors_m.get(sector_pick_m, [])
    if symbols_m:
        acl = "BANKNIFTY" if sector_pick_m == "BANKNIFTY" else "NIFTY"
        rows = []
        for sym in symbols_m:
            sA = [score_event(r, acl, get_rules()) for _, r in dfA.iterrows()]
            sK = [score_kp_only(r, acl, get_rules()) for _, r in dfK.iterrows()] if not dfK.empty else []
            sigA = [classify_score(s, get_rules()) for s in sA]
            sigK = [classify_score(s, get_rules()) for s in sK]
            signals = sigA + sigK
            rows.append({"Symbol": sym, "Bullish": signals.count("Bullish"), "Bearish": signals.count("Bearish"), "Neutral": signals.count("Neutral")})
        sym_df = pd.DataFrame(rows).sort_values(["Bullish","Bearish"], ascending=[False,True])
        st.markdown(f"**{sector_pick_m} — {day_pick_m} (window {m_start_t}–{m_end_t})**")
        st.dataframe(sym_df, use_container_width=True)
    else:
        st.info("No symbols configured for this sector.")
