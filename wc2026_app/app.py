"""
FIFA World Cup 2026 — Prediction App
Refined model: Dixon-Coles correction + recency weighting +
               WC-calibrated base rates + form factor + Elo
"""

import streamlit as st
import pandas as pd
import numpy as np
import urllib.request, io
from scipy.stats import poisson
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS – North American 2026 Vibrant Neon Aesthetic
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

/* ── Design Tokens ── */
:root {
  --bg-dark: #070913;
  --panel-bg: rgba(22, 28, 45, 0.45);
  --panel-border: rgba(255, 255, 255, 0.08);
  --text-primary: #f8fafc;
  --text-muted: #94a3b8;
  
  /* 2026 Host Country Palette */
  --wc-purple: #4f46e5;
  --wc-cyan: #06b6d4;
  --wc-pink: #f43f5e;
  --wc-gold: #f59e0b;
  --success: #10b981;
}

/* ── Global Formatting Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.block-container { 
    padding: 2rem max(2vw, 1.5rem) !important; 
    max-width: 1300px !important; 
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Host-Inspired Branding Hero ── */
.hero-container {
    background: radial-gradient(circle at 90% 10%, rgba(79, 70, 229, 0.25) 0%, transparent 45%),
                radial-gradient(circle at 10% 90%, rgba(244, 63, 94, 0.2) 0%, transparent 45%),
                linear-gradient(135deg, #0f1225 0%, #070913 100%);
    border: 1px solid var(--panel-border);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
}
.hero-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.hero-title-group h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    line-height: 1.1;
    background: linear-gradient(135deg, #fff 20%, var(--text-muted) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: clamp(0.95rem, 1.5vw, 1.1rem);
    margin-top: 0.5rem;
    font-weight: 400;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.2rem;
}
.wc-badge {
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Section Container Cards ── */
.section-card {
    background: var(--panel-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 2rem;
}
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--wc-cyan);
    margin-bottom: 1.25rem;
}

/* ── Dropdowns / Input Styling ── */
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 0.2rem 0.5rem;
}
.stSelectbox > div > div:hover { border-color: var(--wc-purple) !important; }

/* ── Score & Match Predictor Layout ── */
.score-flex-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}
.team-card {
    flex: 1;
    min-width: 260px;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}
.team-card:hover {
    transform: translateY(-2px);
}
.team-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.25rem;
}
.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 6.5rem;
    font-weight: 700;
    line-height: 1;
    color: #fff;
    margin: 1rem 0;
    text-shadow: 0 10px 25px rgba(0,0,0,0.4);
}
.vs-divider {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-muted);
    opacity: 0.5;
    padding: 0 1rem;
}
.stat-sub-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stat-sub-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--wc-cyan);
    margin-top: 0.2rem;
}

/* ── Dynamic Result Banner ── */
.winner-banner {
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 2rem 0;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.winner-home { background: linear-gradient(90deg, rgba(6,182,212,0.15), rgba(79,70,229,0.15)); border: 1px solid var(--wc-cyan); color: #cffafe; }
.winner-away { background: linear-gradient(90deg, rgba(244,63,94,0.15), rgba(79,70,229,0.15)); border: 1px solid var(--wc-pink); color: #ffe4e6; }
.winner-draw { background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(30,41,59,0.15)); border: 1px solid var(--wc-gold); color: #fef3c7; }

/* ── Statistics Micro-Flex Pills ── */
.pills-container {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}
.metric-pill {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    min-width: 120px;
    text-align: center;
}
.metric-pill-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.1;
}
.metric-pill-lbl {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Distribution Bars ── */
.prob-container { margin: 0.5rem 0; width: 100%; }
.prob-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; }
.prob-label { width: 60px; font-size: 0.8rem; color: var(--text-muted); text-align: right; font-weight: 500; }
.prob-bar-wrap { flex: 1; background: rgba(255,255,255,0.05); border-radius: 100px; height: 12px; overflow: hidden; }
.prob-bar-inner { height: 12px; border-radius: 100px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
.bar-home { background: linear-gradient(90deg, var(--wc-purple), var(--wc-cyan)); }
.bar-draw { background: linear-gradient(90deg, #64748b, #94a3b8); }
.bar-away { background: linear-gradient(90deg, var(--wc-pink), #fda4af); }
.prob-pct { width: 50px; font-size: 0.85rem; font-weight: 700; color: #fff; }

/* ── Interactive Grid Matrix ── */
.matrix-cell {
    display: inline-block;
    width: 36px; height: 36px;
    line-height: 36px;
    text-align: center;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 2px;
}

/* ── Action Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--wc-purple) 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    box-shadow: 0 10px 20px -10px rgba(79, 70, 229, 0.5) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px -8px rgba(79, 70, 229, 0.7) !important;
}

/* ── Modernized Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    padding: 0 !important;
    gap: 0.5rem !important;
    border-bottom: 1px solid var(--panel-border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--wc-cyan) !important;
    border-bottom: 2px solid var(--wc-cyan) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--panel-bg) !important;
    border: 1px solid var(--panel-border) !important;
    border-top: none !important;
    border-radius: 0 0 20px 20px !important;
    padding: 2rem !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

/* ── Structural Table / DataFrame Tweaks ── */
div[data-testid="stDataFrame"] { background: transparent !important; }
hr { border-color: var(--panel-border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading historical match data…")
def load_and_train():
    BASE = "https://raw.githubusercontent.com/martj42/international_results/master"

    def fetch(url):
        return pd.read_csv(io.StringIO(
            urllib.request.urlopen(url, timeout=25).read().decode()))

    raw = fetch(f"{BASE}/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])

    # ── Tournament weights ──────────────────────────────────────
    TW = {
        "FIFA World Cup": 3.0, "Copa América": 2.5, "UEFA Euro": 2.5,
        "African Cup of Nations": 2.2, "Gold Cup": 2.0, "AFC Asian Cup": 2.0,
        "FIFA World Cup qualification": 1.5, "UEFA Euro qualification": 1.5,
        "Friendly": 0.6,
    }

    # ── Elo (2000–present, tournament-weighted, goal-diff multiplier) ──
    elo = {}
    df_elo = raw.dropna(subset=["home_score","away_score"]).copy()
    df_elo = df_elo[df_elo["date"] >= "2000-01-01"].sort_values("date")
    df_elo[["home_score","away_score"]] = df_elo[["home_score","away_score"]].astype(int)

    for _, row in df_elo.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        k = 40 * TW.get(row["tournament"], 1.0)
        gd = abs(hs - as_)
        gd_m = 1 if gd<=1 else (1.5 if gd==2 else (1.75 if gd==3 else 2.0))
        eh = elo.get(h, 1500); ea = elo.get(a, 1500)
        exp_h = 1 / (1 + 10**((ea - eh) / 400))
        sh = 1.0 if hs>as_ else (0.0 if hs<as_ else 0.5)
        elo[h] = eh + k * gd_m * (sh - exp_h)
        elo[a] = ea + k * gd_m * ((1-sh) - (1-exp_h))

    # ── Attack / Defense strengths (recency-weighted, 2019+) ──────
    recent = raw[(raw["date"] >= "2019-01-01") &
                 raw["home_score"].notna()].copy()
    recent[["home_score","away_score"]] = recent[["home_score","away_score"]].astype(int)
    max_date = recent["date"].max()
    recent["days_ago"] = (max_date - recent["date"]).dt.days
    recent["time_w"]   = np.exp(-0.003 * recent["days_ago"])
    recent["tourn_w"]  = recent["tournament"].map(TW).fillna(1.0)
    recent["weight"]   = recent["time_w"] * recent["tourn_w"]

    wc = recent[recent["tournament"] == "FIFA World Cup"]
    LAH_WC = wc["home_score"].mean() if len(wc) > 20 else 1.36
    LAA_WC = wc["away_score"].mean() if len(wc) > 20 else 1.21
    LA_BASE = (LAH_WC + LAA_WC) / 2

    all_teams = set(recent["home_team"]) | set(recent["away_team"])
    attack = {}; defense = {}
    form   = {}

    for team in all_teams:
        hm = recent[recent["home_team"] == team]
        aw = recent[recent["away_team"] == team]

        gf_h = (hm["home_score"] * hm["weight"]).sum()
        gf_a = (aw["away_score"] * aw["weight"]).sum()
        ga_h = (hm["away_score"] * hm["weight"]).sum()
        ga_a = (aw["home_score"] * aw["weight"]).sum()
        w_h  = hm["weight"].sum()
        w_a  = aw["weight"].sum()
        total_w = w_h + w_a

        if total_w > 2:
            avg_gf = (gf_h + gf_a) / total_w
            avg_ga = (ga_h + ga_a) / total_w
            attack[team]  = avg_gf / LA_BASE
            defense[team] = avg_ga / LA_BASE
        else:
            attack[team]  = 1.0
            defense[team] = 1.0

        games = recent[(recent["home_team"]==team)|(recent["away_team"]==team)].tail(5)
        pts = 0; possible = 0
        for _, g in games.iterrows():
            possible += 3
            if g["home_team"] == team:
                if g["home_score"] > g["away_score"]:   pts += 3
                elif g["home_score"] == g["away_score"]: pts += 1
            else:
                if g["away_score"] > g["home_score"]:   pts += 3
                elif g["home_score"] == g["away_score"]: pts += 1
        form[team] = pts / possible if possible > 0 else 0.5

    return elo, attack, defense, form, LAH_WC, LAA_WC


# ─────────────────────────────────────────────────────────────────
# TEAM CONSTANTS
# ─────────────────────────────────────────────────────────────────
_WC_TEAMS_CORE = [
    "Mexico","South Africa","South Korea","Czechia",
    "Canada","Switzerland","Qatar","Bosnia and Herzegovina",
    "Brazil","Morocco","Haiti","Scotland",
    "USA","Paraguay","Australia","Türkiye",
    "Germany","Curaçao","Côte d'Ivoire","Ecuador",
    "Netherlands","Japan","Sweden","Tunisia",
    "Belgium","Egypt","Iran","New Zealand",
    "Spain","Cabo Verde","Saudi Arabia","Uruguay",
    "France","Senegal","Norway","Iraq",
    "Argentina","Algeria","Austria","Jordan",
    "Portugal","DR Congo","Uzbekistan","Colombia",
    "England","Croatia","Ghana","Panama",
]

_ALL_NATIONS = [
    "Afghanistan","Albania","Algeria","American Samoa","Andorra","Angola",
    "Antigua and Barbuda","Argentina","Armenia","Aruba","Australia","Austria",
    "Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium",
    "Belize","Benin","Bermuda","Bhutan","Bolivia","Bosnia and Herzegovina",
    "Botswana","Brazil","British Virgin Islands","Brunei","Bulgaria","Burkina Faso",
    "Burundi","Cambodia","Cameroon","Canada","Cape Verde","Cayman Islands",
    "Central African Republic","Chad","Chile","China PR","Colombia","Comoros",
    "Congo","Cook Islands","Costa Rica","Croatia","Cuba","Curaçao","Cyprus",
    "Czech Republic","DR Congo","Denmark","Djibouti","Dominica","Dominican Republic",
    "Ecuador","Egypt","El Salvador","England","Equatorial Guinea","Eritrea",
    "Estonia","Eswatini","Ethiopia","Faroe Islands","Fiji","Finland","France",
    "French Guiana","Gabon","Gambia","Georgia","Germany","Ghana","Gibraltar",
    "Greece","Greenland","Grenada","Guadeloupe","Guam","Guatemala","Guinea",
    "Guinea-Bissau","Guyana","Haiti","Honduras","Hong Kong","Hungary","Iceland",
    "India","Indonesia","Iran","Iraq","Israel","Italy","Ivory Coast","Jamaica",
    "Japan","Jordan","Kazakhstan","Kenya","Kiribati","Kosovo","Kuwait","Kyrgyzstan",
    "Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein",
    "Lithuania","Luxembourg","Macau","Madagascar","Malawi","Malaysia","Maldives",
    "Mali","Malta","Marshall Islands","Martinique","Mauritania","Mauritius",
    "Mexico","Micronesia","Moldova","Mongolia","Montenegro","Montserrat","Morocco",
    "Mozambique","Myanmar","Namibia","Nepal","Netherlands","New Caledonia",
    "New Zealand","Nicaragua","Niger","Nigeria","North Korea","North Macedonia",
    "Northern Ireland","Norway","Oman","Pakistan","Palestine","Panama",
    "Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal",
    "Puerto Rico","Qatar","Republic of Ireland","Romania","Russia","Rwanda",
    "Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines",
    "Samoa","San Marino","Saudi Arabia","Scotland","Senegal","Serbia","Seychelles",
    "Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia",
    "South Africa","South Korea","South Sudan","Spain","Sri Lanka","Sudan",
    "Suriname","Sweden","Switzerland","Syria","Tahiti","Taiwan","Tajikistan",
    "Tanzania","Thailand","Timor-Leste","Togo","Tonga","Trinidad and Tobago",
    "Tunisia","Turkey","Turkmenistan","Turks and Caicos Islands","Uganda","Ukraine",
    "United Arab Emirates","United States","Uruguay","Uzbekistan","Vanuatu",
    "Venezuela","Vietnam","Wales","Yemen","Zambia","Zimbabwe",
]

_wc_set = set(_WC_TEAMS_CORE)
_extra = [t for t in sorted(_ALL_NATIONS) if t not in _wc_set and t != "USA"]
WC_TEAMS = _WC_TEAMS_CORE + ["────────────────"] + _extra

NAME_MAP = {
    "USA":                      "United States",
    "Cabo Verde":               "Cape Verde",
    "Côte d'Ivoire":            "Ivory Coast",
    "Ivory Coast":              "Ivory Coast",
    "Czechia":                  "Czech Republic",
    "Türkiye":                  "Turkey",
    "Bosnia and Herzegovina":   "Bosnia and Herzegovina",
    "DR Congo":                 "DR Congo",
}

PLAYOFF_ELO = {}

FLAG_MAP = {
    "Algeria":"🇩🇿","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Brazil":"🇧🇷","Cabo Verde":"🇨🇻","Cape Verde":"🇨🇻","Canada":"🇨🇦",
    "Colombia":"🇨🇴","Croatia":"🇭🇷","Curaçao":"🇨🇼","Côte d'Ivoire":"🇨🇮","Ivory Coast":"🇨🇮",
    "Ecuador":"🇪🇨","Egypt":"🇪🇬","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷",
    "Germany":"🇩🇪","Ghana":"🇬🇭","Haiti":"🇮🇹","Iran":"🇮🇷",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
    "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦",
    "South Korea":"🇰🇷","Spain":"🇪🇸","Switzerland":"🇨🇭","Tunisia":"🇹🇳",
    "Uruguay":"🇺🇾","USA":"🇺🇸","United States":"🇺🇸","Uzbekistan":"🇺🇿",
    "Bosnia and Herzegovina":"🇧🇦","Sweden":"🇸🇪","Türkiye":"🇹🇷","Turkey":"🇹🇷",
    "Czechia":"🇨🇿","Czech Republic":"🇨🇿","Iraq":"🇮🇶","DR Congo":"🇨🇩",
    "Afghanistan":"🇦🇫","Albania":"🇦🇱","American Samoa":"🇦🇸","Andorra":"🇦🇩",
    "Angola":"🇦🇴","Antigua and Barbuda":"🇦🇬","Armenia":"🇦🇲","Aruba":"🇦🇼",
    "Azerbaijan":"🇦🇿","Bahamas":"🇧🇸","Bahrain":"🇧🇭","Bangladesh":"🇧🇩",
    "Barbados":"🇧🇧","Belarus":"🇧🇾","Belize":"🇧🇿","Benin":"🇧🇯",
    "Bermuda":"🇧🇲","Bhutan":"🇧🇹","Bolivia":"🇧🇴","Botswana":"🇧🇼",
    "British Virgin Islands":"🇻🇬","Brunei":"🇧🇳","Bulgaria":"🇧🇬",
    "Burkina Faso":"🇧🇫","Burundi":"🇧🇮","Cambodia":"🇰🇭","Cameroon":"🇨🇲",
    "Cayman Islands":"🇰🇾","Central African Republic":"🇨🇫","Chad":"🇹🇩",
    "Chile":"🇨🇱","China PR":"🇨🇳","Congo":"🇨🇬","Cook Islands":"🇨🇰",
    "Costa Rica":"🇨🇷","Cuba":"🇨🇺","Cyprus":"🇨🇾","Denmark":"🇩🇰",
    "Djibouti":"🇩🇯","Dominica":"🇩🇲","Dominican Republic":"🇩🇴","El Salvador":"🇸🇻",
    "Equatorial Guinea":"🇬🇶","Eritrea":"🇪🇷","Estonia":"🇪🇪","Eswatini":"🇸🇿",
    "Ethiopia":"🇪🇹","Faroe Islands":"🇫🇴","Fiji":"🇫🇯","Finland":"🇫🇮",
    "French Guiana":"🇬🇫","Gabon":"🇬🇦","Gambia":"🇬🇲","Georgia":"🇬🇪",
    "Gibraltar":"🇬🇮","Greece":"🇬🇷","Greenland":"🇬🇱","Grenada":"🇬🇩",
    "Guadeloupe":"🇬🇵","Guam":"🇬🇺","Guatemala":"🇬🇹","Guinea":"🇬🇳",
    "Guinea-Bissau":"🇬🇼","Guyana":"🇬🇾","Honduras":"🇭🇳","Hong Kong":"🇭🇰",
    "Hungary":"🇭🇺","Iceland":"🇮🇸","India":"🇮🇳","Indonesia":"🇮🇩",
    "Israel":"🇮🇱","Italy":"🇮🇹","Jamaica":"🇯🇲","Kazakhstan":"🇰🇿",
    "Kenya":"🇰🇪","Kiribati":"🇰🇮","Kosovo":"🇽🇰","Kuwait":"🇰🇼",
    "Kyrgyzstan":"🇰🇬","Laos":"🇱🇦","Latvia":"🇱🇻","Lebanon":"🇱🇧",
    "Lesotho":"🇱🇸","Liberia":"🇱🇷","Libya":"🇱🇾","Liechtenstein":"🇱🇮",
    "Lithuania":"🇱🇹","Luxembourg":"🇱🇺","Macau":"🇲🇴","Madagascar":"🇲🇬",
    "Malawi":"🇲🇼","Malaysia":"🇲🇾","Maldives":"🇲🇻","Mali":"🇲🇱",
    "Malta":"🇲🇹","Marshall Islands":"🇲🇭","Martinique":"🇲🇶","Mauritania":"🇲🇷",
    "Mauritius":"🇲🇺","Micronesia":"🇫🇲","Moldova":"🇲🇩","Mongolia":"🇲🇳",
    "Montenegro":"🇲🇪","Montserrat":"🇲🇸","Mozambique":"🇲🇿","Myanmar":"🇲🇲",
    "Namibia":"🇳🇦","Nepal":"🇳🇵","New Caledonia":"🇳🇨","Nicaragua":"🇳🇮",
    "Niger":"🇳🇪","Nigeria":"🇳🇬","North Korea":"🇰🇵","North Macedonia":"🇲🇰",
    "Northern Ireland":"🏴󠁧󠁢󠁮󠁩󠁲󠁿","Oman":"🇴🇲","Pakistan":"🇵🇰","Palestine":"🇵🇸",
    "Papua New Guinea":"🇵🇬","Peru":"🇵🇪","Philippines":"🇵🇭","Poland":"🇵🇱",
    "Puerto Rico":"🇵🇷","Republic of Ireland":"🇮🇪","Romania":"🇷🇴","Russia":"🇷🇺",
    "Rwanda":"🇷🇼","Saint Kitts and Nevis":"🇰🇳","Saint Lucia":"🇱🇨",
    "Saint Vincent and the Grenadines":"🇻🇨","Samoa":"🇼🇸","San Marino":"🇸🇲",
    "Serbia":"🇷🇸","Seychelles":"🇸🇨","Sierra Leone":"🇸🇱","Singapore":"🇸🇬",
    "Slovakia":"🇸🇰","Slovenia":"🇸🇮","Solomon Islands":"🇸🇧","Somalia":"🇸🇴",
    "South Sudan":"🇸🇸","Sri Lanka":"🇱🇰","Sudan":"🇸🇩","Suriname":"🇸🇷",
    "Syria":"🇸🇾","Tahiti":"🇵🇫","Taiwan":"🇹🇼","Tajikistan":"🇹🇯",
    "Tanzania":"🇹🇿","Thailand":"🇹🇭","Timor-Leste":"🇹🇱","Togo":"🇹🇬",
    "Tonga":"🇹🇴","Trinidad and Tobago":"🇹🇹","Turkmenistan":"🇹🇲",
    "Turks and Caicos Islands":"🇹🇨","Uganda":"🇺🇬","Ukraine":"🇺🇦",
    "United Arab Emirates":"🇦🇪","Vanuatu":"🇻🇺","Venezuela":"🇻🇪",
    "Vietnam":"🇻🇳","Wales":"🏴󠁧󠁢󠁷󠁬󠁳󠁿","Yemen":"🇾🇪","Zambia":"🇿🇲","Zimbabwe":"🇿🇼",
}


# ─────────────────────────────────────────────────────────────────
# REFINED PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
def predict_match(home, away, elo, attack, defense, form, LAH, LAA, is_knockout=False):
    def _elo(t):
        if t in PLAYOFF_ELO: return PLAYOFF_ELO[t]
        return elo.get(NAME_MAP.get(t, t), 1550)

    def _atk(t): return attack.get(NAME_MAP.get(t, t), 1.0)
    def _def(t): return defense.get(NAME_MAP.get(t, t), 1.0)
    def _form(t): return form.get(NAME_MAP.get(t, t), 0.5)

    elo_diff = _elo(home) - _elo(away)
    elo_factor = 1 + (elo_diff / 1000) * 0.45

    form_h = 0.95 + _form(home) * 0.1
    form_a = 0.95 + _form(away) * 0.1

    lam_h = max(0.25, min(LAH * _atk(home) * _def(away) * elo_factor * form_h, 5.5))
    lam_a = max(0.25, min(LAA * _atk(away) * _def(home) / elo_factor * form_a, 5.5))

    rho = -0.10
    def dc_correction(i, j, mu1, mu2, r):
        if   i==0 and j==0: return 1 - mu1*mu2*r
        elif i==1 and j==0: return 1 + mu2*r
        elif i==0 and j==1: return 1 + mu1*r
        elif i==1 and j==1: return 1 - r
        return 1.0

    N = 8
    prob_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            base = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            prob_matrix[i, j] = base * dc_correction(i, j, lam_h, lam_a, rho)

    prob_matrix = np.clip(prob_matrix, 0, None)
    prob_matrix /= prob_matrix.sum()

    idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    p_home = float(prob_matrix[np.tril_indices(N, -1)].sum())
    p_draw = float(np.trace(prob_matrix))
    p_away = float(prob_matrix[np.triu_indices(N, 1)].sum())

    if p_home > p_draw and p_home > p_away: result = "home"
    elif p_away > p_home and p_away > p_draw: result = "away"
    else: result = "draw"

    if is_knockout and result == "draw":
        result = "home" if _elo(home) >= _elo(away) else "away"

    total_xg = lam_h + lam_a
    def_ratio = (_def(home) + _def(away)) / 2
    corners = max(6, min(14, int(round(5.5 + total_xg * 1.6 + (1 - def_ratio) * 1.2))))

    competition = 1 - abs(p_home - p_away)
    yellows = max(1, min(7, int(round(1.8 + competition * 2.2))))
    red     = 1 if competition > 0.88 else 0

    return {
        "home_goals": int(idx[0]),
        "away_goals": int(idx[1]),
        "lam_h": round(lam_h, 2),
        "lam_a": round(lam_a, 2),
        "p_home": round(p_home * 100, 1),
        "p_draw": round(p_draw * 100, 1),
        "p_away": round(p_away * 100, 1),
        "result": result,
        "corners": corners,
        "yellows": yellows,
        "red": red,
        "prob_matrix": prob_matrix,
        "elo_home": int(_elo(home)),
        "elo_away": int(_elo(away)),
    }


# ─────────────────────────────────────────────────────────────────
# GROUP STAGE FULL PREDICTIONS (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running full tournament simulation…")
def run_full_predictions(_elo, _attack, _defense, _form, LAH, LAA):
    group_fixtures = knockout_fixtures = None
    for branch in ["main", "master"]:
        try:
            REPO_RAW = f"https://raw.githubusercontent.com/Aditya2022331060/world_cup_2026_predictor/{branch}"
            group_fixtures    = pd.read_csv(f"{REPO_RAW}/group_match.csv")
            knockout_fixtures = pd.read_csv(f"{REPO_RAW}/knokout_match.csv")
            break
        except Exception:
            continue
    if group_fixtures is None:
        try:
            group_fixtures    = pd.read_csv("group_match.csv")
            knockout_fixtures = pd.read_csv("knokout_match.csv")
        except Exception:
            return None, None

    gp = group_fixtures.copy()
    rows = [predict_match(r["home_team"], r["away_team"],
                          _elo, _attack, _defense, _form, LAH, LAA)
            for _, r in group_fixtures.iterrows()]
    gp["predicted_home_goals"] = [r["home_goals"] for r in rows]
    gp["predicted_away_goals"] = [r["away_goals"] for r in rows]
    gp["corners"]              = [r["corners"] for r in rows]
    gp["yellow_cards"]         = [r["yellows"] for r in rows]
    gp["red_cards"]            = [r["red"] for r in rows]
    gp["winning_team"]         = [r["result"] for r in rows]
    gp["xG_home"]              = [r["lam_h"] for r in rows]
    gp["xG_away"]              = [r["lam_a"] for r in rows]

    standings = {}
    for _, r in gp.iterrows():
        grp=r["group"]; h=r["home_team"]; a=r["away_team"]
        hg=r["predicted_home_goals"]; ag=r["predicted_away_goals"]
        for t in [h,a]:
            standings.setdefault(grp,{}).setdefault(t,{"pts":0,"gf":0,"ga":0,"gd":0})
        if   hg>ag: standings[grp][h]["pts"]+=3
        elif hg<ag: standings[grp][a]["pts"]+=3
        else:       standings[grp][h]["pts"]+=1; standings[grp][a]["pts"]+=1
        standings[grp][h]["gf"]+=hg; standings[grp][h]["ga"]+=ag; standings[grp][h]["gd"]+=hg-ag
        standings[grp][a]["gf"]+=ag; standings[grp][a]["ga"]+=hg; standings[grp][a]["gd"]+=ag-hg

    qualifiers={}; thirds=[]
    for grp,teams in standings.items():
        ranked=sorted(teams.items(),key=lambda x:(x[1]["pts"],x[1]["gd"],x[1]["gf"]),reverse=True)
        qualifiers[f"Winner Group {grp}"]=ranked[0][0]
        qualifiers[f"Runner-up Group {grp}"]=ranked[1][0]
        thirds.append((ranked[2][0],ranked[2][1]["pts"],ranked[2][1]["gd"],ranked[2][1]["gf"]))

    thirds_sorted=sorted(thirds,key=lambda x:(x[1],x[2],x[3]),reverse=True)
    best3_ctr=[0]

    def resolve(slot, ko_res):
        if slot in qualifiers: return qualifiers[slot]
        if "Best 3rd" in slot:
            i=best3_ctr[0]; best3_ctr[0]+=1
            return thirds_sorted[i][0] if i<len(thirds_sorted) else "TBD"
        for mid,(ph,pa,pw,_) in ko_res.items():
            if f"Winner Match {mid}"==slot: return pw
            if f"Loser Match {mid}"==slot:  return ph if pw==pa else pa
        return slot

    ko_res={}
    kp=knockout_fixtures.copy()
    hr_l=[]; ar_l=[]; hg_l=[]; ag_l=[]; c_l=[]; y_l=[]; r_l=[]; w_l=[]
    for _,row in knockout_fixtures.iterrows():
        mid=row["match_id"]
        ht=resolve(row["slot_home"],ko_res); at=resolve(row["slot_away"],ko_res)
        res=predict_match(ht,at,_elo,_attack,_defense,_form,LAH,LAA,is_knockout=True)
        wt=ht if res["result"]=="home" else at
        ko_res[mid]=(ht,at,wt,res["result"])
        hr_l.append(ht); ar_l.append(at)
        hg_l.append(res["home_goals"]); ag_l.append(res["away_goals"])
        c_l.append(res["corners"]); y_l.append(res["yellows"]); r_l.append(res["red"])
        w_l.append(res["result"])

    kp["home_team_resolved"]=hr_l; kp["away_team_resolved"]=ar_l
    kp["predicted_home_goals"]=hg_l; kp["predicted_away_goals"]=ag_l
    kp["corners"]=c_l; kp["yellow_cards"]=y_l; kp["red_cards"]=r_l
    kp["winning_team"]=w_l

    return gp, kp


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def prob_bar_html(p_home, p_draw, p_away):
    return f"""
    <div class="prob-container">
      <div class="prob-row">
        <span class="prob-label">Home</span>
        <div class="prob-bar-wrap"><div class="prob-bar-inner bar-home" style="width:{p_home}%"></div></div>
        <span class="prob-pct">{p_home}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">Draw</span>
        <div class="prob-bar-wrap"><div class="prob-bar-inner bar-draw" style="width:{p_draw}%"></div></div>
        <span class="prob-pct">{p_draw}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">Away</span>
        <div class="prob-bar-wrap"><div class="prob-bar-inner bar-away" style="width:{p_away}%"></div></div>
        <span class="prob-pct">{p_away}%</span>
      </div>
    </div>"""

def score_matrix_html(pm, max_show=5):
    html = '<div style="overflow-x:auto">'
    html += '<table style="border-collapse:separate;border-spacing:4px;margin:0 auto">'
    header_style = 'style="text-align:center;font-size:0.75rem;color:#94a3b8;padding:4px 8px;font-weight:700;font-family:\'Space Grotesk\',sans-serif;"'
    html += f'<tr><td {header_style}></td>'
    for j in range(max_show): html += f'<td {header_style}>{j}</td>'
    html += '</tr>'
    max_p = pm[:max_show,:max_show].max()
    for i in range(max_show):
        html += f'<tr><td {header_style}>{i}</td>'
        for j in range(max_show):
            p = pm[i,j]
            if i > j:   bg = f"rgba(6,182,212,{0.15 + (p/max_p)*0.7})"
            elif i < j: bg = f"rgba(244,63,94,{0.15 + (p/max_p)*0.7})"
            else:       bg = f"rgba(245,158,11,{0.15 + (p/max_p)*0.7})"
            pct = f"{p*100:.1f}%"
            html += f'<td><div class="matrix-cell" style="background:{bg};color:#ffffff;border:1px solid rgba(255,255,255,0.05);">{pct}</div></td>'
        html += '</tr>'
    html += '</table></div>'
    return html


# ─────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
  <div class="hero-flex">
    <div class="hero-title-group">
      <h1>🏆 WORLD CUP 2026 PREDICTOR</h1>
      <div class="hero-subtitle">United States · Canada · Mexico • 48 Nations • 104 Matches</div>
      <div class="badge-row">
        <span class="wc-badge" style="background:rgba(79,70,229,0.2); color:#a5b4fc; border-color:rgba(79,70,229,0.4);">⚡ Poisson Engine</span>
        <span class="wc-badge" style="background:rgba(6,182,212,0.2); color:#22d3ee; border-color:rgba(6,182,212,0.4);">📊 Dixon-Coles Calibrated</span>
        <span class="wc-badge" style="background:rgba(244,63,94,0.2); color:#fca5a5; border-color:rgba(244,63,94,0.4);">⚽ Live Elo Weights</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training prediction model on 49,000+ historical matches…"):
    elo_data, atk, dfn, frm, LAH, LAA = load_and_train()

tab1, tab2, tab3 = st.tabs(["🆚  MATCH PREDICTOR", "📋  FULL TOURNAMENT", "⚙️  MODEL INFO"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Fixture Setup</div>', unsafe_allow_html=True)

    col_h, col_vs, col_a = st.columns([5, 1, 5])

    with col_h:
        home = st.selectbox("🏠 Home / Team 1", WC_TEAMS,
                            index=WC_TEAMS.index("Argentina"), key="home")
    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-family:\'Space Grotesk\',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text-muted)">VS</div>', unsafe_allow_html=True)
    with col_a:
        away_default = WC_TEAMS.index("France")
        away = st.selectbox("✈️ Away / Team 2", WC_TEAMS,
                            index=away_default, key="away")

    col_ko, col_btn, _ = st.columns([2, 2, 3])
    with col_ko:
        st.markdown("<br>", unsafe_allow_html=True)
        is_ko = st.toggle("Knockout Stage (Force Winner via Elo)", value=False)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ SIMULATE MATCH", use_container_width=True)

    st.markdown("---")

    _sep = "────────────────"
    if home == _sep or away == _sep:
        st.warning("⚠️ Please select a valid team (not the separator line).")
    elif home == away:
        st.warning("⚠️ Please select two different teams.")
    elif not predict_clicked and "last_prediction" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;color:var(--text-muted)">
          <div style="font-size:3.5rem;margin-bottom:1rem">⚽</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:#fff;letter-spacing:-0.5px">
            Configure Selection Above & Run Simulation
          </div>
          <div style="font-size:0.85rem;margin-top:0.4rem;color:var(--text-muted)">
            Calculations compute live scoreline probabilities using adjusted scoring dynamics.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        if predict_clicked:
            st.session_state["last_prediction"] = (home, away, is_ko)
        pred_home, pred_away, pred_ko = st.session_state.get("last_prediction", (home, away, is_ko))
        res = predict_match(pred_home, pred_away, elo_data, atk, dfn, frm, LAH, LAA, is_knockout=pred_ko)

        flag_h = FLAG_MAP.get(pred_home, "🏳")
        flag_a = FLAG_MAP.get(pred_away, "🏳")

        st.markdown(f"""
        <div class="score-flex-wrapper">
            <div class="team-card">
                <div class="team-name">{flag_h} {pred_home}</div>
                <div class="stat-sub-label" style="font-weight:600;">Elo {res['elo_home']}</div>
                <div class="score-number">{res['home_goals']}</div>
                <div class="stat-sub-label">Expected Goals</div>
                <div class="stat-sub-value">{res['lam_h']}</div>
            </div>
            <div class="vs-divider">:</div>
            <div class="team-card">
                <div class="team-name">{flag_a} {pred_away}</div>
                <div class="stat-sub-label" style="font-weight:600;">Elo {res['elo_away']}</div>
                <div class="score-number">{res['away_goals']}</div>
                <div class="stat-sub-label">Expected Goals</div>
                <div class="stat-sub-value">{res['lam_a']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if res["result"] == "home":
            banner_cls = "winner-home"
            banner_txt = f"🎯 Outcome Prediction: {pred_home} Victory"
        elif res["result"] == "away":
            banner_cls = "winner-away"
            banner_txt = f"🎯 Outcome Prediction: {pred_away} Victory"
        else:
            banner_cls = "winner-draw"
            banner_txt = "🎯 Outcome Prediction: Match Draw"
        st.markdown(f'<div class="winner-banner {banner_cls}">{banner_txt}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pills-container">
          <div class="metric-pill"><span class="metric-pill-val">{res['corners']}</span><span class="metric-pill-lbl">Est. Corners</span></div>
          <div class="metric-pill"><span class="metric-pill-val">{res['yellows']}</span><span class="metric-pill-lbl">Yellow Cards</span></div>
          <div class="metric-pill"><span class="metric-pill-val">{res['red']}</span><span class="metric-pill-lbl">Red Cards</span></div>
          <div class="metric-pill"><span class="metric-pill-val" style="color:var(--wc-cyan);">{res['p_home']}%</span><span class="metric-pill-lbl">Home Win %</span></div>
          <div class="metric-pill"><span class="metric-pill-val" style="color:var(--text-muted);">{res['p_draw']}%</span><span class="metric-pill-lbl">Draw %</span></div>
          <div class="metric-pill"><span class="metric-pill-val" style="color:var(--wc-pink);">{res['p_away']}%</span><span class="metric-pill-lbl">Away Win %</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_prob, col_matrix = st.columns([1, 1])
        with col_prob:
            st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
            st.markdown(prob_bar_html(res["p_home"], res["p_draw"], res["p_away"]),
                        unsafe_allow_html=True)

        with col_matrix:
            st.markdown('<div class="section-label">Exact Scoreline Matrix (5×5)</div>', unsafe_allow_html=True)
            st.markdown(score_matrix_html(res["prob_matrix"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — FULL TOURNAMENT
# ═══════════════════════════════════════════════════════════════════
with tab2:
    gp_df, kp_df = run_full_predictions(elo_data, atk, dfn, frm, LAH, LAA)

    if gp_df is None:
        st.info("📂 Place `group_match.csv` and `knokout_match.csv` in the app folder to see full tournament predictions.")
    else:
        sub1, sub2 = st.tabs(["📊 Group Stage Schedule", "👑 Knockout Tree Predictions"])

        with sub1:
            display_gp = gp_df[[
                "match_id","group","home_team","away_team",
                "predicted_home_goals","predicted_away_goals",
                "xG_home","xG_away","winning_team","corners","yellow_cards","red_cards"
            ]].rename(columns={
                "predicted_home_goals":"Home G","predicted_away_goals":"Away G",
                "xG_home":"xG H","xG_away":"xG A",
                "winning_team":"Result","yellow_cards":"Yellows","red_cards":"Reds"
            })
            st.dataframe(display_gp, use_container_width=True, hide_index=True,
                         column_config={
                             "xG H": st.column_config.NumberColumn(format="%.2f"),
                             "xG A": st.column_config.NumberColumn(format="%.2f"),
                         })
            csv = gp_df.to_csv(index=False)
            st.download_button("⬇️ Download group_predictions.csv", csv,
                               "group_predictions.csv", "text/csv")

        with sub2:
            display_kp = kp_df[[
                "match_id","round","home_team_resolved","away_team_resolved",
                "predicted_home_goals","predicted_away_goals",
                "winning_team","corners","yellow_cards","red_cards"
            ]].rename(columns={
                "home_team_resolved":"Home","away_team_resolved":"Away",
                "predicted_home_goals":"Home G","predicted_away_goals":"Away G",
                "winning_team":"Result","yellow_cards":"Yellows","red_cards":"Reds"
            })
            st.dataframe(display_kp, use_container_width=True, hide_index=True)
            csv2 = kp_df.to_csv(index=False)
            st.download_button("⬇️ Download knockout_predictions.csv", csv2,
                               "knockout_predictions.csv", "text/csv")

            final = kp_df[kp_df["round"]=="Final"]
            if len(final):
                f = final.iloc[0]
                champ = f["home_team_resolved"] if f["winning_team"]=="home" else f["away_team_resolved"]
                runner = f["away_team_resolved"] if f["winning_team"]=="home" else f["home_team_resolved"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 1px solid rgba(245,158,11,0.3);
                     border-radius:20px; padding:2rem; text-align:center; margin-top:2rem; box-shadow:0 15px 30px rgba(0,0,0,0.4);">
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:0.85rem; color:var(--wc-gold); letter-spacing:2px; font-weight:700; text-transform:uppercase;">
                    Simulated Tournament Champion
                  </div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:3.5rem; font-weight:700; color:#fff; margin:0.5rem 0;">
                    🏆 {FLAG_MAP.get(champ,'🏳')} {champ}
                  </div>
                  <div style="color:var(--text-muted); font-size:0.95rem;">
                    Defeated {runner} in the final matchup • Projected Scoreline: {f['predicted_home_goals']}-{f['predicted_away_goals']}
                  </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="max-width:800px">
        <div class="section-label">Engine Mechanics</div>
        <p style="color:var(--text-primary); line-height:1.75; font-size:0.95rem;">
            This prediction simulator applies a multi-layered <strong style="color:var(--wc-cyan)">Poisson regression structure</strong> balanced against long-term operational metrics:
        </p>

        <table style="width:100%; border-collapse:collapse; font-size:0.9rem; margin-top:1.5rem;">
          <tr style="border-bottom:1px solid var(--panel-border)">
            <td style="padding:1rem 0.5rem; color:var(--wc-cyan); font-weight:700; width:180px; font-family:'Space Grotesk',sans-serif;">Elo Tracking</td>
            <td style="padding:1rem 0.5rem; color:var(--text-muted)">Evaluated from historical data. Applies specialized structural weights reflecting tournament prominence alongside scale metrics matching match point adjustments.</td>
          </tr>
          <tr style="border-bottom:1px solid var(--panel-border)">
            <td style="padding:1rem 0.5rem; color:var(--wc-cyan); font-weight:700; font-family:'Space Grotesk',sans-serif;">Poisson Distribution</td>
            <td style="padding:1rem 0.5rem; color:var(--text-muted)">Cross-references calculated operational attack metrics against defense thresholds to forecast target probabilities based on World Cup specific neutral constants.</td>
          </tr>
          <tr style="border-bottom:1px solid var(--panel-border)">
            <td style="padding:1rem 0.5rem; color:var(--wc-cyan); font-weight:700; font-family:'Space Grotesk',sans-serif;">Dixon-Coles Setting</td>
            <td style="padding:1rem 0.5rem; color:var(--text-muted)">Compensates for standard distribution variations in low-scoring scenarios by scaling joint matrix configurations (rho = -0.10).</td>
          </tr>
          <tr style="border-bottom:1px solid var(--panel-border)">
            <td style="padding:1rem 0.5rem; color:var(--wc-cyan); font-weight:700; font-family:'Space Grotesk',sans-serif;">Time Decay Scaling</td>
            <td style="padding:1rem 0.5rem; color:var(--text-muted)">Implements exponential reduction models (lambda = 0.003 / day) targeting historical importance data to prioritize immediate match tracking.</td>
          </tr>
          <tr>
            <td style="padding:1rem 0.5rem; color:var(--wc-cyan); font-weight:700; font-family:'Space Grotesk',sans-serif;">Form Adjustments</td>
            <td style="padding:1rem 0.5rem; color:var(--text-muted)">Integrates final structural performance markers across consecutive fixtures while aligning secondary game parameters against championship benchmark data.</td>
          </tr>
        </table>

        <br><br>
        <div class="section-label">Source Feed</div>
        <p style="color:var(--text-muted); font-size:0.85rem;">
            Data sourced from <code style="background:rgba(255,255,255,0.05); padding:3px 8px; border-radius:6px; color:#fff;">martj42/international_results</code> tracking modern structural history logs.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FOOTER — Flexbox Profile Card
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:5rem; padding-top:2rem; border-top:1px solid var(--panel-border);
     display:flex; align-items:center; justify-content:center; gap:1rem;">
  <a href="https://github.com/Aditya2022331060" target="_blank"
     style="display:flex; align-items:center; gap:0.85rem; text-decoration:none;
            background:rgba(30,41,59,0.3); border:1px solid var(--panel-border); border-radius:100px;
            padding:0.6rem 1.4rem 0.6rem 0.6rem; transition:all 0.2s ease;"
     onmouseover="this.style.borderColor='rgba(6,182,212,0.5)'; this.style.boxShadow='0 0 20px rgba(6,182,212,0.15)'"
     onmouseout="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
    <img src="https://github.com/Aditya2022331060.png?size=40"
         style="width:40px; height:40px; border-radius:50%; border:2px solid var(--wc-purple); object-fit:cover"
         onerror="this.src='https://avatars.githubusercontent.com/u/Aditya2022331060?v=4'" />
    <div>
      <div style="color:#fff; font-weight:700; font-size:0.85rem; line-height:1.2; font-family:'Space Grotesk',sans-serif;">Aditya</div>
      <div style="color:var(--text-muted); font-size:0.75rem;">github.com/Aditya2022331060</div>
    </div>
  </a>
</div>
""", unsafe_allow_html=True)
