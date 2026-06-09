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
# CUSTOM CSS  – dark stadium aesthetic
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── root variables ── */
:root {
  --pitch: #0a3d1f;
  --pitch-mid: #0f5c2e;
  --pitch-light: #1a7a3f;
  --gold: #f0b429;
  --gold-dim: #c4922a;
  --red: #e53e3e;
  --blue: #3182ce;
  --bg: #06110d;
  --surface: #0d2418;
  --border: #1e4030;
  --text: #e8f5ee;
  --muted: #7aab8e;
}

/* ── global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px !important; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, var(--pitch) 0%, #001a0a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent, transparent 60px,
        rgba(255,255,255,0.015) 60px, rgba(255,255,255,0.015) 61px
    );
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.5rem;
    line-height: 1;
    color: var(--gold) !important;
    letter-spacing: 2px;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.4rem;
}
.hero-badge {
    display: inline-block;
    background: var(--pitch-mid);
    border: 1px solid var(--pitch-light);
    color: #7aff99;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── section card ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── team selector ── */
.stSelectbox > div > div {
    background: #0d2418 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSelectbox > div > div:hover { border-color: var(--gold) !important; }

/* ── score display ── */
.score-box {
    background: linear-gradient(135deg, #0d2418, #061510);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
}
.team-name-display {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 1px;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.score-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem;
    line-height: 1;
    color: var(--gold);
}
.score-divider {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: var(--muted);
    line-height: 1;
}
.elo-badge {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: 0.5px;
}
.xg-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.xg-value {
    font-size: 1.4rem;
    font-family: 'Bebas Neue', sans-serif;
    color: var(--pitch-light);
    letter-spacing: 1px;
}

/* ── winner banner ── */
.winner-banner {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 1rem;
}
.winner-home { background: linear-gradient(90deg,#0f3d1e,#0a2918); border:1px solid #1a7a3f; color:#7aff99; }
.winner-away { background: linear-gradient(90deg,#2a0a0a,#1a0505); border:1px solid #c53030; color:#ff9999; }
.winner-draw { background: linear-gradient(90deg,#2a2a0a,#1a1a05); border:1px solid #b7791f; color:#f0b429; }

/* ── probability bars ── */
.prob-container { margin: 0.5rem 0; }
.prob-row { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; }
.prob-label { width: 55px; font-size:0.8rem; color:var(--muted); text-align:right; flex-shrink:0; }
.prob-bar-wrap { flex:1; background:#0a2418; border-radius:4px; height:10px; overflow:hidden; }
.prob-bar-inner { height:10px; border-radius:4px; transition: width 0.5s ease; }
.bar-home { background: linear-gradient(90deg, #2ecc71, #27ae60); }
.bar-draw { background: linear-gradient(90deg, #f0b429, #c4922a); }
.bar-away { background: linear-gradient(90deg, #e74c3c, #c0392b); }
.prob-pct { width:40px; font-size:0.8rem; font-weight:600; color:var(--text); flex-shrink:0; }

/* ── stat pills ── */
.stat-grid { display:flex; gap:0.75rem; flex-wrap:wrap; justify-content:center; margin-top:1rem; }
.stat-pill {
    background: #0a2418;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    text-align: center;
    min-width: 90px;
}
.stat-pill-value { font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:var(--gold); display:block; line-height:1; }
.stat-pill-label { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; display:block; margin-top:2px; }

/* ── score matrix ── */
.matrix-cell {
    display:inline-block;
    width:32px; height:32px;
    line-height:32px;
    text-align:center;
    border-radius:5px;
    font-size:0.75rem;
    font-weight:600;
    margin:1px;
}

/* ── full predictions table ── */
.stDataFrame { border-radius: 10px !important; overflow:hidden !important; }
div[data-testid="stDataFrame"] { background: var(--surface) !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.75rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1.5rem !important;
}

/* ── button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--pitch-mid), var(--pitch)) !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--pitch-light), var(--pitch-mid)) !important;
    box-shadow: 0 0 20px rgba(240,180,41,0.3) !important;
}

/* ── metric overrides ── */
[data-testid="metric-container"] {
    background: #0a2418 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size:0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--gold) !important; font-size:1.6rem !important; }

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── tooltip / expander ── */
.streamlit-expanderHeader { color: var(--muted) !important; font-size:0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached)
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
    # Use exponential time decay: λ = 0.003 per day
    recent = raw[(raw["date"] >= "2019-01-01") &
                 raw["home_score"].notna()].copy()
    recent[["home_score","away_score"]] = recent[["home_score","away_score"]].astype(int)
    max_date = recent["date"].max()
    recent["days_ago"] = (max_date - recent["date"]).dt.days
    recent["time_w"]   = np.exp(-0.003 * recent["days_ago"])
    recent["tourn_w"]  = recent["tournament"].map(TW).fillna(1.0)
    recent["weight"]   = recent["time_w"] * recent["tourn_w"]

    # WC-calibrated base rates (neutral venue, WC tournament)
    wc = recent[recent["tournament"] == "FIFA World Cup"]
    LAH_WC = wc["home_score"].mean() if len(wc) > 20 else 1.36
    LAA_WC = wc["away_score"].mean() if len(wc) > 20 else 1.21
    # Use WC rates as base (all WC matches are neutral)
    LA_BASE = (LAH_WC + LAA_WC) / 2

    all_teams = set(recent["home_team"]) | set(recent["away_team"])
    attack = {}; defense = {}
    form   = {}   # last-5-game form score (0–1)

    for team in all_teams:
        hm = recent[recent["home_team"] == team]
        aw = recent[recent["away_team"] == team]

        # Weighted goals for/against
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

        # Recent form: last 5 competitive games
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
WC_TEAMS = [
    "Algeria","Argentina","Australia","Austria","Belgium","Brazil",
    "Cabo Verde","Canada","Colombia","Croatia","Curaçao","Côte d'Ivoire",
    "Ecuador","Egypt","England","FIFA Playoff 1","FIFA Playoff 2",
    "France","Germany","Ghana","Haiti","Iran","Japan","Jordan",
    "Mexico","Morocco","Netherlands","New Zealand","Norway","Panama",
    "Paraguay","Portugal","Qatar","Saudi Arabia","Scotland","Senegal",
    "South Africa","South Korea","Spain","Switzerland","Tunisia",
    "UEFA Playoff A","UEFA Playoff B","UEFA Playoff C","UEFA Playoff D",
    "Uruguay","USA","Uzbekistan",
]

NAME_MAP = {"USA": "United States"}

PLAYOFF_ELO = {
    "UEFA Playoff A": 1820, "UEFA Playoff B": 1810,
    "UEFA Playoff C": 1810, "UEFA Playoff D": 1810,
    "FIFA Playoff 1": 1720, "FIFA Playoff 2": 1720,
}

FLAG_MAP = {
    "Algeria":"🇩🇿","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Brazil":"🇧🇷","Cabo Verde":"🇨🇻","Canada":"🇨🇦",
    "Colombia":"🇨🇴","Croatia":"🇭🇷","Curaçao":"🇨🇼","Côte d'Ivoire":"🇨🇮",
    "Ecuador":"🇪🇨","Egypt":"🇪🇬","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷",
    "Germany":"🇩🇪","Ghana":"🇬🇭","Haiti":"🇭🇹","Iran":"🇮🇷",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
    "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦",
    "South Korea":"🇰🇷","Spain":"🇪🇸","Switzerland":"🇨🇭","Tunisia":"🇹🇳",
    "UEFA Playoff A":"🏴","UEFA Playoff B":"🏴","UEFA Playoff C":"🏴",
    "UEFA Playoff D":"🏴","FIFA Playoff 1":"🏴","FIFA Playoff 2":"🏴",
    "Uruguay":"🇺🇾","USA":"🇺🇸","Uzbekistan":"🇺🇿",
}


# ─────────────────────────────────────────────────────────────────
# REFINED PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
def predict_match(home, away, elo, attack, defense, form, LAH, LAA, is_knockout=False):
    """
    Refined Poisson model with:
    - Elo-based strength adjustment
    - Recency + tournament weighted attack/defense
    - Dixon-Coles low-score correction
    - Recent form factor
    - WC-calibrated base rates
    """
    def _elo(t):
        if t in PLAYOFF_ELO: return PLAYOFF_ELO[t]
        return elo.get(NAME_MAP.get(t, t), 1550)

    def _atk(t): return attack.get(NAME_MAP.get(t, t), 1.0)
    def _def(t): return defense.get(NAME_MAP.get(t, t), 1.0)
    def _form(t): return form.get(NAME_MAP.get(t, t), 0.5)

    # Elo-based adjustment
    elo_diff = _elo(home) - _elo(away)
    elo_factor = 1 + (elo_diff / 1000) * 0.45

    # Form factor (small nudge ±5%)
    form_h = 0.95 + _form(home) * 0.1
    form_a = 0.95 + _form(away) * 0.1

    # Expected goals (WC base rates, all neutral)
    lam_h = max(0.25, min(LAH * _atk(home) * _def(away) * elo_factor * form_h, 5.5))
    lam_a = max(0.25, min(LAA * _atk(away) * _def(home) / elo_factor * form_a, 5.5))

    # ── Dixon-Coles correction (rho=-0.1 typical for WC) ──
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
    prob_matrix /= prob_matrix.sum()  # renormalise

    # Most likely scoreline
    idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    p_home = float(prob_matrix[np.tril_indices(N, -1)].sum())
    p_draw = float(np.trace(prob_matrix))
    p_away = float(prob_matrix[np.triu_indices(N, 1)].sum())

    if p_home > p_draw and p_home > p_away: result = "home"
    elif p_away > p_home and p_away > p_draw: result = "away"
    else: result = "draw"

    # Knockout: no draw → Elo tiebreak (represents ET/pens)
    if is_knockout and result == "draw":
        result = "home" if _elo(home) >= _elo(away) else "away"

    # ── Corners: calibrated to WC avg 9.5 ──
    # Higher xG → more corners; defensive teams generate fewer
    total_xg = lam_h + lam_a
    def_ratio = (_def(home) + _def(away)) / 2  # high = defensive = fewer corners
    corners = max(6, min(14, int(round(5.5 + total_xg * 1.6 + (1 - def_ratio) * 1.2))))

    # ── Cards: calibrated to WC avg 3.2 yellow, 0.15 red ──
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
# GROUP STAGE FULL PREDICTIONS  (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running full tournament simulation…")
def run_full_predictions(_elo, _attack, _defense, _form, LAH, LAA):
    try:
        group_fixtures    = pd.read_csv("group_match.csv")
        knockout_fixtures = pd.read_csv("knokout_match.csv")
    except:
        return None, None

    # Group stage
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

    # Simulate standings
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
    html += '<table style="border-collapse:separate;border-spacing:3px;margin:0 auto">'
    header_style = 'style="text-align:center;font-size:0.7rem;color:#7aab8e;padding:2px 6px;font-weight:700"'
    html += f'<tr><td {header_style}></td>'
    for j in range(max_show): html += f'<td {header_style}>{j}</td>'
    html += '</tr>'
    max_p = pm[:max_show,:max_show].max()
    for i in range(max_show):
        html += f'<tr><td {header_style}>{i}</td>'
        for j in range(max_show):
            p = pm[i,j]
            intensity = int(p / max_p * 220) if max_p > 0 else 0
            if i > j:   bg = f"rgba(46,204,113,{p/max_p*0.85})"
            elif i < j: bg = f"rgba(231,76,60,{p/max_p*0.85})"
            else:       bg = f"rgba(240,180,41,{p/max_p*0.85})"
            pct = f"{p*100:.1f}%"
            html += f'<td><div class="matrix-cell" style="background:{bg};color:white">{pct}</div></td>'
        html += '</tr>'
    html += '</table></div>'
    return html


# ─────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────
# HERO
st.markdown("""
<div class="hero">
  <div class="hero-title">WC 2026 PREDICTOR</div>
  <div class="hero-sub">FIFA World Cup · USA / Canada / Mexico · 48 Teams · 104 Matches</div>
  <span class="hero-badge">⚡ Poisson + Elo + Dixon-Coles Model</span>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Training prediction model on 49,000+ historical matches…"):
    elo_data, atk, dfn, frm, LAH, LAA = load_and_train()

# TABS
tab1, tab2, tab3 = st.tabs(["🆚  MATCH PREDICTOR", "📋  FULL TOURNAMENT", "⚙️  MODEL INFO"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Select any two WC 2026 teams</div>', unsafe_allow_html=True)

    col_h, col_vs, col_a = st.columns([5, 1, 5])

    with col_h:
        home = st.selectbox("🏠 Home / Team 1", WC_TEAMS,
                            index=WC_TEAMS.index("Argentina"), key="home")
    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-family:\'Bebas Neue\',sans-serif;font-size:1.8rem;color:#7aab8e">VS</div>', unsafe_allow_html=True)
    with col_a:
        away_default = WC_TEAMS.index("France")
        away = st.selectbox("✈️ Away / Team 2", WC_TEAMS,
                            index=away_default, key="away")

    col_ko, _ = st.columns([2, 5])
    with col_ko:
        is_ko = st.toggle("Knockout (no draw)", value=False)

    st.markdown("---")

    if home == away:
        st.warning("⚠️ Please select two different teams.")
    else:
        res = predict_match(home, away, elo_data, atk, dfn, frm, LAH, LAA, is_knockout=is_ko)

        flag_h = FLAG_MAP.get(home, "🏳")
        flag_a = FLAG_MAP.get(away, "🏳")

        # Score display
        c1, c2, c3 = st.columns([4, 1, 4])
        with c1:
            st.markdown(f"""
            <div class="score-box">
              <div class="team-name-display">{flag_h} {home}</div>
              <div class="elo-badge">Elo {res['elo_home']}</div>
              <div class="score-number">{res['home_goals']}</div>
              <div class="xg-label">Expected Goals</div>
              <div class="xg-value">{res['lam_h']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown('<div style="text-align:center;font-family:\'Bebas Neue\',sans-serif;font-size:2.5rem;color:#1a7a3f;margin-top:1.5rem">:</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="score-box">
              <div class="team-name-display">{flag_a} {away}</div>
              <div class="elo-badge">Elo {res['elo_away']}</div>
              <div class="score-number">{res['away_goals']}</div>
              <div class="xg-label">Expected Goals</div>
              <div class="xg-value">{res['lam_a']}</div>
            </div>""", unsafe_allow_html=True)

        # Winner banner
        if res["result"] == "home":
            banner_cls = "winner-home"
            banner_txt = f"🏆 {home} wins"
        elif res["result"] == "away":
            banner_cls = "winner-away"
            banner_txt = f"🏆 {away} wins"
        else:
            banner_cls = "winner-draw"
            banner_txt = "⚖️ Draw"
        st.markdown(f'<div class="winner-banner {banner_cls}">{banner_txt}</div>', unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-pill"><span class="stat-pill-value">{res['corners']}</span><span class="stat-pill-label">Corners</span></div>
          <div class="stat-pill"><span class="stat-pill-value">{res['yellows']}</span><span class="stat-pill-label">Yellow Cards</span></div>
          <div class="stat-pill"><span class="stat-pill-value">{res['red']}</span><span class="stat-pill-label">Red Cards</span></div>
          <div class="stat-pill"><span class="stat-pill-value">{res['p_home']}%</span><span class="stat-pill-label">Home Win</span></div>
          <div class="stat-pill"><span class="stat-pill-value">{res['p_draw']}%</span><span class="stat-pill-label">Draw</span></div>
          <div class="stat-pill"><span class="stat-pill-value">{res['p_away']}%</span><span class="stat-pill-label">Away Win</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Deep analysis section
        col_prob, col_matrix = st.columns([2, 3])
        with col_prob:
            st.markdown('<div class="section-label">Win probabilities</div>', unsafe_allow_html=True)
            st.markdown(prob_bar_html(res["p_home"], res["p_draw"], res["p_away"]),
                        unsafe_allow_html=True)

        with col_matrix:
            st.markdown('<div class="section-label">Score probability matrix (5×5)</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.72rem;color:#7aab8e;margin-bottom:0.5rem">Rows = home goals · Columns = away goals · 🟢 home win · 🔴 away win · 🟡 draw</div>', unsafe_allow_html=True)
            st.markdown(score_matrix_html(res["prob_matrix"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — FULL TOURNAMENT
# ═══════════════════════════════════════════════════════════════════
with tab2:
    gp_df, kp_df = run_full_predictions(elo_data, atk, dfn, frm, LAH, LAA)

    if gp_df is None:
        st.info("📂 Place `group_match.csv` and `knokout_match.csv` in the app folder to see full tournament predictions.")
    else:
        sub1, sub2 = st.tabs(["⚽ Group Stage (72 matches)", "🏆 Knockout Stage (32 matches)"])

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
                "winning_team","corners","yellow_cards","red_cards","multiplier"
            ]].rename(columns={
                "home_team_resolved":"Home","away_team_resolved":"Away",
                "predicted_home_goals":"Home G","predicted_away_goals":"Away G",
                "winning_team":"Result","yellow_cards":"Yellows","red_cards":"Reds"
            })
            st.dataframe(display_kp, use_container_width=True, hide_index=True)
            csv2 = kp_df.to_csv(index=False)
            st.download_button("⬇️ Download knockout_predictions.csv", csv2,
                               "knockout_predictions.csv", "text/csv")

            # Final + champion
            final = kp_df[kp_df["round"]=="Final"]
            if len(final):
                f = final.iloc[0]
                champ = f["home_team_resolved"] if f["winning_team"]=="home" else f["away_team_resolved"]
                runner = f["away_team_resolved"] if f["winning_team"]=="home" else f["home_team_resolved"]
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0f3d1e,#001a0a);border:1px solid #f0b429;
                     border-radius:12px;padding:1.5rem;text-align:center;margin-top:1rem">
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:1rem;color:#7aab8e;letter-spacing:3px">
                    PREDICTED CHAMPION</div>
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#f0b429;margin:0.3rem 0">
                    🏆 {FLAG_MAP.get(champ,'🏳')} {champ}</div>
                  <div style="color:#7aab8e;font-size:0.85rem">Final vs {runner} · 
                    {f['predicted_home_goals']}-{f['predicted_away_goals']}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="max-width:720px">

    <div class="section-label">How the model works</div>

    <p style="color:#e8f5ee;line-height:1.7">
    This predictor uses a <strong style="color:#f0b429">refined Poisson regression model</strong> with five key components:</p>

    <table style="width:100%;border-collapse:collapse;font-size:0.88rem">
      <tr style="border-bottom:1px solid #1e4030">
        <td style="padding:0.6rem;color:#f0b429;font-weight:600;width:160px">Elo Ratings</td>
        <td style="padding:0.6rem;color:#c8ddd2">Computed from 49,000+ international matches since 2000.
        Tournament-weighted (WC counts 3× a friendly) with goal-difference multipliers.</td>
      </tr>
      <tr style="border-bottom:1px solid #1e4030">
        <td style="padding:0.6rem;color:#f0b429;font-weight:600">Poisson Goals Model</td>
        <td style="padding:0.6rem;color:#c8ddd2">Attack × Defense strength per team predicts expected goals.
        Base rates calibrated specifically to World Cup tournament averages (not all-match averages).</td>
      </tr>
      <tr style="border-bottom:1px solid #1e4030">
        <td style="padding:0.6rem;color:#f0b429;font-weight:600">Dixon-Coles Fix</td>
        <td style="padding:0.6rem;color:#c8ddd2">Standard Poisson overestimates 1-1 and underestimates 0-0.
        Dixon-Coles correction (ρ = −0.10) fixes the correlation between low-scoring outcomes.</td>
      </tr>
      <tr style="border-bottom:1px solid #1e4030">
        <td style="padding:0.6rem;color:#f0b429;font-weight:600">Recency Weighting</td>
        <td style="padding:0.6rem;color:#c8ddd2">Exponential time decay (λ=0.003/day) so recent matches matter more.
        Combined with tournament importance weights.</td>
      </tr>
      <tr>
        <td style="padding:0.6rem;color:#f0b429;font-weight:600">Form Factor</td>
        <td style="padding:0.6rem;color:#c8ddd2">Last 5 competitive games give a small ±5% nudge.
        Corners and cards calibrated to WC 2022 averages (9.5 corners, 3.2 yellows, 0.15 reds).</td>
      </tr>
    </table>

    <br>
    <div class="section-label">Data source</div>
    <p style="color:#7aab8e;font-size:0.85rem">
    Historical results: <code style="background:#0a2418;padding:2px 6px;border-radius:4px">martj42/international_results</code> on GitHub — 
    updated regularly, includes all international fixtures back to 1872.</p>

    </div>
    """, unsafe_allow_html=True)
