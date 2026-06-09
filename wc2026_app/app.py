# =========================
# ONLY UI IMPROVEMENTS APPLIED
# =========================

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  – WC 2026 FLEXBOX + NATION VIBES UI UPGRADE
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

/* ─────────────────────────────
   WC 2026 GLOBAL COLOR SYSTEM
───────────────────────────── */
:root {
  --pitch: #0a3d1f;
  --pitch-mid: #0f5c2e;
  --pitch-light: #1a7a3f;

  --gold: #f0b429;
  --gold-dim: #c4922a;

  /* NEW: WC 2026 "nation vibe" accents */
  --host-blue: #1e3a8a;
  --host-red: #b91c1c;
  --host-green: #166534;
  --host-gold: #d4af37;

  --bg: #06110d;
  --surface: #0d2418;
  --surface-2: #0a1a12;
  --border: #1e4030;
  --text: #e8f5ee;
  --muted: #7aab8e;
}

/* ─────────────────────────────
   GLOBAL
───────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem !important; max-width: 1200px !important; }

/* ─────────────────────────────
   FLEX ROOT LAYOUT SYSTEM
───────────────────────────── */
.flex {
    display: flex;
    gap: 1rem;
}

.flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.flex-between {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.flex-col {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.flex-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

/* ─────────────────────────────
   HERO (STADIUM BROADCAST STYLE)
───────────────────────────── */
.hero {
    background: linear-gradient(135deg, var(--pitch) 0%, #001a0a 50%, var(--host-blue) 120%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

/* subtle stadium grid */
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(90deg, transparent, transparent 60px, rgba(255,255,255,0.03) 60px),
        repeating-linear-gradient(0deg, transparent, transparent 60px, rgba(255,255,255,0.02) 60px);
    opacity: 0.25;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    color: var(--gold);
    letter-spacing: 2px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--surface);
}

/* ─────────────────────────────
   SCOREBOARD FLEX CARD
───────────────────────────── */
.score-wrapper {
    display: flex;
    gap: 1rem;
    justify-content: center;
    align-items: stretch;
    flex-wrap: wrap;
}

.score-box {
    flex: 1;
    min-width: 240px;
    background: linear-gradient(145deg, var(--surface), var(--surface-2));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    transition: transform 0.2s ease;
}

.score-box:hover {
    transform: translateY(-4px);
    border-color: var(--gold);
}

.team-name-display {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
}

.score-number {
    font-size: 5rem;
    font-family: 'Bebas Neue', sans-serif;
    color: var(--gold);
}

/* ─────────────────────────────
   STATS FLEX GRID
───────────────────────────── */
.stat-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.8rem;
}

.stat-pill {
    flex: 1;
    min-width: 120px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.7rem;
    text-align: center;
}

/* WC accent strip per stat */
.stat-pill:nth-child(1) { border-top: 2px solid var(--host-green); }
.stat-pill:nth-child(2) { border-top: 2px solid var(--host-red); }
.stat-pill:nth-child(3) { border-top: 2px solid var(--gold); }
.stat-pill:nth-child(4) { border-top: 2px solid var(--host-blue); }

/* ─────────────────────────────
   PROBABILITY FLEX CLEANUP
───────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.prob-bar-wrap {
    flex: 1;
}

/* ─────────────────────────────
   SECTION CARDS
───────────────────────────── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
}

/* ─────────────────────────────
   BUTTON (WC STYLE)
───────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--pitch-mid), var(--host-blue)) !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    box-shadow: 0 0 18px rgba(240,180,41,0.25) !important;
    transform: translateY(-1px);
}

</style>
""", unsafe_allow_html=True)
