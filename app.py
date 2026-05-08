"""
app.py — F1 Hybrid AI Streamlit Dashboard
═══════════════════════════════════════════════════════════════════
Full-featured F1 prediction & analytics dashboard covering 1950-2026.

Pages:
  🏠 Home          — Hero, live model summary, quick-predict
  🔮 Predict       — Race predictor with weather controls
  📊 Analytics     — Historical driver/team/circuit stats
  🏆 Championships — Season-by-season champion table + era trends
  🌧  Weather Lab   — Wet vs dry impact simulator
  🗺  Circuit Map   — World map of all 67 F1 venues

Run:
  streamlit run app.py
═══════════════════════════════════════════════════════════════════
"""

import sys, os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="RaceIQ-F1",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Team colour palette ──────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull": "#3671C6",   "McLaren": "#FF8000",    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",   "Aston Martin": "#358C75","Alpine": "#0093CC",
    "Williams": "#64C4FF",   "Haas": "#B6BABD",       "Sauber": "#C92D4B",
    "Renault": "#FFD800",    "Lotus": "#FFD700",       "Tyrrell": "#0040A0",
    "Brabham": "#B0B0B0",    "BRM": "#007744",         "Maserati": "#CC0000",
    "Vanwall": "#005500",    "Cooper": "#4444CC",      "Jordan": "#FF8C00",
    "Benetton": "#00AA44",   "Alfa Romeo": "#9B0000",  "Force India": "#F596C8",
    "AlphaTauri": "#5E8FAA","RB / Racing Bulls": "#6692FF","Racing Point": "#F596C8",
    "Ligier": "#0055AA",     "Arrows": "#FF6600",      "Porsche": "#AA8800",
    "Honda": "#FFFFFF",      "March": "#CC3300",
}

ERA_COLORS = {
    "1950s": "#E63946", "1960s": "#F4A261", "1970s": "#E9C46A",
    "1980s": "#2A9D8F", "1990s": "#264653", "2000s": "#6A4C93",
    "2010s": "#1982C4", "2020s": "#FF595E",
}

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark racing theme */
.stApp { background-color: #0a0a0f; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#12121a 0%,#0d0d16 100%); border-right: 1px solid #2a2a3a; }
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a0a0a 0%, #0a0a1a 50%, #0a1a0a 100%);
    border: 1px solid #e8002d44;
    border-radius: 12px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(90deg, transparent, transparent 40px, #e8002d08 40px, #e8002d08 41px);
}
.hero-title { font-size: 2.8rem; font-weight: 900; letter-spacing: 4px;
    background: linear-gradient(90deg, #e8002d, #ff6b35, #ffd700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-transform: uppercase; margin: 0; }
.hero-sub { color: #888; font-size: 1rem; letter-spacing: 2px; margin-top: 0.5rem; }

/* Metric cards */
.metric-card {
    background: #12121e; border: 1px solid #2a2a3a; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card .value { font-size: 2.2rem; font-weight: 800; color: #e8002d; }
.metric-card .label { font-size: 0.75rem; letter-spacing: 1.5px; color: #666; text-transform: uppercase; }

/* Position badge */
.pos-badge {
    display: inline-block; width: 32px; height: 32px; border-radius: 50%;
    text-align: center; line-height: 32px; font-weight: 800; font-size: 0.85rem;
}
.p1 { background: #ffd700; color: #000; }
.p2 { background: #c0c0c0; color: #000; }
.p3 { background: #cd7f32; color: #fff; }

/* Result table rows */
.result-row { padding: 0.5rem 0; border-bottom: 1px solid #1e1e2e; }

/* Section headers */
.section-head { font-size: 0.75rem; letter-spacing: 3px; color: #e8002d;
    text-transform: uppercase; font-weight: 700; margin-bottom: 1rem;
    border-bottom: 1px solid #2a2a3a; padding-bottom: 0.5rem; }

/* Sidebar nav */
.nav-item { padding: 0.5rem 1rem; border-radius: 6px; margin: 2px 0;
    cursor: pointer; transition: background 0.2s; }
.nav-item:hover { background: #2a2a3a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("data/f1_1950_2026.csv")
    df["era"] = pd.cut(df["year"],
        bins=[1949,1959,1969,1979,1989,1999,2009,2019,2026],
        labels=["1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"])
    return df


@st.cache_resource(show_spinner=False)
def load_engine():
    from models.hybrid_engine import HybridF1Engine
    from models.lstm_model import SimpleLSTMFallback
    SimpleLSTMFallback.fit = lambda self, *a, **kw: self
    if os.path.exists("models/saved/pipeline.pkl"):
        return HybridF1Engine.load("models/saved")
    return None


@st.cache_data(show_spinner=False)
def get_circuit_list():
    from data.generate_dataset import CIRCUITS
    return sorted(CIRCUITS, key=lambda c: c[1])


@st.cache_data(show_spinner=False)
def get_driver_list():
    from data.generate_dataset import DRIVERS
    return DRIVERS


def team_color(name):
    return TEAM_COLORS.get(name, "#888888")

def make_weather(rainfall, track_temp, humidity, wind):
    return {
        "air_temp": track_temp * 0.72,
        "track_temp": track_temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind,
        "weather_condition": "Wet" if rainfall > 0 else "Dry",
        "is_wet": rainfall > 0,
    }

def build_race_df(year, circuit_id):
    """Build inference DataFrame for the selected year + circuit."""
    from data.generate_dataset import DRIVERS, CONSTRUCTORS, get_dominant
    active = [d for d in DRIVERS if d[3] <= year <= d[4]]
    if not active:
        return pd.DataFrame()
    dominant = get_dominant(year)
    rows = []
    for d in active[:20]:
        did, dname, nat, y0, y1, skill, main_team = d
        team_options = [c for c in CONSTRUCTORS if c[2] <= year <= c[3]]
        if any(t[0] == main_team and t[2] <= year <= t[3] for t in CONSTRUCTORS):
            team_id = main_team
        else:
            team_id = team_options[0][0] if team_options else "unknown"
        team_name = next((t[1] for t in CONSTRUCTORS if t[0] == team_id), "Unknown")
        team_tier = next((t[4] for t in CONSTRUCTORS if t[0] == team_id), 3)
        elapsed = max(0, year - y0)
        career_len = y1 - y0 + 1
        age_factor = 1.0 - 0.002 * max(0, elapsed - career_len * 0.6)
        cur_skill = min(skill * age_factor, 0.99)
        bonus = 0.12 if team_id == dominant else 0
        grid_score = cur_skill + bonus + np.random.normal(0, 0.06)
        rows.append({
            "driver_id": did, "driver_name": dname, "nationality": nat,
            "constructor_id": team_id, "constructor_name": team_name, "team_tier": team_tier,
            "grid": 0, "championship_position": np.random.randint(1, len(active)+1),
            "championship_points": max(0, cur_skill * 300 + np.random.normal(0, 30)),
            "constructor_points": max(0, (1 - team_tier/5) * 400),
            "constructor_position": team_tier,
            "driver_avg_pos_5": max(1, (1-cur_skill)*15 + np.random.normal(0,1.5)),
            "driver_avg_pts_5": max(0, cur_skill*20 + np.random.normal(0,2)),
            "driver_wins_5": max(0, round(cur_skill*3)),
            "driver_podiums_5": max(0, round(cur_skill*4)),
            "driver_dnf_rate": max(0, 0.15 - cur_skill*0.08 + np.random.uniform(0,0.05)),
            "grid_delta": np.random.uniform(-1, 2),
            "team_avg_pts_5": max(0, (1-team_tier/5)*25),
            "team_wins_5": max(0, round((1-team_tier/5)*2)),
            "driver_circuit_avg": max(1, (1-cur_skill)*12 + np.random.normal(0,2)),
            "driver_momentum": max(0, cur_skill*18),
            "position": int((1-cur_skill)*15 + np.random.randint(1,6)),
            "points": max(0, cur_skill*20),
            "is_wet": 0, "rain_factor": 0,
            "year": year, "round": 1, "circuit_id": circuit_id,
            "_grid_score": grid_score,
        })
    rdf = pd.DataFrame(rows).sort_values("_grid_score", ascending=False).reset_index(drop=True)
    rdf["grid"] = range(1, len(rdf)+1)
    return rdf.drop(columns=["_grid_score"])


# ══════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem;'>
        <div style='font-size:0.7rem; color:#555; letter-spacing:2px;'>1950 — 2026</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Race Predictor", "📊 Driver Analytics",
         "🏆 Championship History", "🌧 Weather Lab", "🗺 Circuit Map"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#444; text-align:center; letter-spacing:1px; line-height:1.8;'>
        <span style='color:#e8002d; font-weight:700; letter-spacing:2px;'>RaceIQ-F1</span><br>
        XGBoost × LSTM × Weather<br>
        14,264 race entries<br>
        77 seasons · 67 circuits · 74 drivers
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
with st.spinner("Loading F1 data..."):
    df = load_dataset()
    engine = load_engine()
    circuits = get_circuit_list()
    drivers_list = get_driver_list()


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.image("assets/banner.jpg", use_container_width=True)

    # ── Key stats ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        (len(df), "Race Entries"),
        (df["year"].nunique(), "Seasons"),
        (df["circuit_id"].nunique(), "Circuits"),
        (df["driver_name"].nunique(), "Drivers"),
        (df["constructor_name"].nunique(), "Constructors"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{val:,}</div>
                <div class='label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model status ───────────────────────────────────────────
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("<div class='section-head'>Model Architecture</div>", unsafe_allow_html=True)
        status = "✅ Trained & Ready" if engine else "⚠️ Model not found — run train.py"
        st.info(status)

        arch_data = {
            "Component": ["XGBoost", "LSTM / EWM", "Weather", "Hybrid MAE"],
            "Detail": ["500 trees · depth 6 · L1+L2", "BiLSTM(64) → LSTM(32) → Dense", "Grip · Rain · Wind · Temp", "~2.2 positions (full dataset)"],
            "Weight": ["50%", "30%", "20%", "—"],
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("<div class='section-head'>All-Time Wins Leaderboard</div>", unsafe_allow_html=True)
        wins = df[df["position"]==1]["driver_name"].value_counts().head(10).reset_index()
        wins.columns = ["Driver", "Wins"]
        fig = px.bar(wins, x="Wins", y="Driver", orientation="h",
                     color="Wins", color_continuous_scale=["#3a0010","#e8002d","#ff6b35"],
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"), height=320,
            font=dict(color="#cccccc", size=11),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Wins over time ─────────────────────────────────────────
    st.markdown("<div class='section-head'>Race Wins by Constructor — All Eras</div>", unsafe_allow_html=True)
    cw = (df[df["position"]==1]
          .groupby(["year","constructor_name"])
          .size().reset_index(name="wins"))
    top_teams = df[df["position"]==1]["constructor_name"].value_counts().head(10).index.tolist()
    cw_filtered = cw[cw["constructor_name"].isin(top_teams)]
    cw_cum = (cw_filtered.sort_values("year")
              .groupby(["constructor_name","year"])["wins"].sum()
              .groupby(level=0).cumsum().reset_index())
    fig2 = px.line(cw_cum, x="year", y="wins", color="constructor_name",
                   color_discrete_map=TEAM_COLORS, template="plotly_dark",
                   labels={"wins": "Cumulative Wins", "year": "Season", "constructor_name": "Team"})
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                       margin=dict(l=10,r=10,t=10,b=10), height=340,
                       font=dict(color="#cccccc", size=11),
                       legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10))
    fig2.update_traces(line_width=2)
    st.plotly_chart(fig2, use_container_width=True)

    # ── DNF rate trend ─────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-head'>DNF Rate by Decade</div>", unsafe_allow_html=True)
        dnf = df.groupby("era", observed=True)["is_dnf"].mean().reset_index()
        dnf.columns = ["Era", "DNF Rate"]
        dnf["pct"] = (dnf["DNF Rate"]*100).round(1)
        era_colors = [ERA_COLORS.get(str(e), "#888") for e in dnf["Era"]]
        fig3 = px.bar(dnf, x="Era", y="pct", template="plotly_dark",
                      color="Era", color_discrete_map=ERA_COLORS,
                      labels={"pct": "DNF %", "Era": ""})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                           margin=dict(l=10,r=10,t=10,b=10), height=260,
                           showlegend=False, font=dict(color="#cccccc", size=11))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-head'>Wet Race Frequency by Decade</div>", unsafe_allow_html=True)
        wet = df.groupby("era", observed=True)["is_wet"].mean().reset_index()
        wet.columns = ["Era", "Wet Rate"]
        wet["pct"] = (wet["Wet Rate"]*100).round(1)
        fig4 = px.bar(wet, x="Era", y="pct", template="plotly_dark",
                      color="Era", color_discrete_map=ERA_COLORS,
                      labels={"pct": "Wet Race %", "Era": ""})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                           margin=dict(l=10,r=10,t=10,b=10), height=260,
                           showlegend=False, font=dict(color="#cccccc", size=11))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: RACE PREDICTOR
# ══════════════════════════════════════════════════════════════════
elif page == "🔮 Race Predictor":
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("<div class='hero-title' style='font-size:1.6rem; margin-top:0.75rem;'>🔮 Race Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub' style='margin-bottom:1.5rem;'>Predict any Grand Prix from 1950 to 2026</div>", unsafe_allow_html=True)

    if not engine:
        st.error("Model not loaded. Please run `python train.py` first.")
        st.stop()

    # Controls
    col_c, col_y, col_w = st.columns([2, 1, 2])
    with col_c:
        circuit_options = [(f"{c[1]} ({c[2]})", c[0], c[5], c[6]) for c in circuits]
        circuit_labels  = [co[0] for co in circuit_options]
        default_idx = next((i for i,co in enumerate(circuit_options) if co[1]=="monaco"), 0)
        selected_label  = st.selectbox("🏟 Circuit", circuit_labels, index=default_idx)
        sel = circuit_options[circuit_labels.index(selected_label)]
        circuit_id, yr_min, yr_max = sel[1], sel[2], sel[3]

    with col_y:
        year = st.number_input("📅 Season", min_value=yr_min, max_value=yr_max,
                               value=min(2024, yr_max), step=1)

    with col_w:
        weather_preset = st.selectbox("🌤 Weather preset",
            ["☀️ Dry & Hot","🌤 Mild Dry","🌧 Light Rain","⛈ Heavy Wet","❄️ Cold Damp"])

    # Weather values from preset
    presets = {
        "☀️ Dry & Hot":   (0,   55, 40, 10),
        "🌤 Mild Dry":    (0,   55, 25, 15),
        "🌧 Light Rain":  (8,   80, 18, 20),
        "⛈ Heavy Wet":   (30,  90, 14, 35),
        "❄️ Cold Damp":   (5,   75, 12, 18),
    }
    rainfall, humidity, track_temp, wind = presets[weather_preset]

    with st.expander("⚙️ Fine-tune weather"):
        wc1, wc2, wc3, wc4 = st.columns(4)
        rainfall   = wc1.slider("Rainfall (mm)", 0, 60, rainfall)
        track_temp = wc2.slider("Track Temp (°C)", 5, 65, track_temp)
        humidity   = wc3.slider("Humidity (%)", 20, 100, humidity)
        wind       = wc4.slider("Wind (km/h)", 0, 60, wind)

    weather = make_weather(rainfall, track_temp, humidity, wind)

    if st.button("🚦 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running hybrid model..."):
            np.random.seed(42)
            race_df = build_race_df(year, circuit_id)
            if race_df.empty:
                st.warning("No drivers found for this era.")
            else:
                results = engine.predict_race(race_df, weather=weather)
                team_df = engine.predict_constructor_championship(results)

                st.markdown("---")

                # ── Podium ────────────────────────────────────
                st.markdown("<div class='section-head'>🏆 Predicted Podium</div>", unsafe_allow_html=True)
                pod_cols = st.columns(3)
                podium_order = [1, 0, 2]  # display: P2, P1, P3
                col_heights  = ["🥈", "🥇", "🥉"]
                for display_slot, (result_idx, medal) in enumerate(zip(podium_order, col_heights)):
                    if result_idx < len(results):
                        r = results[result_idx]
                        tc = team_color(r.constructor)
                        with pod_cols[display_slot]:
                            st.markdown(f"""
                            <div style='background:#12121e; border:1px solid {tc}55;
                                border-radius:10px; padding:1rem; text-align:center;'>
                                <div style='font-size:2rem;'>{medal}</div>
                                <div style='font-size:1rem; font-weight:800; color:#fff; margin:0.3rem 0;'>{r.driver}</div>
                                <div style='font-size:0.8rem; color:{tc}; font-weight:600;'>{r.constructor}</div>
                                <div style='font-size:1.3rem; font-weight:800; color:#e8002d; margin-top:0.5rem;'>{r.win_probability*100:.1f}%</div>
                                <div style='font-size:0.65rem; color:#555;'>win probability</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Full grid + probability chart ─────────────
                col_grid, col_prob = st.columns([1.2, 1])

                with col_grid:
                    st.markdown("<div class='section-head'>Full Grid Prediction</div>", unsafe_allow_html=True)
                    medals = {1:"🥇",2:"🥈",3:"🥉"}
                    for r in results:
                        tc = team_color(r.constructor)
                        m  = medals.get(r.predicted_position, f"P{r.predicted_position:2d}")
                        st.markdown(f"""
                        <div style='display:flex; align-items:center; padding:0.4rem 0.5rem;
                            border-bottom:1px solid #1e1e2e; border-radius:4px;'>
                            <span style='width:36px; font-weight:800; font-size:0.9rem;'>{m}</span>
                            <span style='width:8px; height:8px; border-radius:50%;
                                background:{tc}; display:inline-block; margin-right:8px; flex-shrink:0;'></span>
                            <span style='flex:1; font-size:0.9rem; color:#ddd;'>{r.driver}</span>
                            <span style='font-size:0.75rem; color:#888; margin-right:12px;'>{r.constructor}</span>
                            <span style='font-size:0.85rem; font-weight:700; color:#e8002d;'>{r.win_probability*100:.1f}%</span>
                        </div>""", unsafe_allow_html=True)

                with col_prob:
                    st.markdown("<div class='section-head'>Win Probability Distribution</div>", unsafe_allow_html=True)
                    prob_df = pd.DataFrame({
                        "Driver": [r.driver for r in results],
                        "Win %": [r.win_probability*100 for r in results],
                        "Constructor": [r.constructor for r in results],
                    }).head(12)
                    colors = [team_color(c) for c in prob_df["Constructor"]]
                    fig = go.Figure(go.Bar(
                        x=prob_df["Win %"], y=prob_df["Driver"],
                        orientation="h", marker_color=colors,
                        text=[f"{v:.1f}%" for v in prob_df["Win %"]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#0d0d16", margin=dict(l=10,r=60,t=10,b=10),
                        height=420, xaxis_title="Win %",
                        yaxis=dict(autorange="reversed"),
                        font=dict(color="#ccc", size=11),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Constructor championship ───────────────────
                st.markdown("<div class='section-head'>Constructor Championship Forecast</div>", unsafe_allow_html=True)
                tc_colors = [team_color(t) for t in team_df["constructor"]]
                fig_team = go.Figure(go.Bar(
                    x=team_df["constructor"], y=team_df["total_points"],
                    marker_color=tc_colors,
                    text=team_df["total_points"].astype(int),
                    textposition="outside",
                ))
                fig_team.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d0d16", margin=dict(l=10,r=10,t=10,b=10),
                    height=280, yaxis_title="Projected Points",
                    font=dict(color="#ccc", size=11),
                )
                st.plotly_chart(fig_team, use_container_width=True)

                # ── Weather summary ────────────────────────────
                from data.features import compute_weather_impact_score
                w_score = compute_weather_impact_score(weather)
                wc = st.columns(5)
                wc[0].metric("Condition", "🌧 Wet" if rainfall>0 else "☀️ Dry")
                wc[1].metric("Rainfall", f"{rainfall} mm")
                wc[2].metric("Track Temp", f"{track_temp}°C")
                wc[3].metric("Humidity", f"{humidity}%")
                wc[4].metric("Weather Score", f"{w_score:.3f}")


# ══════════════════════════════════════════════════════════════════
# PAGE: DRIVER ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Driver Analytics":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>📊 Driver Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub' style='margin-bottom:1.5rem;'>Performance deep-dive across all eras</div>", unsafe_allow_html=True)

    driver_names = sorted(df["driver_name"].unique())
    default_drivers = ["Lewis Hamilton","Michael Schumacher","Ayrton Senna","Max Verstappen","Alain Prost"]
    default_sel = [d for d in default_drivers if d in driver_names]

    selected_drivers = st.multiselect("Select drivers to compare", driver_names, default=default_sel[:4])

    if not selected_drivers:
        st.info("Select at least one driver to see analytics.")
        st.stop()

    drv_df = df[df["driver_name"].isin(selected_drivers)]

    # ── Career wins per season ─────────────────────────────────
    st.markdown("<div class='section-head'>Wins per Season</div>", unsafe_allow_html=True)
    wins_yr = (drv_df[drv_df["position"]==1]
               .groupby(["driver_name","year"]).size().reset_index(name="wins"))
    fig = px.line(wins_yr, x="year", y="wins", color="driver_name",
                  markers=True, template="plotly_dark",
                  labels={"wins":"Race Wins","year":"Season","driver_name":"Driver"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                      height=300, margin=dict(l=10,r=10,t=10,b=10),
                      font=dict(color="#ccc", size=11),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # ── Points per season ──────────────────────────────────
        st.markdown("<div class='section-head'>Points per Season</div>", unsafe_allow_html=True)
        pts_yr = drv_df.groupby(["driver_name","year"])["points"].sum().reset_index()
        fig2 = px.area(pts_yr, x="year", y="points", color="driver_name",
                       template="plotly_dark",
                       labels={"points":"Season Points","year":"Season","driver_name":"Driver"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                           height=280, margin=dict(l=10,r=10,t=10,b=10),
                           font=dict(color="#ccc", size=11),
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # ── DNF rate by driver ─────────────────────────────────
        st.markdown("<div class='section-head'>DNF Rate</div>", unsafe_allow_html=True)
        dnf_r = drv_df.groupby("driver_name")["is_dnf"].mean().reset_index()
        dnf_r["pct"] = (dnf_r["is_dnf"]*100).round(1)
        dnf_r = dnf_r.sort_values("pct", ascending=True)
        fig3 = px.bar(dnf_r, x="pct", y="driver_name", orientation="h",
                      template="plotly_dark", color="pct",
                      color_continuous_scale=["#00ff88","#ffcc00","#e8002d"],
                      labels={"pct":"DNF %","driver_name":"Driver"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                           height=280, margin=dict(l=10,r=10,t=10,b=10),
                           coloraxis_showscale=False, font=dict(color="#ccc", size=11))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Summary stats table ────────────────────────────────────
    st.markdown("<div class='section-head'>Career Summary</div>", unsafe_allow_html=True)
    summary = drv_df.groupby("driver_name").agg(
        Races=("position","count"),
        Wins=("position", lambda x: (x==1).sum()),
        Podiums=("position", lambda x: (x<=3).sum()),
        Points=("points","sum"),
        DNFs=("is_dnf","sum"),
        Best_Position=("position","min"),
        Avg_Position=("position","mean"),
        Wet_Starts=("is_wet","sum"),
    ).reset_index()
    summary["Win%"]    = (summary["Wins"] / summary["Races"] * 100).round(1)
    summary["Podium%"] = (summary["Podiums"] / summary["Races"] * 100).round(1)
    summary["Avg_Position"] = summary["Avg_Position"].round(2)
    summary["Points"] = summary["Points"].astype(int)
    summary = summary.rename(columns={"driver_name": "Driver"})
    st.dataframe(summary.set_index("Driver"), use_container_width=True)

    # ── Grid vs Finish scatter ─────────────────────────────────
    st.markdown("<div class='section-head'>Grid Position vs Finishing Position</div>", unsafe_allow_html=True)
    scatter = drv_df[["driver_name","grid","position","year","circuit_name"]].dropna()
    fig4 = px.scatter(scatter, x="grid", y="position", color="driver_name",
                      template="plotly_dark", opacity=0.6,
                      hover_data=["year","circuit_name"],
                      labels={"grid":"Grid","position":"Finish","driver_name":"Driver"})
    fig4.add_shape(type="line", x0=1, y0=1, x1=20, y1=20,
                   line=dict(color="#555", dash="dash", width=1))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                       height=360, margin=dict(l=10,r=10,t=10,b=10),
                       font=dict(color="#ccc", size=11),
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: CHAMPIONSHIP HISTORY
# ══════════════════════════════════════════════════════════════════
elif page == "🏆 Championship History":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🏆 Championship History</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub' style='margin-bottom:1.5rem;'>Season champions and era dominance — 1950 to 2026</div>", unsafe_allow_html=True)

    # Build champions table
    champs = []
    for year, grp in df.groupby("year"):
        wins_yr = grp[grp["position"]==1].groupby(["driver_name","constructor_name"]).size()
        if len(wins_yr):
            winner = wins_yr.idxmax()
            pts    = grp[grp["driver_name"]==winner[0]]["points"].sum()
            champs.append({"Season": year, "Champion": winner[0],
                            "Constructor": winner[1], "Wins": int(wins_yr.max()),
                            "Points": int(pts)})
    champ_df = pd.DataFrame(champs)

    # ── All champions table ────────────────────────────────────
    col_t, col_c = st.columns([1.4, 1])

    with col_t:
        st.markdown("<div class='section-head'>Season Champions</div>", unsafe_allow_html=True)
        year_range = st.slider("Filter seasons", 1950, 2026, (1980, 2026))
        filtered = champ_df[(champ_df["Season"]>=year_range[0]) &
                            (champ_df["Season"]<=year_range[1])]
        st.dataframe(filtered.set_index("Season"), use_container_width=True, height=400)

    with col_c:
        st.markdown("<div class='section-head'>Most Championship Titles</div>", unsafe_allow_html=True)
        multi = champ_df["Champion"].value_counts().reset_index()
        multi.columns = ["Driver", "Titles"]
        multi = multi[multi["Titles"] > 1].head(12)
        fig = px.bar(multi, x="Titles", y="Driver", orientation="h",
                     template="plotly_dark",
                     color="Titles", color_continuous_scale=["#3a0010","#e8002d","#ffd700"],
                     labels={"Titles":"Championships","Driver":""})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                          height=360, margin=dict(l=10,r=10,t=10,b=10),
                          coloraxis_showscale=False, font=dict(color="#ccc",size=11),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Constructor titles
        st.markdown("<div class='section-head'>Constructor Titles</div>", unsafe_allow_html=True)
        ct = champ_df["Constructor"].value_counts().reset_index()
        ct.columns = ["Constructor","Titles"]
        ct = ct[ct["Titles"]>0].head(10)
        tc_colors = [team_color(t) for t in ct["Constructor"]]
        fig2 = go.Figure(go.Bar(
            x=ct["Constructor"], y=ct["Titles"],
            marker_color=tc_colors, text=ct["Titles"], textposition="outside",
        ))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="#0d0d16", height=240,
                           margin=dict(l=10,r=10,t=10,b=40),
                           font=dict(color="#ccc",size=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Champion wins per season timeline ─────────────────────
    st.markdown("<div class='section-head'>Champion's Win Count per Season</div>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=champ_df["Season"], y=champ_df["Wins"],
        mode="markers+lines", marker_size=8,
        marker_color=champ_df["Wins"], marker_colorscale="Reds",
        line_color="#444",
        text=champ_df["Champion"] + " (" + champ_df["Constructor"] + ")",
        hovertemplate="%{x}: %{text}<br>Wins: %{y}<extra></extra>",
    ))
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="#0d0d16", height=300,
                       margin=dict(l=10,r=10,t=10,b=10),
                       yaxis_title="Race Wins", font=dict(color="#ccc",size=11))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: WEATHER LAB
# ══════════════════════════════════════════════════════════════════
elif page == "🌧 Weather Lab":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🌧 Weather Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub' style='margin-bottom:1.5rem;'>Simulate weather impact on race outcomes</div>", unsafe_allow_html=True)

    if not engine:
        st.error("Model not loaded. Please run `python train.py` first.")
        st.stop()

    col_set, col_out = st.columns([1, 2])

    with col_set:
        st.markdown("<div class='section-head'>Scenario Setup</div>", unsafe_allow_html=True)

        circuit_options = [(f"{c[1]}", c[0], c[5], c[6]) for c in circuits]
        circuit_labels  = [co[0] for co in circuit_options]
        default_idx = next((i for i,co in enumerate(circuit_options) if co[1]=="spa"), 0)
        sel_label = st.selectbox("Circuit", circuit_labels, index=default_idx)
        sel = circuit_options[circuit_labels.index(sel_label)]
        cid, yr_min, yr_max = sel[1], sel[2], sel[3]
        yr = st.number_input("Season", min_value=yr_min, max_value=yr_max,
                              value=min(2024, yr_max), step=1)

        st.markdown("**Scenario A — Dry**")
        a_temp = st.slider("Track Temp A (°C)", 15, 65, 42)
        a_hum  = st.slider("Humidity A (%)", 20, 100, 50)
        a_wind = st.slider("Wind A (km/h)", 0, 60, 10)

        st.markdown("**Scenario B — Wet**")
        b_rain = st.slider("Rainfall B (mm)", 0, 60, 25)
        b_temp = st.slider("Track Temp B (°C)", 5, 40, 16)
        b_hum  = st.slider("Humidity B (%)", 40, 100, 88)
        b_wind = st.slider("Wind B (km/h)", 0, 60, 30)

        run_btn = st.button("🚦 Compare Scenarios", type="primary", use_container_width=True)

    with col_out:
        st.markdown("<div class='section-head'>Results</div>", unsafe_allow_html=True)

        if run_btn:
            np.random.seed(99)
            race_df = build_race_df(yr, cid)
            if race_df.empty:
                st.warning("No drivers found.")
            else:
                wa = make_weather(0, a_temp, a_hum, a_wind)
                wb = make_weather(b_rain, b_temp, b_hum, b_wind)
                ra = engine.predict_race(race_df, weather=wa)
                rb = engine.predict_race(race_df, weather=wb)

                from data.features import compute_weather_impact_score
                score_a = compute_weather_impact_score(wa)
                score_b = compute_weather_impact_score(wb)

                mc1, mc2 = st.columns(2)
                mc1.metric("Weather Score — Dry", f"{score_a:.3f}", "Optimal")
                mc2.metric("Weather Score — Wet", f"{score_b:.3f}", f"{score_b-score_a:+.3f}")

                # Side-by-side bar chart
                comp = pd.DataFrame({
                    "Driver": [r.driver for r in ra[:12]],
                    "Dry Win%": [r.win_probability*100 for r in ra[:12]],
                    "Team_A": [r.constructor for r in ra[:12]],
                })
                wet_map = {r.driver: r.win_probability*100 for r in rb}
                comp["Wet Win%"] = comp["Driver"].map(wet_map).fillna(0)
                comp["Δ"] = (comp["Wet Win%"] - comp["Dry Win%"]).round(1)
                comp["color"] = comp["Δ"].apply(lambda x: "#00cc66" if x > 0 else "#e8002d")

                fig = go.Figure()
                fig.add_trace(go.Bar(name="☀️ Dry",   x=comp["Driver"], y=comp["Dry Win%"],
                                     marker_color="#f4a261", text=comp["Dry Win%"].round(1),
                                     textposition="outside"))
                fig.add_trace(go.Bar(name="🌧 Wet",   x=comp["Driver"], y=comp["Wet Win%"],
                                     marker_color="#4895ef", text=comp["Wet Win%"].round(1),
                                     textposition="outside"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#0d0d16", barmode="group",
                                  height=360, margin=dict(l=10,r=10,t=10,b=60),
                                  yaxis_title="Win %", font=dict(color="#ccc",size=10),
                                  legend=dict(bgcolor="rgba(0,0,0,0)"))
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True)

                # Delta table
                st.markdown("<div class='section-head'>Wet vs Dry Impact (Δ Win%)</div>", unsafe_allow_html=True)
                comp_show = comp[["Driver","Team_A","Dry Win%","Wet Win%","Δ"]].copy()
                comp_show = comp_show.sort_values("Δ", ascending=False)
                comp_show.columns = ["Driver","Team","☀️ Dry %","🌧 Wet %","Δ"]
                st.dataframe(comp_show.set_index("Driver"), use_container_width=True)

                # Weather breakdown gauges
                st.markdown("<div class='section-head'>Weather Feature Breakdown</div>", unsafe_allow_html=True)
                from data.features import compute_weather_impact_score
                labels  = ["Grip Index","Wind Effect","Temp Effect","Overall Score"]
                rain_f  = 1 if b_rain > 0 else 0
                grip_a  = (1 - a_hum/100)
                grip_b  = (1 - b_hum/100) * (1 - rain_f)
                wind_a  = 1 - min(a_wind/60, 1)
                wind_b  = 1 - min(b_wind/60, 1)
                temp_a  = min(a_temp/50, 0.9)
                temp_b  = min(b_temp/50, 0.9)
                vals_a  = [grip_a, wind_a, temp_a, score_a]
                vals_b  = [grip_b, wind_b, temp_b, score_b]

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="☀️ Dry", x=labels, y=vals_a, marker_color="#f4a261"))
                fig2.add_trace(go.Bar(name="🌧 Wet", x=labels, y=vals_b, marker_color="#4895ef"))
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="#0d0d16", barmode="group",
                                   height=240, margin=dict(l=10,r=10,t=10,b=10),
                                   yaxis_title="Score [0-1]",
                                   font=dict(color="#ccc",size=11),
                                   legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure your scenarios and click **Compare Scenarios**.")

    # ── Historical wet performance ─────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-head'>Wettest Circuits Historically</div>", unsafe_allow_html=True)
    wet_circ = (df.groupby("circuit_name")["is_wet"].mean() * 100).round(1).sort_values(ascending=False).head(15).reset_index()
    wet_circ.columns = ["Circuit","Wet Race %"]
    fig3 = px.bar(wet_circ, x="Circuit", y="Wet Race %", template="plotly_dark",
                  color="Wet Race %", color_continuous_scale=["#264653","#4895ef","#00bbff"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d16",
                       height=280, margin=dict(l=10,r=10,t=10,b=80),
                       coloraxis_showscale=False, font=dict(color="#ccc",size=10))
    fig3.update_xaxes(tickangle=35)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: CIRCUIT MAP
# ══════════════════════════════════════════════════════════════════
elif page == "🗺 Circuit Map":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🗺 Circuit Map</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub' style='margin-bottom:1.5rem;'>All 67 Formula 1 venues — 1950 to 2026</div>", unsafe_allow_html=True)

    from data.generate_dataset import CIRCUITS

    circuit_df = pd.DataFrame([
        {"circuit_id": c[0], "name": c[1], "country": c[2],
         "lat": c[3], "lon": c[4], "first": c[5], "last": c[6], "type": c[7]}
        for c in CIRCUITS
    ])

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        type_filter = st.multiselect("Circuit type", ["permanent","street","oval"],
                                     default=["permanent","street","oval"])
    with fc2:
        era_filter = st.slider("Active during year range", 1950, 2026, (1950, 2026))

    filtered_circuits = circuit_df[
        (circuit_df["type"].isin(type_filter)) &
        (circuit_df["first"] <= era_filter[1]) &
        (circuit_df["last"]  >= era_filter[0])
    ].copy()

    # Add win count
    wins_per_circuit = df[df["position"]==1]["circuit_name"].value_counts().to_dict()
    filtered_circuits["wins"] = filtered_circuits["name"].map(wins_per_circuit).fillna(0).astype(int)
    filtered_circuits["seasons"] = filtered_circuits["last"] - filtered_circuits["first"] + 1
    filtered_circuits["status"] = filtered_circuits["last"].apply(
        lambda y: "🟢 Active" if y >= 2024 else "⚫ Historic"
    )

    type_color = {"permanent": "#3671c6", "street": "#e8002d", "oval": "#ffd700"}
    filtered_circuits["color"] = filtered_circuits["type"].map(type_color)

    fig = px.scatter_geo(
        filtered_circuits,
        lat="lat", lon="lon",
        hover_name="name",
        hover_data={"country":True,"first":True,"last":True,"type":True,"wins":True,"lat":False,"lon":False},
        color="type",
        color_discrete_map=type_color,
        size="seasons",
        size_max=20,
        template="plotly_dark",
        projection="natural earth",
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        geo=dict(
            bgcolor="#0a0a0f",
            landcolor="#12121e",
            oceancolor="#0a0a14",
            lakecolor="#0a0a14",
            framecolor="#2a2a3a",
            showland=True, showocean=True, showlakes=True,
            showcountries=True, countrycolor="#2a2a3a",
        ),
        height=520, margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc")),
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Legend ─────────────────────────────────────────────────
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("🔵 **Permanent** — traditional race tracks")
    lc2.markdown("🔴 **Street** — temporary city circuits")
    lc3.markdown("🟡 **Oval** — Indianapolis-style tracks")

    # ── Circuit table ──────────────────────────────────────────
    st.markdown("<div class='section-head' style='margin-top:1rem;'>Circuit Details</div>", unsafe_allow_html=True)
    sort_by = st.selectbox("Sort by", ["name","first","last","seasons","wins","type"])
    tbl = filtered_circuits[["name","country","type","first","last","seasons","wins","status"]].copy()
    tbl = tbl.rename(columns={"name":"Circuit","country":"Country","type":"Type",
                               "first":"First","last":"Last","seasons":"Seasons","wins":"Wins","status":"Status"})
    tbl = tbl.sort_values(sort_by if sort_by in tbl.columns else "Circuit")
    st.dataframe(tbl.set_index("Circuit"), use_container_width=True, height=400)

    # ── Quick stats ────────────────────────────────────────────
    st.markdown("---")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Circuits shown", len(filtered_circuits))
    sc2.metric("Active (2024+)", len(filtered_circuits[filtered_circuits["last"]>=2024]))
    sc3.metric("Historic", len(filtered_circuits[filtered_circuits["last"]<2024]))
    sc4.metric("Street circuits", len(filtered_circuits[filtered_circuits["type"]=="street"]))
