"""
TAURASI — Trajectory Analysis Using Regression And Similarity Index
WNBA Player Projection System · Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="TAURASI · WNBA Projections",
    page_icon="edittau.jpg",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .main { background-color: #0d0d10; }
  .block-container { padding: 2rem 2.5rem; max-width: 1100px; }

  /* Header */
  .taurasi-header {
    border-bottom: 1px solid #222;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
  }
  .taurasi-title {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #e8e0d0;
    margin: 0;
  }
  .taurasi-sub {
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Player card */
  .player-card {
    background: #111115;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .player-name {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8e0d0;
    margin: 0 0 4px 0;
  }
  .player-meta {
    font-size: 0.82rem;
    color: #555;
    letter-spacing: 0.06em;
  }

  /* Tier badge */
  .tier-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    margin-top: 8px;
  }
  .tier-alltime   { background: #2a1f00; color: #f5c842; border: 1px solid #4a3800; }
  .tier-franchise { background: #0f2a1a; color: #4ade80; border: 1px solid #1a4a2a; }
  .tier-star      { background: #0d1f2a; color: #60c8f5; border: 1px solid #1a3a4a; }
  .tier-starter   { background: #1a1a2a; color: #a0a0e0; border: 1px solid #2a2a4a; }
  .tier-role      { background: #1a1a1a; color: #888; border: 1px solid #333; }
  .tier-replacement { background: #1a0d0d; color: #e06060; border: 1px solid #3a1a1a; }

  /* Metric cards */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1.2rem 0;
  }
  .metric-box {
    background: #16161a;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 14px 16px;
  }
  .metric-label {
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #e8e0d0;
  }

  /* Comparable rows */
  .comp-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #1a1a1a;
  }
  .comp-name { font-size: 0.9rem; color: #c8c0b0; font-weight: 500; }
  .comp-meta { font-size: 0.75rem; color: #444; margin-top: 2px; }
  .comp-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #60c8f5;
  }

  /* Section labels */
  .section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 12px;
  }

  /* Overseas flag */
  .overseas-flag {
    font-size: 0.72rem;
    color: #888;
    font-style: italic;
    margin-top: 6px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0a0a0d;
    border-right: 1px solid #1a1a1a;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {
    color: #666 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Footer */
  .taurasi-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1a1a1a;
    font-size: 0.72rem;
    color: #333;
    letter-spacing: 0.06em;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("wnba_data.csv")
    return df


def compute_warp_w(df, replacement_bpm=-2.0, league_avg_mp=28.5, games=40):
    df = df.copy()
    df["WARP_W"] = (
        (df["BPM"] - replacement_bpm)
        * (df["MP"] / league_avg_mp)
        * (1 / 22.0)
        * games
    ).round(1)
    return df


def clean(df):
    pos_map = {"G": 1, "G-F": 2, "F-G": 2, "F": 3, "F-C": 4, "C-F": 4, "C": 5}
    df = df.copy()
    df["PosNum"] = df["Pos"].map(pos_map).fillna(3)
    df = df.dropna(subset=["BPM", "Age"])
    return df


FEATURES = ["Age", "BPM", "WS40", "PosNum", "PTS", "TRB", "AST"]

@st.cache_resource
def build_engine(df):
    X = df[FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=min(15, len(df)), metric="euclidean")
    knn.fit(X_scaled)
    return scaler, knn


def find_comparables(player_row, df, scaler, knn, top_n=5):
    X_p = pd.DataFrame([player_row[FEATURES].fillna(0)])
    X_scaled = scaler.transform(X_p)
    distances, indices = knn.kneighbors(X_scaled)
    comps = df.iloc[indices[0][1:]].copy()
    raw = distances[0][1:top_n+1]
    max_d = raw.max() if raw.max() > 0 else 1
    comps = comps.head(top_n).copy()
    comps["Sim"] = (100 * (1 - raw / (max_d * 1.8))).clip(0, 100).round(0).astype(int)
    return comps[["Player", "Season", "Age", "Pos", "BPM", "WARP_W", "Sim"]]


def project_player(player_row, df, scaler, knn, seasons=5, top_n=10):
    X_p = pd.DataFrame([player_row[FEATURES].fillna(0)])
    X_scaled = scaler.transform(X_p)
    distances, indices = knn.kneighbors(X_scaled, n_neighbors=min(top_n+1, len(df)))
    comp_indices = indices[0][1:]
    comp_distances = distances[0][1:]
    weights = 1 / (comp_distances + 1e-5)
    weights /= weights.sum()

    projs, lows, highs = [], [], []
    current_age = player_row["Age"]

    for yr in range(1, seasons + 1):
        target_age = current_age + yr
        vals, ws = [], []
        for i, idx in enumerate(comp_indices):
            comp_name = df.iloc[idx]["Player"]
            future = df[(df["Player"] == comp_name) & (df["Age"] == target_age)]
            if not future.empty:
                vals.append(future["WARP_W"].values[0])
                ws.append(weights[i])
        if vals:
            ws = np.array(ws); ws /= ws.sum()
            mean = float(np.average(vals, weights=ws))
            std = float(np.std(vals))
        else:
            decay = 0.91 ** yr
            mean = float(player_row["WARP_W"] * decay)
            std = mean * 0.25
        projs.append(round(mean, 1))
        lows.append(round(mean - std, 1))
        highs.append(round(mean + std, 1))

    return projs, lows, highs


def assign_tier(peak):
    if peak >= 9:    return "All-Time Great",   "tier-alltime"
    if peak >= 6:    return "Franchise Player", "tier-franchise"
    if peak >= 4:    return "Star",             "tier-star"
    if peak >= 2:    return "Starter",          "tier-starter"
    if peak >= 0:    return "Role Player",      "tier-role"
    return "Replacement Level", "tier-replacement"


def outcome_dist(peak):
    """Returns rough probability estimates per tier bucket."""
    if peak >= 9:
        return [1, 3, 6, 10, 30, 50]
    if peak >= 6:
        return [3, 5, 9, 18, 47, 18]
    if peak >= 4:
        return [5, 10, 20, 40, 20, 5]
    if peak >= 2:
        return [8, 20, 40, 25, 5, 2]
    return [20, 35, 30, 12, 2, 1]


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

raw = load_data()
df = clean(raw)
df = compute_warp_w(df)
scaler, knn = build_engine(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### TAURASI")
    st.markdown("<p style='font-size:0.72rem;color:#444;letter-spacing:0.12em;'>WNBA PROJECTION SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("---")

    players = sorted(df["Player"].unique())
    selected_player = st.selectbox("Select player", players)

    player_seasons = sorted(df[df["Player"] == selected_player]["Season"].unique(), reverse=True)
    selected_season = st.selectbox("Season", player_seasons)

    seasons_ahead = st.slider("Projection years", 1, 7, 5)

    st.markdown("---")
    st.markdown("<p style='font-size:0.7rem;color:#333;'>Data: Basketball-Reference<br>Model: k-NN similarity + aging curves<br>Value metric: WARP-W</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown("""
<div class='taurasi-header'>
            <br></br>
  <p class='taurasi-title'>TAURASI</p>
  <p class='taurasi-sub'>Trajectory Analysis Using Regression And Similarity Index · WNBA</p>
</div>
""", unsafe_allow_html=True)

player_row = df[(df["Player"] == selected_player) & (df["Season"] == selected_season)].iloc[0]

projs, lows, highs = project_player(player_row, df, scaler, knn, seasons=seasons_ahead)
peak = max(projs) if projs else player_row["WARP_W"]
tier_label, tier_class = assign_tier(peak)

# Player header card
overseas_txt = "· plays overseas" if player_row["Overseas"] else ""
st.markdown(f"""
<div class='player-card'>
  <p class='player-name'>{selected_player}</p>
  <p class='player-meta'>{player_row['Pos']} &nbsp;·&nbsp; Age {int(player_row['Age'])} &nbsp;·&nbsp; {int(selected_season)} season {overseas_txt}</p>
  <span class='tier-badge {tier_class}'>{tier_label}</span>
</div>
""", unsafe_allow_html=True)

# Metric row
warp = player_row["WARP_W"]
bpm  = player_row["BPM"]
ws40 = player_row["WS40"]
st.markdown(f"""
<div class='metric-grid'>
  <div class='metric-box'><div class='metric-label'>Current WARP-W</div><div class='metric-value'>{warp}</div></div>
  <div class='metric-box'><div class='metric-label'>Box Plus/Minus</div><div class='metric-value'>{bpm}</div></div>
  <div class='metric-box'><div class='metric-label'>Win Shares / 40</div><div class='metric-value'>{ws40:.3f}</div></div>
  <div class='metric-box'><div class='metric-label'>Peak projection</div><div class='metric-value'>{peak}</div></div>
</div>
""", unsafe_allow_html=True)

# Two column layout
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("<p class='section-label'>WARP-W projections</p>", unsafe_allow_html=True)

    ages = [int(player_row["Age"]) + y for y in range(1, seasons_ahead + 1)]
    labels = [f"Yr {y+1}  (age {ages[y]})" for y in range(seasons_ahead)]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=highs + lows[::-1],
        fill="toself",
        fillcolor="rgba(74,222,128,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Projection line
    fig.add_trace(go.Scatter(
        x=labels,
        y=projs,
        mode="lines+markers",
        line=dict(color="#4ade80", width=2),
        marker=dict(size=8, color="#4ade80", line=dict(color="#0d0d10", width=2)),
        name="Projected WARP-W",
        hovertemplate="<b>%{x}</b><br>WARP-W: %{y}<extra></extra>",
    ))

    # Current season reference
    fig.add_hline(
        y=float(warp),
        line_dash="dot",
        line_color="#333",
        annotation_text=f"Current ({warp})",
        annotation_font_color="#555",
        annotation_font_size=11,
    )

    fig.update_layout(
        plot_bgcolor="#111115",
        paper_bgcolor="#111115",
        font=dict(color="#666", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        showlegend=False,
        xaxis=dict(showgrid=False, color="#333", tickfont=dict(size=11, color="#555")),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", color="#333", tickfont=dict(size=11, color="#555"), title="WARP-W"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Outcome distribution
    st.markdown("<p class='section-label' style='margin-top:1rem;'>Outcome distribution (year 1)</p>", unsafe_allow_html=True)
    tiers_list = ["Replacement", "Role Player", "Starter", "Star", "Franchise", "All-Time"]
    probs = outcome_dist(peak)
    highlight = probs.index(max(probs))

    colors = ["#3a1a1a", "#1a1a2a", "#1a1f2a", "#0f2a1a", "#2a1f00", "#2a1f00"]
    text_colors = ["#e06060", "#a0a0e0", "#60c8f5", "#4ade80", "#f5c842", "#f5c842"]
    bar_colors = [text_colors[i] if i == highlight else colors[i] for i in range(6)]

    fig2 = go.Figure(go.Bar(
        x=tiers_list,
        y=probs,
        marker_color=[text_colors[i] if i == highlight else "#222" for i in range(6)],
        marker_line_width=0,
        text=[f"{p}%" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#555"),
        hovertemplate="%{x}: %{y}%<extra></extra>",
    ))
    fig2.update_layout(
        plot_bgcolor="#111115",
        paper_bgcolor="#111115",
        font=dict(color="#555", size=11),
        margin=dict(l=10, r=10, t=20, b=10),
        height=180,
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#444")),
        yaxis=dict(showgrid=False, visible=False),
        bargap=0.3,
    )
    st.plotly_chart(fig2, use_container_width=True)


with col2:
    st.markdown("<p class='section-label'>Top comparables</p>", unsafe_allow_html=True)
    comps = find_comparables(player_row, df, scaler, knn, top_n=5)

    for _, row in comps.iterrows():
        sim = int(row["Sim"])
        bar_w = int(sim * 0.9)
        st.markdown(f"""
        <div class='comp-row'>
          <div>
            <div class='comp-name'>{row['Player']}</div>
            <div class='comp-meta'>Age {int(row['Age'])} · {int(row['Season'])} · BPM {row['BPM']:.1f} · WARP-W {row['WARP_W']}</div>
          </div>
          <div style='text-align:right;'>
            <div class='comp-score'>{sim}</div>
            <div style='width:70px;height:3px;background:#1a1a1a;border-radius:2px;margin-top:4px;'>
              <div style='width:{bar_w}%;height:100%;background:#378ADD;border-radius:2px;'></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart
    st.markdown("<p class='section-label' style='margin-top:1.5rem;'>Player profile</p>", unsafe_allow_html=True)
    categories = ["Scoring", "Rebounding", "Playmaking", "Efficiency", "Value"]
    max_vals = {"PTS": 30, "TRB": 14, "AST": 10, "WS40": 0.35, "WARP_W": 12}
    vals_radar = [
        min(player_row["PTS"] / max_vals["PTS"] * 10, 10),
        min(player_row["TRB"] / max_vals["TRB"] * 10, 10),
        min(player_row["AST"] / max_vals["AST"] * 10, 10),
        min(player_row["WS40"] / max_vals["WS40"] * 10, 10),
        min(max(player_row["WARP_W"], 0) / max_vals["WARP_W"] * 10, 10),
    ]

    fig3 = go.Figure(go.Scatterpolar(
        r=vals_radar + [vals_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(74,222,128,0.12)",
        line=dict(color="#4ade80", width=1.5),
        marker=dict(size=5, color="#4ade80"),
    ))
    fig3.update_layout(
        polar=dict(
            bgcolor="#111115",
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor="#1a1a1a", linecolor="#1a1a1a"),
            angularaxis=dict(tickfont=dict(size=11, color="#555"), linecolor="#222", gridcolor="#1a1a1a"),
        ),
        paper_bgcolor="#111115",
        margin=dict(l=20, r=20, t=20, b=20),
        height=220,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

#
# Footer
st.markdown("""
<div class='taurasi-footer'>
  TAURASI v0.1 · Sample dataset · Replace load_data() with scraped Basketball-Reference data for full projections
</div>
""", unsafe_allow_html=True)