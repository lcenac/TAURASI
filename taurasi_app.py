"""
TAURASI — Trajectory Analysis Using Regression And Similarity Index
WNBA Player Projection System · Streamlit Dashboard
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px

st.write("CSV size:", os.path.getsize("wnba_data.csv") / 1024 / 1024, "MB")
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="TAURASI · WNBA Projections",
    page_icon="taurasi_favicon_lineart.svg",
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
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .player-card-info { flex: 1; }
  .player-card-logo {
    width: 56px;
    height: 56px;
    object-fit: contain;
    flex-shrink: 0;
    opacity: 0.95;
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


def get_team_logo_url(team_abbr: str) -> str:
    """
    Team logos are served by stats.wnba.com keyed on team abbreviation
    (the same value already stored in the 'Team' column from the
    scraper). No separate team_id lookup needed.
    """
    if not isinstance(team_abbr, str) or not team_abbr:
        return ""
    return f"https://stats.wnba.com/media/img/teams/logos/{team_abbr}.svg"


SAMPLE_MIN_GAMES = 15  # below this, tier/peak projections get flagged as low-confidence


def compute_warp_w(df, replacement_bpm=-2.0, league_avg_mp=28.5):
    """
    WARP-W is now scaled by each player's ACTUAL games played (df["G"])
    instead of a flat 40-game assumption. The old flat constant treated
    a hot 8-game stretch the same as a full healthy season, which is
    what was inflating low-minute players (e.g. someone who logs great
    per-minute numbers in a handful of appearances gets extrapolated
    to a full season of that rate with no penalty for the tiny sample
    it came from).
    """
    df = df.copy()
    games = df["G"].fillna(0)
    df["WARP_W"] = (
        (df["BPM"] - replacement_bpm)
        * (df["MP"] / league_avg_mp)
        * (1 / 22.0)
        * 44
    ).round(1)
    return df


def clean(df):
    pos_map = {"G": 1, "G-F": 2, "F-G": 2, "F": 3, "F-C": 4, "C-F": 4, "C": 5}
    df = df.copy()
    df["PosNum"] = df["Pos"].map(pos_map).fillna(3)
    df = df.dropna(subset=["BPM", "Age"])
    return df


# PosNum included so comps are drawn from similar positional roles --
# without it, a high-usage guard and a high-usage forward with similar
# counting stats could rank as close comps despite playing very
# different roles.
FEATURES = [
    "Age",
    "BPM",
    "WS40",
    "PTS",
    "TRB",
    "AST",
    "Height_IN",
    "PosNum",
]


def _feature_frame(df):
    """
    Build the model-ready feature matrix. Missing values are filled
    with each column's league median rather than 0 -- filling with 0
    would treat a player with an unknown Height_IN or BPM as an
    extreme outlier once the values are scaled, which most often hits
    rookies and overseas players (exactly the players you'd most want
    reasonable comps for).
    """
    X = df[FEATURES].copy()
    medians = X.median(numeric_only=True)
    return X.fillna(medians), medians


@st.cache_resource
def build_engine(df):
    X, medians = _feature_frame(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=min(15, len(df)), metric="euclidean")
    knn.fit(X_scaled)

    # Reference distance distribution used to turn raw distances into
    # a similarity score that reflects this dataset, rather than a
    # fixed fudge factor. See find_comparables().
    ref_distances, _ = knn.kneighbors(X_scaled, n_neighbors=min(6, len(df)))
    # column 0 is distance to self (0); use the rest
    typical_neighbor_dist = np.median(ref_distances[:, 1:]) if ref_distances.shape[1] > 1 else 1.0

    return scaler, knn, medians, typical_neighbor_dist


def _prep_query_row(player_row, scaler, medians):
    X_p = player_row[FEATURES].copy()
    X_p = X_p.fillna(medians)
    return scaler.transform(pd.DataFrame([X_p]))


def find_comparables(player_row, df, scaler, knn, medians, typical_neighbor_dist, top_n=5):
    X_scaled = _prep_query_row(player_row, scaler, medians)
    distances, indices = knn.kneighbors(X_scaled)
    comps = df.iloc[indices[0][1:]].copy()
    raw = distances[0][1:top_n + 1]
    comps = comps.head(top_n).copy()

    # Similarity as an exponential decay anchored to the dataset's own
    # typical neighbor distance, instead of an arbitrary max_d * 1.8
    # rescale. A comp at roughly the "typical" nearest-neighbor
    # distance lands near ~37% (1/e); much closer comps push toward
    # 100%, much farther ones decay toward 0. Still a heuristic, but
    # it moves with the actual data instead of a hardcoded constant.
    scale = typical_neighbor_dist if typical_neighbor_dist > 0 else 1.0
    comps["Sim"] = (100 * np.exp(-raw[:len(comps)] / scale)).clip(0, 100).round(0).astype(int)

    return comps[["Player", "Season", "Age", "Pos", "BPM", "WARP_W", "Sim"]]


def project_player(player_row, df, scaler, knn, medians, seasons=5, top_n=10):
    X_scaled = _prep_query_row(player_row, scaler, medians)
    distances, indices = knn.kneighbors(X_scaled, n_neighbors=min(top_n + 1, len(df)))
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


def is_low_sample(player_row, min_games=SAMPLE_MIN_GAMES):
    g = player_row.get("G", None)
    return (g is None) or (pd.isna(g)) or (g < min_games)


def percentile_tier(value, season_distribution):
    """
    Tier based on percentile rank within that season's full league
    WARP-W distribution, instead of fixed absolute cutoffs. Fixed
    cutoffs compress the gap between a clearly elite player and a
    merely solid one whenever both happen to land in the same few
    raw WARP-W points -- which is common when comparing an
    established star's (declining) future projection against a
    low-minute player's best-case breakout projection. Percentile
    keeps "the best player in the league this season" and "a good
    rotation piece" from sharing a tier just because their raw
    numbers happen to be close.
    """
    season_distribution = season_distribution.dropna()
    if season_distribution.empty:
        return "Unranked", "tier-role"
    pct = (season_distribution < value).mean() * 100
    if pct >= 97: return "All-Time Great", "tier-alltime"
    if pct >= 90: return "Franchise Player", "tier-franchise"
    if pct >= 75: return "Star", "tier-star"
    if pct >= 50: return "Starter", "tier-starter"
    if pct >= 20: return "Role Player", "tier-role"
    return "Replacement Level", "tier-replacement"


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
scaler, knn, feature_medians, typical_neighbor_dist = build_engine(df)

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

# projs, lows, highs = project_player(player_row, df, scaler, knn, feature_medians, seasons=seasons_ahead)
projs = [player_row["WARP_W"]] * seasons_ahead
lows = projs
highs = projs
peak = max(projs) if projs else player_row["WARP_W"]

# Current-season tier: how good is this player RIGHT NOW, ranked against
# the full league that season -- not a projected future peak, which can
# make an established star's normal aging decline look similar to a
# low-minute player's optimistic breakout scenario.
season_distribution = df[df["Season"] == selected_season]["WARP_W"]
tier_label, tier_class = percentile_tier(player_row["WARP_W"], season_distribution)

# Separate, secondary label for where the projection thinks they're headed.
peak_tier_label, peak_tier_class = percentile_tier(peak, season_distribution)

# Player header card
team_logo_url = get_team_logo_url(player_row.get("Team", ""))
low_sample = is_low_sample(player_row)
games_played = player_row.get("G", None)

low_sample_badge = ""
if low_sample:
    gp_display = int(games_played) if pd.notna(games_played) else "?"
    low_sample_badge = f"""<span class='tier-badge tier-role' style='margin-left:8px;'>
        Small Sample &middot; {gp_display} GP
    </span>"""

peak_badge = ""
if peak_tier_class != tier_class:
    peak_badge = f"""<span class='tier-badge {peak_tier_class}' style='margin-left:8px;opacity:0.75;' title='Where the model projects this player trending toward'>
        Trajectory: {peak_tier_label}
    </span>"""

st.markdown(f"""
<div class='player-card'>
  <div class='player-card-info'>
    <p class='player-name'>{selected_player}</p>
    <p class='player-meta'>{player_row['Pos']} &nbsp;·&nbsp; Age {int(player_row['Age'])} &nbsp;·&nbsp; {int(selected_season)} season &nbsp;·&nbsp; {player_row.get('Team', '')}</p>
    <span class='tier-badge {tier_class}'>{tier_label}</span>{peak_badge}{low_sample_badge}
  </div>
  <img class='player-card-logo' src='{team_logo_url}' onerror="this.style.display='none'" />
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
    comps = find_comparables(player_row, df, scaler, knn, feature_medians, typical_neighbor_dist, top_n=5)

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

