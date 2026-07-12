# TAURASI 

Trajectory Analysis Using Regression And Similarity Index 

> Named after Diana Taurasi, the WNBA's all-time leading scorer.

🔗 **Live demo:** [taurasi.streamlit.app]

---

## Overview

TAURASI projects WNBA player performance by finding statistically similar players from historical data and using their career trajectories to forecast future output. Instead of relying on a single regression line, the model identifies each player's closest statistical "comps" and blends their outcomes — an approach inspired by tools like PECOTA in baseball, adapted for the structure and scale of WNBA data.

The project includes an interactive dashboard for exploring projections, comparing players, and visualizing performance trends.

## Features

- **k-NN based projections** — forecasts player stats using weighted similarity to historical player-seasons
- **Interactive dashboard** built with Streamlit and Plotly for exploring projections and comparables
- **Player comparison tool** to see which historical players a given player most closely resembles
- **Position data scraper** for pulling and maintaining up-to-date WNBA roster and position data
- **Visual trend analysis** of player development curves across a career arc

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | Python, scikit-learn (k-NN) |
| Data | Custom WNBA scraper, pandas |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/taurasi.git
cd taurasi
pip install -r requirements.txt
```

### Running locally

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Data

TAURASI pulls WNBA player statistics and position data via a custom scraper. [Add a note here on data source(s), update frequency, and any caveats about coverage/seasons included.]

To refresh the underlying dataset:

```bash
python scraper.py
```

## How the Model Works

1. **Feature engineering** — player-seasons are represented as vectors of relevant statistical features (e.g., usage rate, efficiency metrics, age, position)
2. **Neighbor search** — for a target player, the model finds the *k* most similar historical player-seasons using scikit-learn's k-NN
3. **Projection** — the target player's future performance is projected as a weighted blend of how those comparable players performed at the same career stage

## Roadmap

- [ ] Expand feature set for position-specific modeling
- [ ] Add confidence intervals to projections
- [ ] Improve scraper reliability for position data
- [ ] Historical backtesting view to validate projection accuracy

## Project Background

TAURASI was built as part of a portfolio of sports analytics projects exploring the intersection of full-stack engineering and data science, alongside [Hot Shot](#), a full-stack NBA/WNBA streak tracker.


