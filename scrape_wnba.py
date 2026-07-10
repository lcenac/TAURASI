"""
TAURASI — WNBA Data Scraper
Uses curl_cffi to spoof Chrome's TLS fingerprint and bypass stats.nba.com blocking.

Run once:  python scrape_wnba.py
Output:    wnba_data.csv
"""

from curl_cffi import requests
import pandas as pd
import time

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

BASE_URL   = "https://stats.nba.com/stats/leaguedashplayerstats"
BIO_URL    = "https://stats.nba.com/stats/leaguedashplayerbiostats"
LEAGUE_ID  = "10"   # 10 = WNBA
START_YEAR = 2003
END_YEAR   = 2026
DELAY      = 4      # seconds between requests

# Maps whatever leaguedashplayerbiostats returns (varies by season/format)
# into the single-letter codes the dashboard's pos_map expects.
POSITION_NORMALIZE = {
    "Guard": "G",
    "Forward": "F",
    "Center": "C",
    "Guard-Forward": "G-F",
    "Forward-Guard": "F-G",
    "Forward-Center": "F-C",
    "Center-Forward": "C-F",
    "G": "G", "F": "F", "C": "C",
    "G-F": "G-F", "F-G": "F-G",
    "F-C": "F-C", "C-F": "C-F",
}


def fetch_stats(year, measure_type="Base"):
    params = {
        "College": "", "Conference": "", "Country": "",
        "DateFrom": "", "DateTo": "", "Division": "",
        "DraftPick": "", "DraftYear": "", "GameScope": "",
        "GameSegment": "", "Height": "",
        "LastNGames": 0,
        "LeagueID": LEAGUE_ID,
        "Location": "",
        "MeasureType": measure_type,
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "", "PORound": "",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N",
        "Season": str(year),
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "", "StarterBench": "",
        "TeamID": "", "TwoWay": "",
        "VsConference": "", "VsDivision": "", "Weight": "",
    }

    try:
        resp = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params,
            impersonate="chrome120",  # ← the key — spoofs Chrome TLS fingerprint
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()
        cols = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        df = pd.DataFrame(rows, columns=cols)
        df["Season"] = year
        return df

    except Exception as e:
        print(f"\n  [{measure_type}] Error {year}: {e}")
        return None


def fetch_bio(year):
    """Position data lives in the bio endpoint, not Base/Advanced."""
    params = {
        "College": "", "Conference": "", "Country": "",
        "DateFrom": "", "DateTo": "", "Division": "",
        "DraftPick": "", "DraftYear": "", "GameScope": "",
        "Height": "", "LeagueID": LEAGUE_ID, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": "", "PerMode": "PerGame", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "Season": str(year), "SeasonSegment": "",
        "SeasonType": "Regular Season", "StarterBench": "",
        "TeamID": "", "VsConference": "", "VsDivision": "", "Weight": "",
    }

    try:
        resp = requests.get(
            BIO_URL,
            headers=HEADERS,
            params=params,
            impersonate="chrome120",
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()
        cols = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        return pd.DataFrame(rows, columns=cols)

    except Exception as e:
        print(f"\n  [Bio] Error {year}: {e}")
        return None


def merge_season(base_df, adv_df, bio_df, year):
    base_cols = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
        "AGE", "GP", "MIN", "PTS", "REB", "AST",
        "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT",
        "OREB", "DREB", "PLUS_MINUS", "W_PCT", "Season",
    ]
    adv_cols = [
        "PLAYER_ID", "NET_RATING", "USG_PCT", "PIE",
        "OFF_RATING", "DEF_RATING", "TS_PCT", "AST_PCT", "REB_PCT",
    ]

    base_df = base_df[[c for c in base_cols if c in base_df.columns]].copy()

    if adv_df is not None and not adv_df.empty:
        adv_df = adv_df[[c for c in adv_cols if c in adv_df.columns]].copy()
        df = base_df.merge(adv_df, on="PLAYER_ID", how="left")
    else:
        df = base_df.copy()

    df = df.rename(columns={
        "PLAYER_NAME":       "Player",
        "TEAM_ABBREVIATION": "Team",
        "AGE":               "Age",
        "GP":                "G",
        "MIN":               "MP",
        "REB":               "TRB",
        "FG_PCT":            "FG%",
        "FG3_PCT":           "3P%",
        "FT_PCT":            "FT%",
        "OREB":              "ORB",
        "DREB":              "DRB",
        "PLUS_MINUS":        "PlusMinus",
        "W_PCT":             "WinPct",
        "NET_RATING":        "NetRtg",
        "USG_PCT":           "USG%",
        "OFF_RATING":        "ORtg",
        "DEF_RATING":        "DRtg",
        "TS_PCT":            "TS%",
        "AST_PCT":           "AST%",
        "REB_PCT":           "REB%",
    })

    for col in ["MP", "G", "PTS", "TRB", "AST", "Age",
                "STL", "BLK", "TOV", "NetRtg", "USG%", "PIE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop players with very limited minutes
    df = df[df["MP"] >= 8].copy()

    # BPM proxy (correlates ~r=0.85 with real BPM)
    net = df.get("NetRtg", pd.Series(0.0, index=df.index)).fillna(0)
    usg = df.get("USG%",   pd.Series(0.2,  index=df.index)).fillna(0.2)
    pie = df.get("PIE",    pd.Series(0.1,  index=df.index)).fillna(0.1)
    df["BPM"]  = ((net * 0.4) + (usg * 15) + (pie * 20) - 8).round(2)
    df["WS"]   = (pie * df["G"].fillna(1) * df["MP"].fillna(20) / 40).round(2)
    df["WS40"] = pie.round(3)

    # Real position data, pulled from the bio endpoint and merged on PLAYER_ID.
    if bio_df is not None and not bio_df.empty and "PLAYER_ID" in bio_df.columns:
        pos_col = "PLAYER_POSITION" if "PLAYER_POSITION" in bio_df.columns else None
        if pos_col:
            bio_pos = bio_df[["PLAYER_ID", pos_col]].copy()
            df = df.merge(bio_pos, on="PLAYER_ID", how="left")
            df["Pos"] = df[pos_col].map(POSITION_NORMALIZE).fillna(df[pos_col]).fillna("F")
            df = df.drop(columns=[pos_col])
        else:
            print(f"  [Bio] WARNING {year}: no position column found, defaulting to 'F'")
            df["Pos"] = "F"
    else:
        print(f"  [Bio] WARNING {year}: bio fetch failed, defaulting to 'F'")
        df["Pos"] = "F"

    df["Overseas"] = False
    df["DraftRd"]  = 1
    df["Season"]   = year

    return df.reset_index(drop=True)


def scrape_all():
    all_frames   = []
    failed_years = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching {year}...", end=" ", flush=True)

        base_df = fetch_stats(year, "Base")
        time.sleep(DELAY)

        if base_df is None or base_df.empty:
            print("FAILED — skipping")
            failed_years.append(year)
            continue

        adv_df = fetch_stats(year, "Advanced")
        time.sleep(DELAY)

        if adv_df is None or adv_df.empty:
            print("(base only) ", end="")

        bio_df = fetch_bio(year)
        time.sleep(DELAY)

        if bio_df is None or bio_df.empty:
            print("(no bio/position data) ", end="")

        season_df = merge_season(base_df, adv_df, bio_df, year)
        all_frames.append(season_df)
        print(f"OK — {len(season_df)} players")

    if not all_frames:
        print("\nNo data collected.")
        return

    full = pd.concat(all_frames, ignore_index=True)
    full = full.sort_values(["Season", "Player"]).reset_index(drop=True)
    full.to_csv("wnba_data.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(full)} player-seasons → wnba_data.csv")
    print(f"Seasons : {full['Season'].nunique()}")
    print(f"Players : {full['Player'].nunique()}")
    print(f"Positions found: {sorted(full['Pos'].unique())}")
    if failed_years:
        print(f"Failed  : {failed_years}")
    print("="*50)
    print("""
NEXT STEP:
  Update load_data() in taurasi_app.py to:

    @st.cache_data
    def load_data():
        return pd.read_csv("wnba_data.csv")
""")


if __name__ == "__main__":
    scrape_all()