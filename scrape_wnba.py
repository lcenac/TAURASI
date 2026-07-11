"""
TAURASI — WNBA Data Scraper
Uses curl_cffi to spoof Chrome's TLS fingerprint and bypass stats.nba.com blocking.

Run once:  python scrape_wnba.py
Output:    wnba_data.csv
"""

from curl_cffi import requests
import pandas as pd
import time
import random


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
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


BASE_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
BIO_URL = "https://stats.nba.com/stats/leaguedashplayerbiostats"
INDEX_URL = "https://stats.nba.com/stats/playerindex"

LEAGUE_ID = "10"       # WNBA
START_YEAR = 2026
END_YEAR = 2026

DELAY_MIN = 4
DELAY_MAX = 7


POSITION_NORMALIZE = {
    "Guard": "G",
    "Forward": "F",
    "Center": "C",
    "Guard-Forward": "G-F",
    "Forward-Guard": "F-G",
    "Forward-Center": "F-C",
    "Center-Forward": "C-F",
    "G": "G",
    "F": "F",
    "C": "C",
    "G-F": "G-F",
    "F-G": "F-G",
    "F-C": "F-C",
    "C-F": "C-F",
}


def random_delay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def nba_request(url, params=None):
    """
    More reliable NBA request for GitHub Actions.
    Uses warmup + retries because stats.nba.com
    sometimes silently drops cloud requests.
    """

    # Warm up NBA session
    try:
        requests.get(
            "https://www.nba.com",
            headers=HEADERS,
            impersonate="chrome120",
            timeout=30,
        )
    except:
        pass


    for attempt in range(1, 4):

        try:
            print(f"    Request attempt {attempt}/3")

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                impersonate="chrome120",
                timeout=90,
            )

            response.raise_for_status()

            return response


        except Exception as e:
            print(
                f"    Attempt failed: {str(e)[:150]}"
            )

            if attempt < 3:
                wait = attempt * 15
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)


    return None



def fetch_stats(year, measure_type="Base"):

    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": 0,
        "LeagueID": LEAGUE_ID,
        "Location": "",
        "MeasureType": measure_type,
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": "",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": str(year),
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "",
        "TwoWay": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }


    try:

        resp = nba_request(
            BASE_URL,
            params
        )

        if resp is None:
            return None


        data = resp.json()

        cols = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        df = pd.DataFrame(
            rows,
            columns=cols
        )

        df["Season"] = year

        return df


    except Exception as e:

        print(
            f"\n  [Base] Error {year}: {e}"
        )

        return None
    

def fetch_bio(year):

    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "Height": "",
        "LeagueID": LEAGUE_ID,
        "Location": "",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": "",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "Season": str(year),
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "StarterBench": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }


    try:

        resp = nba_request(
            BIO_URL,
            params
        )

        if resp is None:
            return None


        data = resp.json()

        cols = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        return pd.DataFrame(
            rows,
            columns=cols
        )


    except Exception as e:

        print(
            f"\n  [Bio] Error {year}: {e}"
        )

        return None


def fetch_player_index(year):
    """
    One call returns POSITION (plus height/weight/draft info) for every
    player in the league for a season. Much faster than hitting
    commonplayerinfo per player, since leaguedashplayerbiostats does
    not expose position for the WNBA.
    """

    season_str = f"{year}-{str(year + 1)[-2:]}"  # e.g. 2026 -> "2026-27"

    params = {
        "College": "",
        "Country": "",
        "DraftPick": "",
        "DraftRound": "",
        "DraftYear": "",
        "Height": "",
        "Historical": 1,
        "LeagueID": LEAGUE_ID,
        "Season": season_str,
        "SeasonType": "Regular Season",
        "TeamID": 0,
        "Weight": "",
    }


    try:

        resp = nba_request(
            INDEX_URL,
            params
        )

        if resp is None:
            return None


        data = resp.json()

        cols = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        return pd.DataFrame(
            rows,
            columns=cols
        )


    except Exception as e:

        print(
            f"\n  [PlayerIndex] Error {year}: {e}"
        )

        return None



def merge_season(base_df, adv_df, bio_df, index_df, year):

    base_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "AGE",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "OREB",
        "DREB",
        "PLUS_MINUS",
        "W_PCT",
        "Season",
    ]


    adv_cols = [
        "PLAYER_ID",
        "NET_RATING",
        "USG_PCT",
        "PIE",
        "OFF_RATING",
        "DEF_RATING",
        "TS_PCT",
        "AST_PCT",
        "REB_PCT",
    ]


    base_df = base_df[
        [c for c in base_cols if c in base_df.columns]
    ].copy()


    if adv_df is not None and not adv_df.empty:

        adv_df = adv_df[
            [c for c in adv_cols if c in adv_df.columns]
        ].copy()

        df = base_df.merge(
            adv_df,
            on="PLAYER_ID",
            how="left"
        )

    else:
        df = base_df.copy()



    # ==========================
    # Add bio information
    # ==========================

    if bio_df is not None and not bio_df.empty:

        bio_cols = [
            "PLAYER_ID",
            "PLAYER_HEIGHT_INCHES",
            "PLAYER_HEIGHT",
            "PLAYER_WEIGHT",
            "DRAFT_YEAR",
            "DRAFT_ROUND",
            "DRAFT_NUMBER",
        ]


        bio_cols = [
            c for c in bio_cols
            if c in bio_df.columns
        ]


        df = df.merge(
            bio_df[bio_cols],
            on="PLAYER_ID",
            how="left"
        )



    # ==========================
    # Add position from PlayerIndex
    # ==========================

    if index_df is not None and not index_df.empty:

        idx_cols = [
            c for c in ["PERSON_ID", "POSITION"]
            if c in index_df.columns
        ]

        if "PERSON_ID" in idx_cols and "POSITION" in idx_cols:

            idx_slim = (
                index_df[idx_cols]
                .rename(columns={"PERSON_ID": "PLAYER_ID"})
                .drop_duplicates(subset=["PLAYER_ID"])
            )

            df = df.merge(
                idx_slim,
                on="PLAYER_ID",
                how="left"
            )

        else:
            print(
                "  [PlayerIndex] Expected columns not found — "
                f"got {list(index_df.columns)}"
            )



    df = df.rename(columns={

        "PLAYER_NAME": "Player",
        "TEAM_ABBREVIATION": "Team",

        "AGE": "Age",
        "GP": "G",
        "MIN": "MP",

        "REB": "TRB",

        "FG_PCT": "FG%",
        "FG3_PCT": "3P%",
        "FT_PCT": "FT%",

        "OREB": "ORB",
        "DREB": "DRB",

        "PLUS_MINUS": "PlusMinus",
        "W_PCT": "WinPct",

        "NET_RATING": "NetRtg",
        "USG_PCT": "USG%",

        "OFF_RATING": "ORtg",
        "DEF_RATING": "DRtg",

        "TS_PCT": "TS%",
        "AST_PCT": "AST%",
        "REB_PCT": "REB%",
        
        "PLAYER_HEIGHT_INCHES": "Height_IN",
        "PLAYER_HEIGHT": "Height",
        "PLAYER_WEIGHT": "Weight",

        "DRAFT_YEAR": "DraftYear",
        "DRAFT_ROUND": "DraftRound",
        "DRAFT_NUMBER": "DraftNumber",

    })



    numeric_cols = [

        "MP",
        "G",
        "PTS",
        "TRB",
        "AST",
        "Age",
        "STL",
        "BLK",
        "TOV",

        "NetRtg",
        "USG%",
        "PIE",

        "Height_IN",
        "Weight",

        "DraftYear",
        "DraftRound",
        "DraftNumber",

    ]


    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )



    # minimum sample size

    df = df[df["MP"] >= 8].copy()



    # ==========================
    # Advanced metrics
    # ==========================


    net = df.get(
        "NetRtg",
        pd.Series(0.0,index=df.index)
    ).fillna(0)


    usg = df.get(
        "USG%",
        pd.Series(0.2,index=df.index)
    ).fillna(0.2)


    pie = df.get(
        "PIE",
        pd.Series(0.1,index=df.index)
    ).fillna(0.1)



    df["BPM"] = (
        (net * .4)
        + (usg * 15)
        + (pie * 20)
        - 8
    ).round(2)



    df["WS"] = (
        pie
        * df["G"].fillna(1)
        * df["MP"].fillna(20)
        / 40
    ).round(2)



    df["WS40"] = pie.round(3)



    # ==========================
    # Position
    # ==========================

    # PlayerIndex supplies raw POSITION (e.g. "Guard", "Forward-Center").
    # Normalize to short codes; fall back to UNK if missing.

    if "POSITION" in df.columns:
        df["Pos"] = df["POSITION"].map(POSITION_NORMALIZE).fillna("UNK")
        df = df.drop(columns=["POSITION"])
    else:
        df["Pos"] = "UNK"



    df["Season"] = year


    return df.reset_index(drop=True)





def scrape_all():

    all_frames = []
    failed_years = []


    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        print(
            f"Fetching {year}...",
            flush=True
        )


        base_df = fetch_stats(
            year,
            "Base"
        )

        random_delay()


        if base_df is None or base_df.empty:

            print(
                "FAILED"
            )

            failed_years.append(year)

            continue



        adv_df = fetch_stats(
            year,
            "Advanced"
        )

        random_delay()



        bio_df = fetch_bio(
            year
        )

        random_delay()


        index_df = fetch_player_index(
            year
        )

        random_delay()



        season_df = merge_season(
            base_df,
            adv_df,
            bio_df,
            index_df,
            year
        )


        all_frames.append(
            season_df
        )


        print(
            f"OK — {len(season_df)} players"
        )



    if not all_frames:

        print(
            "No data collected."
        )

        return



    new_data = pd.concat(
        all_frames,
        ignore_index=True
    )


    try:

        old_data = pd.read_csv(
            "wnba_data.csv"
        )


        # Remove old versions of scraped seasons

        old_data = old_data[
            ~old_data["Season"].isin(
                new_data["Season"].unique()
            )
        ]


        full = pd.concat(
            [
                old_data,
                new_data
            ],
            ignore_index=True
        )


    except FileNotFoundError:

        full = new_data



    full = (
        full
        .sort_values(
            [
                "Season",
                "Player"
            ]
        )
        .reset_index(drop=True)
    )


    full.to_csv(
        "wnba_data.csv",
        index=False
    )



    print(
        "\n=============================="
    )

    print(
        f"Saved {len(full)} rows"
    )

    print(
        f"Seasons: {sorted(full['Season'].unique())}"
    )

    print(
        f"Players: {full['Player'].nunique()}"
    )


    if failed_years:

        print(
            f"Failed years: {failed_years}"
        )


    print(
        "=============================="
    )





if __name__ == "__main__":

    scrape_all()