# features/context.py
# ─────────────────────────────────────────────────────────────────
# "Context" features: things Dixon-Coles structurally CANNOT know.
#
# DC estimates team strength by MLE from goals, with time decay. Any
# feature that merely restates team strength is redundant with it (that
# is why rolling xG form alone failed). These features are different in
# kind — DC has no representation of fatigue, league position, season
# stakes, or the gap between xG and goals:
#
#   rest / congestion  - DC treats a team on 2 days' rest as fresh
#   table & stakes     - DC treats a May dead rubber like a title decider
#   new-team status    - DC handles promoted sides badly (we hand-coded
#                        a prior for exactly this)
#   xG overperformance - DC only ever sees goals, so it cannot know a
#                        team has been out-finishing its chances and is
#                        a regression candidate
#   schedule strength  - whether recent form came against good teams
#
# Leakage safety: every value uses only matches strictly BEFORE the one
# it describes. The league table is rebuilt chronologically and read
# one match before kickoff.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

CONTEXT_FEATURES = [
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_matches_14d", "away_matches_14d",
    "home_ppg", "away_ppg", "ppg_diff",
    "home_position", "away_position", "position_diff",
    "games_played", "season_progress",
    "home_is_new", "away_is_new",
    "home_xg_overperf", "away_xg_overperf",
    "home_opp_strength", "away_opp_strength",
]

_ROLL = 5   # window for overperformance / schedule strength


def _long_form(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with that team's perspective."""
    home = pd.DataFrame({
        "_mid": df["_mid"], "Date": df["Date"], "season": df["season"],
        "team": df["HomeTeam"], "opponent": df["AwayTeam"], "venue": "home",
        "gf": df["FTHG"], "ga": df["FTAG"],
        "xg_for": df["xg_home"], "xg_against": df["xg_away"],
    })
    away = pd.DataFrame({
        "_mid": df["_mid"], "Date": df["Date"], "season": df["season"],
        "team": df["AwayTeam"], "opponent": df["HomeTeam"], "venue": "away",
        "gf": df["FTAG"], "ga": df["FTHG"],
        "xg_for": df["xg_away"], "xg_against": df["xg_home"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.where(long["gf"] > long["ga"], 3,
                              np.where(long["gf"] == long["ga"], 1, 0))
    return long.sort_values(["team", "Date", "_mid"]).reset_index(drop=True)


def _rest_and_congestion(long: pd.DataFrame) -> pd.DataFrame:
    """Days since previous match, and matches played in the prior 14 days."""
    grp = long.groupby("team", group_keys=False)

    prev_date = grp["Date"].shift(1)
    long["rest_days"] = (long["Date"] - prev_date).dt.days

    # Count of that team's matches inside the 14 days before this one
    counts = np.zeros(len(long), dtype=int)
    for _, idx in long.groupby("team").groups.items():
        idx = list(idx)
        dates = long.loc[idx, "Date"].to_numpy()
        for pos, d in enumerate(dates):
            window_start = d - np.timedelta64(14, "D")
            counts[idx[pos]] = int(((dates[:pos] >= window_start) &
                                    (dates[:pos] < d)).sum())
    long["matches_14d"] = counts
    return long


def _table_state(long: pd.DataFrame) -> pd.DataFrame:
    """
    Points, games played and league position BEFORE each match.

    Rebuilt chronologically per season so a match never sees its own
    result, nor any later result.
    """
    long = long.sort_values(["season", "Date", "_mid"]).reset_index(drop=True)

    ppg      = np.full(len(long), np.nan)
    position = np.full(len(long), np.nan)
    played_a = np.zeros(len(long), dtype=int)

    for season, block in long.groupby("season", sort=False):
        pts, played, gd = {}, {}, {}

        # Process one matchday at a time: every match on a given date
        # reads the table as it stood before that date.
        for date, day in block.groupby("Date", sort=True):
            standing = sorted(pts, key=lambda t: (-pts.get(t, 0), -gd.get(t, 0)))
            rank = {t: i + 1 for i, t in enumerate(standing)}

            for i in day.index:
                t = long.at[i, "team"]
                p, g = pts.get(t, 0), played.get(t, 0)
                ppg[i]      = p / g if g else np.nan
                position[i] = rank.get(t, np.nan)
                played_a[i] = g

            # Now apply this date's results
            for i in day.index:
                t = long.at[i, "team"]
                pts[t]    = pts.get(t, 0) + long.at[i, "points"]
                played[t] = played.get(t, 0) + 1
                gd[t]     = gd.get(t, 0) + (long.at[i, "gf"] - long.at[i, "ga"])

    long["ppg"]      = ppg
    long["position"] = position
    long["played"]   = played_a
    return long


def _rolling_context(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling xG overperformance and strength of recent opposition."""
    long = long.sort_values(["team", "Date", "_mid"]).reset_index(drop=True)

    # Finishing vs chance quality; positive = out-scoring its xG
    long["overperf"] = long["gf"] - long["xg_for"]

    grp = long.groupby("team", group_keys=False)
    long["xg_overperf"] = grp["overperf"].transform(
        lambda s: s.shift(1).rolling(_ROLL, min_periods=1).mean())

    # Schedule strength: how good were the opponents recently faced?
    opp_ppg = long.merge(
        long[["_mid", "team", "ppg"]].rename(columns={"team": "opponent",
                                                      "ppg": "_opp_ppg"}),
        on=["_mid", "opponent"], how="left")["_opp_ppg"].to_numpy()
    long["_opp_ppg"] = opp_ppg
    long["opp_strength"] = grp["_opp_ppg"].transform(
        lambda s: s.shift(1).rolling(_ROLL, min_periods=1).mean())
    return long


def _new_team_flags(long: pd.DataFrame) -> pd.DataFrame:
    """1 if the team did not appear in this division the previous season."""
    seasons = sorted(long["season"].unique())
    order   = {s: i for i, s in enumerate(seasons)}
    present = {s: set(long.loc[long["season"] == s, "team"]) for s in seasons}

    def is_new(row):
        i = order[row["season"]]
        if i == 0:
            return np.nan          # unknowable for the first season loaded
        return int(row["team"] not in present[seasons[i - 1]])

    long["is_new"] = long.apply(is_new, axis=1)
    return long


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CONTEXT_FEATURES to a match frame.

    Requires: Date, HomeTeam, AwayTeam, FTHG, FTAG, season, and
    xg_home/xg_away (NaN tolerated — overperformance just becomes NaN).
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["_mid"] = df.index

    long = _long_form(df)
    long = _rest_and_congestion(long)
    long = _table_state(long)
    long = _rolling_context(long)
    long = _new_team_flags(long)

    cols = ["rest_days", "matches_14d", "ppg", "position", "played",
            "xg_overperf", "opp_strength", "is_new"]

    home = (long[long["venue"] == "home"].set_index("_mid")[cols]
            .add_prefix("home_"))
    away = (long[long["venue"] == "away"].set_index("_mid")[cols]
            .add_prefix("away_"))
    df = df.join(home, on="_mid").join(away, on="_mid")

    # ── Derived differentials / season context ────────────────────
    df["rest_diff"]     = df["home_rest_days"] - df["away_rest_days"]
    df["ppg_diff"]      = df["home_ppg"] - df["away_ppg"]
    df["position_diff"] = df["away_position"] - df["home_position"]  # +ve = home ranked better
    df["games_played"]  = df[["home_played", "away_played"]].mean(axis=1)

    # How far into the season we are (proxy for stakes / dead rubbers)
    per_season = df.groupby("season")["games_played"].transform("max")
    df["season_progress"] = df["games_played"] / per_season.replace(0, np.nan)

    return df.drop(columns=["_mid", "home_played", "away_played"])
