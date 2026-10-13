# features/rolling_form.py
# ─────────────────────────────────────────────────────────────────
# Rolling xG "form" features for a match dataset.
#
# For each team we track a rolling average of its recent xG created
# and xG conceded (plus actual goals), then attach the home team's and
# away team's form to each fixture.
#
# Leakage safety
# --------------
# Each team's rolling window is shifted by one match, so the features
# for match i use only that team's matches strictly BEFORE i. A team's
# very first match has no history -> NaN (XGBoost handles NaN natively,
# which is why newly promoted teams need no special treatment).
# ─────────────────────────────────────────────────────────────────

import pandas as pd

# Feature columns produced by add_rolling_xg_features (in a stable order).
ROLLING_FEATURES = [
    "home_roll_xg_for", "home_roll_xg_against",
    "away_roll_xg_for", "away_roll_xg_against",
    "home_roll_gf", "home_roll_ga",
    "away_roll_gf", "away_roll_ga",
    "xg_diff_att", "xg_diff_def", "form_xg_diff",
]

_BASE_COLS = ["xg_for", "xg_against", "gf", "ga"]


def add_rolling_xg_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Add rolling xG/goals form features to a match frame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Date, HomeTeam, AwayTeam, xg_home, xg_away,
        FTHG, FTAG. Rows may have NaN xG (unmatched fixtures); their
        rolling features simply propagate NaN.
    n : int
        Rolling window length (number of previous matches per team).

    Returns
    -------
    pd.DataFrame — original columns plus ROLLING_FEATURES.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["_match_id"] = df.index

    # ── Long form: one row per team per match ─────────────────────
    home = df[["_match_id", "Date", "HomeTeam",
               "xg_home", "xg_away", "FTHG", "FTAG"]].rename(columns={
        "HomeTeam": "team", "xg_home": "xg_for", "xg_away": "xg_against",
        "FTHG": "gf", "FTAG": "ga"})
    home["venue"] = "home"

    away = df[["_match_id", "Date", "AwayTeam",
               "xg_away", "xg_home", "FTAG", "FTHG"]].rename(columns={
        "AwayTeam": "team", "xg_away": "xg_for", "xg_home": "xg_against",
        "FTAG": "gf", "FTHG": "ga"})
    away["venue"] = "away"

    long = (pd.concat([home, away], ignore_index=True)
              .sort_values(["team", "Date", "_match_id"])
              .reset_index(drop=True))

    # ── Rolling mean of the previous n matches (shifted, no leakage) ─
    grp = long.groupby("team", group_keys=False)
    for col in _BASE_COLS:
        long[f"roll_{col}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(n, min_periods=1).mean()
        )

    roll_cols = [f"roll_{c}" for c in _BASE_COLS]

    # ── Pivot rolling form back onto each fixture ─────────────────
    home_form = (long[long["venue"] == "home"]
                 .set_index("_match_id")[roll_cols]
                 .rename(columns={
                     "roll_xg_for": "home_roll_xg_for",
                     "roll_xg_against": "home_roll_xg_against",
                     "roll_gf": "home_roll_gf",
                     "roll_ga": "home_roll_ga"}))

    away_form = (long[long["venue"] == "away"]
                 .set_index("_match_id")[roll_cols]
                 .rename(columns={
                     "roll_xg_for": "away_roll_xg_for",
                     "roll_xg_against": "away_roll_xg_against",
                     "roll_gf": "away_roll_gf",
                     "roll_ga": "away_roll_ga"}))

    df = df.join(home_form, on="_match_id").join(away_form, on="_match_id")

    # ── Matchup differentials ─────────────────────────────────────
    # Home attack vs away defence, and vice-versa; plus overall form gap.
    df["xg_diff_att"]  = df["home_roll_xg_for"]     - df["away_roll_xg_against"]
    df["xg_diff_def"]  = df["away_roll_xg_for"]     - df["home_roll_xg_against"]
    df["form_xg_diff"] = (
        (df["home_roll_xg_for"] - df["home_roll_xg_against"]) -
        (df["away_roll_xg_for"] - df["away_roll_xg_against"])
    )

    return df.drop(columns="_match_id")
