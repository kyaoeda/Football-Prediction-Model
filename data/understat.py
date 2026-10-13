# data/understat.py
# ─────────────────────────────────────────────────────────────────
# Fetch match-level xG data from Understat (via penaltyblog) and
# align it with football-data.co.uk match rows.
#
# Understat labels a season by its starting year, e.g. '2025' is the
# 2025-26 campaign. Team names are the full versions ("Manchester
# City"), so we map them to football-data's short names ("Man City").
#
# Results are cached as CSV under data/ so we only scrape each season
# once. Delete the cache file (or pass force_refresh=True) to refetch.
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import penaltyblog as pb

from config import ROOT_DIR

UNDERSTAT_COMPETITION = "ENG Premier League"
CACHE_DIR = ROOT_DIR / "data" / "understat_cache"

# Understat full name -> football-data.co.uk short name.
# Only the clubs whose spelling differs need an entry; the other ~22
# Premier League names match exactly across both sources.
UNDERSTAT_TO_FD = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Newcastle United":        "Newcastle",
    "Nottingham Forest":       "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
}


def _understat_season(season_label: str) -> str:
    """'2025-26' -> '2025' (Understat's starting-year season id)."""
    return season_label.split("-")[0]


def _fetch_season(season_label: str) -> pd.DataFrame:
    """Scrape one season of Understat fixtures with xG."""
    us_season = _understat_season(season_label)
    scraper   = pb.scrapers.Understat(UNDERSTAT_COMPETITION, us_season)
    fx        = scraper.get_fixtures()

    out = pd.DataFrame({
        "Date":     pd.to_datetime(fx["date"]),
        "HomeTeam": fx["team_home"].replace(UNDERSTAT_TO_FD),
        "AwayTeam": fx["team_away"].replace(UNDERSTAT_TO_FD),
        "xg_home":  pd.to_numeric(fx["xg_home"], errors="coerce"),
        "xg_away":  pd.to_numeric(fx["xg_away"], errors="coerce"),
        "season":   season_label,
    })
    # Keep only played matches (xG present)
    out = out.dropna(subset=["xg_home", "xg_away"]).reset_index(drop=True)
    return out


def load_understat_xg(seasons: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load Understat xG for the given seasons, one cached CSV per season.

    Parameters
    ----------
    seasons : dict
        Same mapping used elsewhere (e.g. config.SEASONS); only the keys
        ('2021-22', ...) are used to derive Understat season ids.
    force_refresh : bool
        Re-scrape even if a cache file exists.

    Returns
    -------
    pd.DataFrame with columns:
        Date, HomeTeam, AwayTeam, xg_home, xg_away, season
        (team names already mapped to football-data spellings)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for season_label in seasons:
        cache_file = CACHE_DIR / f"understat_epl_{_understat_season(season_label)}.csv"

        if cache_file.exists() and not force_refresh:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
        else:
            print(f"Scraping Understat xG for {season_label}...")
            df = _fetch_season(season_label)
            df.to_csv(cache_file, index=False)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data


def merge_xg(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join Understat xG onto a football-data match frame.

    Matches on Date + HomeTeam + AwayTeam. Rows with no Understat match
    keep NaN xG (XGBoost tolerates NaN, so those simply lack xG form).
    """
    merged = matches.merge(
        xg[["Date", "HomeTeam", "AwayTeam", "xg_home", "xg_away"]],
        on  = ["Date", "HomeTeam", "AwayTeam"],
        how = "left",
    )

    matched = merged["xg_home"].notna().sum()
    print(f"xG merge: {matched}/{len(merged)} matches have Understat xG "
          f"({len(merged) - matched} missing)")
    return merged
