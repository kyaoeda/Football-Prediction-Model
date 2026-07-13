# main.py

import pandas as pd
from models.dixon_coles         import DixonColesModel
from evaluation.backtester      import Backtester
from evaluation.bookmaker       import BookmakerComparison
from config import SEASONS, DIXON_COLES_XI


def load_seasons(seasons: dict) -> pd.DataFrame:
    dfs = []
    for season, url in seasons.items():
        print(f"Loading {season}...")
        df           = pd.read_csv(url)
        df['season'] = season
        dfs.append(df)

    data         = pd.concat(dfs, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data         = data.sort_values('Date').reset_index(drop=True)
    data         = data.dropna(subset=['FTHG', 'FTAG'])

    print(f"\nLoaded {len(data)} matches across {len(seasons)} seasons")
    print(f"Date range: {data['Date'].min().date()} "
          f"→ {data['Date'].max().date()}\n")
    return data


if __name__ == '__main__':

    # ── Load data ──────────────────────────────────────────────────
    df = load_seasons(SEASONS)

    # ── Backtest ───────────────────────────────────────────────────
    bt      = Backtester(xi=DIXON_COLES_XI, min_train=380*4, retrain_every=10)
    results = bt.run(df)
    bt.summary(results)

    # ── Bookmaker comparison ───────────────────────────────────────
    bk       = BookmakerComparison(margin_method='basic')
    bk_probs = bk.extract_odds(df)
    merged   = bk.merge_with_results(results, bk_probs)

    # Full report
    bk.full_report(merged)

    # Visualisation
    bk.plot(merged)

    # Where does model disagree most with bookmaker?
    disagree = bk.disagreement(merged, threshold=0.07)
    print(f"\nTop 10 biggest disagreements:")
    print(disagree[[
        'date', 'home_team', 'away_team',
        'home_win', 'bk_home',
        'diff_home', 'outcome'
    ]].head(10).to_string())