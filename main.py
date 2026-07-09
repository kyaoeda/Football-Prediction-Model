# main.py

import pandas as pd
from models.dixon_coles    import DixonColesModel
from evaluation.backtester import Backtester
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
    print(f"Date range: {data['Date'].min().date()} → {data['Date'].max().date()}\n")

    return data


if __name__ == '__main__':

    # ── Load data ──────────────────────────────────────────────────
    df = load_seasons(SEASONS)

    # ── Backtest ───────────────────────────────────────────────────
    bt = Backtester(
        xi            = DIXON_COLES_XI,
        min_train     = 380,      # one full season before predicting
        retrain_every = 10        # retrain every 10 matches
    )

    results = bt.run(df)
    bt.summary(results)

    # ── Save results ───────────────────────────────────────────────
    results.to_csv('outputs/backtest_results.csv', index=False)
    print("Results saved to outputs/backtest_results.csv")