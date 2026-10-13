# main.py

import pandas as pd
from models.dixon_coles         import DixonColesModel
from evaluation.backtester      import Backtester
from evaluation.bookmaker       import BookmakerComparison
from betting.strategy           import BettingStrategy
from config import (SEASONS, DIXON_COLES_XI, TEST_SEASON, BOOKMAKER,
                    CLOSING_BOOKMAKER, BANK, STAKE, MIN_EDGE, MAX_EDGE)


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
          f"-> {data['Date'].max().date()}\n")
    return data


if __name__ == '__main__':

    # ── Load ───────────────────────────────────────────────────────
    df = load_seasons(SEASONS)

    # ── Train/test split ───────────────────────────────────────────
    # Predict only TEST_SEASON, training on every match before it.
    # min_train = number of matches that precede the test season, so the
    # first prediction is made after fitting on all prior seasons.
    if TEST_SEASON not in set(df['season']):
        raise ValueError(f"TEST_SEASON '{TEST_SEASON}' not found in loaded data")

    min_train = int((df['season'] != TEST_SEASON).sum())
    n_test    = int((df['season'] == TEST_SEASON).sum())
    print(f"Training on {min_train} matches before {TEST_SEASON}; "
          f"predicting {n_test} matches in {TEST_SEASON}.\n")

    # ── Backtest ───────────────────────────────────────────────────
    bt      = Backtester(
        xi            = DIXON_COLES_XI,
        min_train     = min_train,
        retrain_every = 10
    )
    results = bt.run(df)
    bt.summary(results)

    # ── Bookmaker comparison ───────────────────────────────────────
    # Bet at BOOKMAKER (opening), grade every bet vs CLOSING_BOOKMAKER.
    bk       = BookmakerComparison(margin_method='basic', bookmaker=BOOKMAKER)
    bk_probs = bk.extract_odds(df, reference=CLOSING_BOOKMAKER)
    merged   = bk.merge_with_results(results, bk_probs)
    bk.full_report(merged)

    # ── Betting simulation ─────────────────────────────────────────
    print("\nRunning betting simulation...\n")

    # Try all three staking methods
    for method in ['flat', 'kelly', 'edge']:
        print(f"\n{'-' * 55}")
        strategy = BettingStrategy(
            bank     = BANK,
            stake    = STAKE,
            min_edge = MIN_EDGE,
            max_edge = MAX_EDGE,
            method   = method
        )
        bets = strategy.find_value_bets(merged)
        strategy.summary(bets)

    # ── Plot best method ───────────────────────────────────────────
    strategy = BettingStrategy(
        bank     = BANK,
        stake    = STAKE,
        min_edge = MIN_EDGE,
        max_edge = MAX_EDGE,
        method   = 'flat'
    )
    bets = strategy.find_value_bets(merged)
    strategy.plot(bets)

    # ── Save bets ──────────────────────────────────────────────────
    bets.to_csv('outputs/bets.csv', index=False)
    print("\nBets saved to outputs/bets.csv")