# main_totals.py
# ─────────────────────────────────────────────────────────────────
# Over/Under 2.5 goals: can Dixon-Coles beat the totals market?
#
# The scoreline matrix already prices totals, so no new model is fitted
# — this reuses the same walk-forward DC predictions and asks whether
# the two-way market (thinner margin, less efficient than 1X2) is
# softer than the 1X2 market we failed to beat.
#
# Evaluated over 2019-20..2025-26, the seasons carrying Max>2.5 and
# MaxC>2.5 columns, betting the best available price and grading every
# bet against the closing best price.
#
# Run:  python main_totals.py
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from main import load_seasons
from evaluation.backtester import Backtester
from evaluation.totals import (devig_totals, brier, log_loss_binary,
                               find_totals_bets, summarise_totals_bets)
from evaluation.walkforward import paired_bootstrap
from config import (SEASONS_PL_FULL, DIXON_COLES_XI,
                    MIN_EDGE, MAX_EDGE, MIN_ODDS, MAX_ODDS, BANK, STAKE)

FIRST_ODDS_SEASON = "2019-20"    # first season with Max>2.5 / MaxC>2.5
BET_BOOK          = "market_max"
CLOSE_BOOK        = "market_max_closing"
DC_MIN_TRAIN      = 380
DC_RETRAIN_EVERY  = 20


def main():
    df = load_seasons(SEASONS_PL_FULL)

    # ── Walk-forward DC (prices 1X2 and totals from one matrix) ───
    print("\nWalk-forward Dixon-Coles over full history...\n")
    bt      = Backtester(xi=DIXON_COLES_XI,
                         min_train=DC_MIN_TRAIN,
                         retrain_every=DC_RETRAIN_EVERY,
                         totals_line=2.5)
    results = bt.run(df)

    # Restrict to seasons where totals odds exist
    results = results[results['season'] >= FIRST_ODDS_SEASON].copy()
    results['date'] = pd.to_datetime(results['date'])
    print(f"Totals-eligible predictions: {len(results)} "
          f"({results['season'].nunique()} seasons)")

    # ── Market: best available price + closing best price ─────────
    bet_odds = devig_totals(df, BET_BOOK)
    close    = devig_totals(df, CLOSE_BOOK)[
        ['date', 'home_team', 'away_team', 'odds_over', 'odds_under',
         'p_over_bk']].rename(columns={'odds_over':  'close_over',
                                       'odds_under': 'close_under',
                                       'p_over_bk':  'p_over_close'})
    for frame in (bet_odds, close):
        frame['date'] = pd.to_datetime(frame['date'])

    merged = (results
              .merge(bet_odds, on=['date', 'home_team', 'away_team'], how='inner')
              .merge(close,    on=['date', 'home_team', 'away_team'], how='left'))
    print(f"Merged with totals odds: {len(merged)} matches "
          f"(mean overround {merged['overround'].mean():.4f})\n")

    y = merged['over_actual'].to_numpy()

    # ── Model vs market accuracy ──────────────────────────────────
    print("=" * 64)
    print(f"   OVER/UNDER 2.5 — MODEL vs MARKET ({len(merged)} matches)")
    print("=" * 64)
    rows = []
    for name, p in [("Dixon-Coles",            merged['p_over'].to_numpy()),
                    (f"Market ({BET_BOOK})",   merged['p_over_bk'].to_numpy()),
                    (f"Market (closing)",      merged['p_over_close'].to_numpy())]:
        ok = np.isfinite(p)
        rows.append({
            "model":    name,
            "n":        int(ok.sum()),
            "accuracy": round(float(((p[ok] > 0.5).astype(int) == y[ok]).mean()), 4),
            "brier":    round(brier(p[ok], y[ok]), 4),
            "log_loss": round(log_loss_binary(p[ok], y[ok]), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  actual over-2.5 rate: {y.mean():.1%}   "
          f"model mean P(over): {merged['p_over'].mean():.1%}")

    # Significance: model vs closing line (paired on squared error)
    p_m = merged['p_over'].to_numpy()
    p_c = merged['p_over_close'].to_numpy()
    ok  = np.isfinite(p_m) & np.isfinite(p_c)
    bs  = paired_bootstrap((p_m[ok] - y[ok]) ** 2, (p_c[ok] - y[ok]) ** 2)
    verdict = ("model SIGNIFICANTLY better" if bs['ci_hi'] < 0 else
               "closing line SIGNIFICANTLY better" if bs['ci_lo'] > 0 else
               "no significant difference")
    print(f"\n  Brier diff vs closing = {bs['mean_diff']:+.5f}  "
          f"95% CI [{bs['ci_lo']:+.5f}, {bs['ci_hi']:+.5f}]  -> {verdict}")

    # ── Value betting on totals ───────────────────────────────────
    print("\n" + "=" * 64)
    print(f"   TOTALS BETTING — bet {BET_BOOK}, grade vs {CLOSE_BOOK}")
    print("=" * 64)
    bets = find_totals_bets(merged,
                            min_edge=MIN_EDGE, max_edge=MAX_EDGE,
                            min_odds=MIN_ODDS, max_odds=MAX_ODDS,
                            stake=STAKE, bank=BANK)
    summarise_totals_bets(bets, start_bank=BANK)

    if not bets.empty:
        print("\n  Per season:")
        per = (bets.groupby('season')
               .agg(bets=('profit', 'count'), staked=('stake', 'sum'),
                    profit=('profit', 'sum'), win_rate=('won', 'mean'),
                    mean_clv=('clv', 'mean'))
               .assign(roi=lambda d: d['profit'] / d['staked'])
               .round(4))
        print(per.to_string())
        bets.to_csv('outputs/totals_bets.csv', index=False)
        print("\nBets saved to outputs/totals_bets.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
