# main_cv.py
# ─────────────────────────────────────────────────────────────────
# Walk-forward cross-validation of every model on the same folds.
#
# Replaces the single 380-match split (which cannot resolve a 0.005
# RPS difference) with one refit per test season over 2019-20..2025-26,
# pooled to ~2,660 test matches and tested with a paired bootstrap.
#
# Training history goes back to 2014-15 (Understat's first season).
# Folds start at 2019-20 so every fold has market odds to compare to.
#
# Run:  python main_cv.py
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from main import load_seasons
from data.understat import load_understat_xg, merge_xg
from features.rolling_form import add_rolling_xg_features, ROLLING_FEATURES
from features.context import add_context_features, CONTEXT_FEATURES
from evaluation.walkforward import run_walkforward, paired_bootstrap
from evaluation.bookmaker import BookmakerComparison
from main_stack import dc_out_of_fold, DC_FEATURES
from config import SEASONS_PL_FULL, CLOSING_BOOKMAKER

ROLLING_WINDOW = 5
FIRST_TEST_SEASON = "2019-20"      # first season with Max*/closing odds


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Attach devigged closing-line probabilities as mkt_* columns."""
    bk = BookmakerComparison(margin_method="basic", bookmaker=CLOSING_BOOKMAKER)
    odds = bk.extract_odds(df)[["date", "home_team", "away_team",
                                "bk_home", "bk_draw", "bk_away"]]
    odds = odds.rename(columns={"date": "Date",
                                "home_team": "HomeTeam",
                                "away_team": "AwayTeam",
                                "bk_home": "mkt_home",
                                "bk_draw": "mkt_draw",
                                "bk_away": "mkt_away"})
    out = df.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    print(f"Market probs on {out['mkt_home'].notna().sum()}/{len(out)} matches")
    return out


def main():
    # ── Build the full feature frame ──────────────────────────────
    df = load_seasons(SEASONS_PL_FULL)
    xg = load_understat_xg(SEASONS_PL_FULL)
    df = merge_xg(df, xg)
    df = add_rolling_xg_features(df, n=ROLLING_WINDOW)
    df = add_context_features(df)
    df = add_market_probs(df)

    print("\nGenerating out-of-fold Dixon-Coles base features "
          "(walk-forward over full history)...\n")
    # Coarser retrain interval: over 12 seasons the number of refits, not
    # the fit cost, dominates runtime.
    dc = dc_out_of_fold(df, retrain_every=20)
    df = df.merge(dc, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    print(f"\nDC features on {df['dc_home'].notna().sum()}/{len(df)} matches")

    seasons_order = list(SEASONS_PL_FULL)
    test_seasons  = [s for s in seasons_order
                     if s >= FIRST_TEST_SEASON and s in set(df["season"])]

    specs = {
        "Stacked + context":     {"kind": "xgb",
                                  "features": ROLLING_FEATURES + DC_FEATURES
                                              + CONTEXT_FEATURES},
        "Stacked (DC + xG/XGB)": {"kind": "xgb",
                                  "features": ROLLING_FEATURES + DC_FEATURES},
        "XGBoost (xG form)":     {"kind": "xgb", "features": ROLLING_FEATURES},
        "Dixon-Coles":           {"kind": "columns",
                                  "cols": ["dc_away", "dc_draw", "dc_home"]},
        "Market (closing)":      {"kind": "columns",
                                  "cols": ["mkt_away", "mkt_draw", "mkt_home"]},
    }

    print(f"\nWalk-forward folds: {', '.join(test_seasons)}\n")
    res = run_walkforward(df, specs, seasons_order, test_seasons)

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(f"   WALK-FORWARD CV — {len(test_seasons)} folds, "
          f"{res['pooled']['n'].iloc[0]} matches")
    print("=" * 66)
    print(res["pooled"].to_string(index=False))

    print("\nPer-season RPS (lower = better):")
    pivot = (res["per_fold"]
             .pivot(index="season", columns="model", values="rps")
             .round(4))
    print(pivot.to_string())

    # ── Significance: everything vs Dixon-Coles ───────────────────
    print("\n" + "=" * 66)
    print("   PAIRED BOOTSTRAP vs Dixon-Coles  (10,000 resamples)")
    print("   negative diff = model beats DC")
    print("=" * 66)
    base = res["losses"]["Dixon-Coles"]
    for name, loss in res["losses"].items():
        if name == "Dixon-Coles":
            continue
        b = paired_bootstrap(loss, base)
        # Significance runs both ways: a CI entirely above zero means DC
        # significantly beats this model, not "no difference".
        if b["ci_hi"] < 0:
            verdict = "SIGNIFICANTLY BETTER than DC"
        elif b["ci_lo"] > 0:
            verdict = "SIGNIFICANTLY WORSE than DC"
        else:
            verdict = "no significant difference"
        print(f"  {name:24s} diff={b['mean_diff']:+.5f}  "
              f"95% CI [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}]  "
              f"P(better)={b['p_a_better']:.1%}  {verdict}")

    # Stacked vs pure xG form
    print("\n  Stacked vs XGBoost(xG form):")
    b = paired_bootstrap(res["losses"]["Stacked (DC + xG/XGB)"],
                         res["losses"]["XGBoost (xG form)"])
    print(f"    diff={b['mean_diff']:+.5f}  "
          f"95% CI [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}]  "
          f"P(stacked better)={b['p_a_better']:.1%}")
    print("=" * 66)

    res["per_fold"].to_csv("outputs/cv_per_fold.csv", index=False)
    res["pooled"].to_csv("outputs/cv_pooled.csv", index=False)
    print("\nSaved outputs/cv_per_fold.csv and outputs/cv_pooled.csv")


if __name__ == "__main__":
    main()
