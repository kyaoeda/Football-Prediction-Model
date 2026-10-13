# main_stack.py
# ─────────────────────────────────────────────────────────────────
# Stacked model: Dixon-Coles base layer + XGBoost meta-layer.
#
#   Base   : walk-forward Dixon-Coles -> out-of-fold 1X2 probs + xG
#   Meta   : XGBoost trained on [rolling xG form  +  DC base probs]
#
# Why out-of-fold DC probs?
#   Each match's DC feature is produced by a model trained only on
#   EARLIER matches (the walk-forward backtester). If we instead fit
#   DC once and predicted the training rows in-sample, XGBoost would
#   learn from over-confident probs it never sees at test time. The
#   walk-forward pass makes train and test DC features identically
#   distributed and leakage-free.
#
# Run:  python main_stack.py
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from main import load_seasons
from data.understat import load_understat_xg, merge_xg
from features.rolling_form import add_rolling_xg_features, ROLLING_FEATURES
from models.xgb_model import XGBResultModel, result_target
from evaluation.backtester import Backtester
from main_xgb import rps, evaluate, bookmaker_probs
from config import SEASONS, TEST_SEASON, DIXON_COLES_XI, CLOSING_BOOKMAKER

ROLLING_WINDOW   = 5
DC_MIN_TRAIN     = 380    # 1 season warm-up before DC starts predicting
DC_RETRAIN_EVERY = 10

DC_FEATURES = ["dc_home", "dc_draw", "dc_away", "dc_exp_home", "dc_exp_away"]


def dc_out_of_fold(df: pd.DataFrame,
                   min_train: int = DC_MIN_TRAIN,
                   retrain_every: int = DC_RETRAIN_EVERY) -> pd.DataFrame:
    """
    Walk-forward Dixon-Coles over the full dataset, returning one
    out-of-fold prediction per match (keyed by Date/HomeTeam/AwayTeam).
    Matches inside the warm-up window get no DC prediction -> NaN later.

    retrain_every trades accuracy for speed: over a long history the
    number of refits dominates runtime, so CV runs use a coarser value.
    """
    bt  = Backtester(xi=DIXON_COLES_XI,
                     min_train=min_train,
                     retrain_every=retrain_every)
    res = bt.run(df)

    dc = res[["date", "home_team", "away_team",
              "home_win", "draw", "away_win",
              "exp_home", "exp_away"]].rename(columns={
        "date": "Date", "home_team": "HomeTeam", "away_team": "AwayTeam",
        "home_win": "dc_home", "draw": "dc_draw", "away_win": "dc_away",
        "exp_home": "dc_exp_home", "exp_away": "dc_exp_away"})
    dc["Date"] = pd.to_datetime(dc["Date"])
    return dc


def main():
    # 1. Results + xG + rolling form
    df = load_seasons(SEASONS)
    xg = load_understat_xg(SEASONS)
    df = merge_xg(df, xg)
    df = add_rolling_xg_features(df, n=ROLLING_WINDOW)

    # 2. Base layer: out-of-fold Dixon-Coles probs as features
    print("\nGenerating out-of-fold Dixon-Coles base features...\n")
    dc = dc_out_of_fold(df)
    df = df.merge(dc, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    dc_cov = df["dc_home"].notna().sum()
    print(f"\nDC base features present on {dc_cov}/{len(df)} matches "
          f"(first {DC_MIN_TRAIN} are warm-up).")

    # 3. Train/test split by season
    train = df[df["season"] != TEST_SEASON].reset_index(drop=True)
    test  = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    outcomes = result_target(test)

    stack_features = ROLLING_FEATURES + DC_FEATURES
    print(f"\nTrain: {len(train)}  Test: {len(test)}  "
          f"({TEST_SEASON})   |  {len(stack_features)} features\n")

    # 4. Models
    #    a) XGBoost on xG form only
    xgb_form   = XGBResultModel(features=ROLLING_FEATURES).fit(train)
    form_probs = xgb_form.predict_proba(test)

    #    b) Stacked: XGBoost on xG form + DC base probs
    xgb_stack   = XGBResultModel(features=stack_features).fit(train)
    stack_probs = xgb_stack.predict_proba(test)

    #    c) Dixon-Coles alone = its out-of-fold probs on the test rows
    dc_probs = test[["dc_away", "dc_draw", "dc_home"]].to_numpy()

    rows = [
        evaluate("Stacked (DC + xG/XGB)", stack_probs, outcomes),
        evaluate("XGBoost (xG form)",     form_probs,  outcomes),
        evaluate("Dixon-Coles (walk-fwd)", dc_probs,   outcomes),
    ]

    bkp = bookmaker_probs(test)
    if bkp is not None:
        rows.append(evaluate(f"Market ({CLOSING_BOOKMAKER})", bkp, outcomes))

    # 5. Report
    print("=" * 62)
    print(f"   STACKED MODEL COMPARISON — {TEST_SEASON} ({len(test)} matches)")
    print("=" * 62)
    summary = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    print(summary.to_string(index=False))
    print("=" * 62)

    print("\nTop stacked features by gain:")
    print(xgb_stack.feature_importance().head(8).to_string(index=False))

    out = test[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    out["p_away"], out["p_draw"], out["p_home"] = (
        np.round(stack_probs[:, 0], 4),
        np.round(stack_probs[:, 1], 4),
        np.round(stack_probs[:, 2], 4),
    )
    out.to_csv("outputs/stack_predictions.csv", index=False)
    print("\nPredictions saved to outputs/stack_predictions.csv")


if __name__ == "__main__":
    main()
