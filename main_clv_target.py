# main_clv_target.py
# ─────────────────────────────────────────────────────────────────
# Bucket 3: predict the CLOSING LINE instead of the match outcome.
#
# Why change target
# -----------------
# A match result is close to irreducibly random — the best predictor
# on earth manages RPS 0.197. A closing price is a market consensus:
# far more predictable, continuous, much better signal-to-noise. And
# it is the quantity that actually decides whether you have an edge,
# because beating the close is what CLV measures.
#
# So instead of P(home win) we predict the MOVEMENT
#       delta = P_close - P_open
# for each outcome, then bet the outcomes predicted to shorten (the
# market is expected to move toward them, meaning the opening price
# was generous).
#
# The benchmark that matters is the naive "the line will not move"
# predictor (delta = 0). Beating that is the whole question; if we
# cannot, there is no line-movement signal to trade.
#
# Central hypothesis: when Dixon-Coles disagrees with the opening
# price, does the close drift toward DC?
#
# Run:  python main_clv_target.py
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from main import load_seasons
from data.understat import load_understat_xg, merge_xg
from features.rolling_form import add_rolling_xg_features, ROLLING_FEATURES
from features.context import add_context_features, CONTEXT_FEATURES
from evaluation.bookmaker import BookmakerComparison
from evaluation.walkforward import paired_bootstrap
from main_stack import dc_out_of_fold, DC_FEATURES
from config import SEASONS_PL_FULL

FIRST_TEST_SEASON = "2019-20"     # first season with closing odds
OPEN_BOOK         = "bet365"      # same book both ends: movement is real repricing
CLOSE_BOOK        = "bet365_closing"
DC_CACHE          = "outputs/dc_oof_cache.csv"
OUTCOMES          = ["home", "draw", "away"]
BET_THRESHOLD     = 0.01          # predicted shortening of >=1 percentage point


def cached_dc(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward DC features, cached (the pass is the slow part)."""
    if os.path.exists(DC_CACHE):
        print(f"Loading cached DC features from {DC_CACHE}")
        dc = pd.read_csv(DC_CACHE, parse_dates=["Date"])
    else:
        print("\nGenerating out-of-fold Dixon-Coles features...\n")
        dc = dc_out_of_fold(df, retrain_every=20)
        dc.to_csv(DC_CACHE, index=False)
        print(f"Cached to {DC_CACHE}")
    return dc


def attach_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Attach devigged opening and closing probabilities plus raw odds."""
    def grab(book, tag):
        bk = BookmakerComparison(margin_method="basic", bookmaker=book)
        o  = bk.extract_odds(df)
        return o.rename(columns={
            "date": "Date", "home_team": "HomeTeam", "away_team": "AwayTeam",
            "bk_home": f"p_{tag}_home", "bk_draw": f"p_{tag}_draw",
            "bk_away": f"p_{tag}_away",
            "odds_home": f"odds_{tag}_home", "odds_draw": f"odds_{tag}_draw",
            "odds_away": f"odds_{tag}_away",
        })[["Date", "HomeTeam", "AwayTeam",
            f"p_{tag}_home", f"p_{tag}_draw", f"p_{tag}_away",
            f"odds_{tag}_home", f"odds_{tag}_draw", f"odds_{tag}_away"]]

    df = df.merge(grab(OPEN_BOOK, "open"),   on=["Date", "HomeTeam", "AwayTeam"], how="left")
    df = df.merge(grab(CLOSE_BOOK, "close"), on=["Date", "HomeTeam", "AwayTeam"], how="left")
    return df


def main():
    # ── Build frame ───────────────────────────────────────────────
    df = load_seasons(SEASONS_PL_FULL)
    xg = load_understat_xg(SEASONS_PL_FULL)
    df = merge_xg(df, xg)
    df = add_rolling_xg_features(df)
    df = add_context_features(df)
    df = attach_prices(df)

    dc = cached_dc(df)
    df = df.merge(dc, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # Need both prices and a DC view
    need = ([f"p_open_{o}" for o in OUTCOMES] +
            [f"p_close_{o}" for o in OUTCOMES] + ["dc_home"])
    df = df.dropna(subset=need).reset_index(drop=True)

    # ── Targets and DC-disagreement features ──────────────────────
    dc_col = {"home": "dc_home", "draw": "dc_draw", "away": "dc_away"}
    for o in OUTCOMES:
        df[f"delta_{o}"]    = df[f"p_close_{o}"] - df[f"p_open_{o}"]
        # The core hypothesis: DC minus the opening price
        df[f"dc_disagree_{o}"] = df[dc_col[o]] - df[f"p_open_{o}"]

    feats = (ROLLING_FEATURES + DC_FEATURES + CONTEXT_FEATURES +
             [f"p_open_{o}" for o in OUTCOMES] +
             [f"dc_disagree_{o}" for o in OUTCOMES])

    seasons_order = [s for s in SEASONS_PL_FULL if s in set(df["season"])]
    rank = {s: i for i, s in enumerate(seasons_order)}
    test_seasons = [s for s in seasons_order if s >= FIRST_TEST_SEASON]

    print(f"\nRows with open+close prices: {len(df)}")
    print(f"Folds: {', '.join(test_seasons)}   |  {len(feats)} features\n")

    # ── Walk-forward: predict the movement ────────────────────────
    preds = {o: [] for o in OUTCOMES}
    actual = {o: [] for o in OUTCOMES}
    keep = []

    for season in test_seasons:
        train = df[df["season"].map(rank) < rank[season]]
        test  = df[df["season"] == season]
        if len(train) < 200:
            print(f"  [skip] {season}: only {len(train)} training rows")
            continue
        print(f"  fold {season}: train={len(train):5d} test={len(test):4d}")
        keep.append(test.index.to_numpy())

        for o in OUTCOMES:
            model = XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
                reg_lambda=1.0, random_state=42, n_jobs=-1,
            )
            model.fit(train[feats], train[f"delta_{o}"])
            preds[o].append(model.predict(test[feats]))
            actual[o].append(test[f"delta_{o}"].to_numpy())

    idx = np.concatenate(keep)
    pooled = df.loc[idx].reset_index(drop=True)
    for o in OUTCOMES:
        pooled[f"pred_delta_{o}"] = np.concatenate(preds[o])
        pooled[f"true_delta_{o}"] = np.concatenate(actual[o])

    # ── Q1: can we predict line movement at all? ──────────────────
    print("\n" + "=" * 68)
    print(f"   PREDICTING LINE MOVEMENT ({len(pooled)} matches)")
    print("   benchmark = naive 'line will not move' (delta = 0)")
    print("=" * 68)
    rows = []
    for o in OUTCOMES:
        p = pooled[f"pred_delta_{o}"].to_numpy()
        t = pooled[f"true_delta_{o}"].to_numpy()
        rows.append({
            "outcome":     o,
            "MAE_model":   round(float(np.abs(p - t).mean()), 5),
            "MAE_naive":   round(float(np.abs(t).mean()), 5),
            "corr":        round(float(np.corrcoef(p, t)[0, 1]), 4),
            "mean_move":   round(float(np.abs(t).mean()), 5),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  Significance (model vs naive, paired bootstrap on |error|):")
    for o in OUTCOMES:
        p = pooled[f"pred_delta_{o}"].to_numpy()
        t = pooled[f"true_delta_{o}"].to_numpy()
        b = paired_bootstrap(np.abs(p - t), np.abs(t))
        verdict = ("model BETTER" if b["ci_hi"] < 0 else
                   "model WORSE"  if b["ci_lo"] > 0 else "no difference")
        print(f"    {o:5s} diff={b['mean_diff']:+.5f} "
              f"CI [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}]  {verdict}")

    # ── Q2: does betting the predicted movement earn CLV? ─────────
    print("\n" + "=" * 68)
    print(f"   BETTING PREDICTED MOVEMENT (threshold {BET_THRESHOLD:+.1%})")
    print("=" * 68)
    # Actual result from the scoreline (the frame carries FTHG/FTAG, not
    # the backtester's encoded 'outcome' column).
    pooled["result"] = np.where(pooled["FTHG"] > pooled["FTAG"], "home",
                        np.where(pooled["FTHG"] == pooled["FTAG"], "draw", "away"))

    bets = []
    for _, r in pooled.iterrows():
        for o in OUTCOMES:
            if r[f"pred_delta_{o}"] < BET_THRESHOLD:
                continue
            open_odds, close_odds = r[f"odds_open_{o}"], r[f"odds_close_{o}"]
            if not (np.isfinite(open_odds) and np.isfinite(close_odds)):
                continue
            won = (r["result"] == o)
            bets.append({
                "season": r["season"], "outcome_bet": o,
                "clv":    open_odds / close_odds - 1,
                "profit": (open_odds - 1) * 10 if won else -10,
                "won":    won,
            })
    bets = pd.DataFrame(bets)

    if bets.empty:
        print("  No bets cleared the threshold.")
    else:
        clv = bets["clv"]
        print(f"  Bets              : {len(bets)}")
        print(f"  Mean CLV          : {clv.mean():+.2%}")
        print(f"  % beating close   : {(clv > 0).mean():.1%}")
        print(f"  Win rate          : {bets['won'].mean():.1%}")
        print(f"  ROI               : {bets['profit'].sum() / (10 * len(bets)):+.2%}")
        print("\n  Per season:")
        per = (bets.groupby("season")
               .agg(bets=("clv", "count"), mean_clv=("clv", "mean"),
                    pct_beat=("clv", lambda s: (s > 0).mean()),
                    roi=("profit", lambda s: s.sum() / (10 * len(s))))
               .round(4))
        print(per.to_string())
        bets.to_csv("outputs/clv_target_bets.csv", index=False)
        print("\nSaved outputs/clv_target_bets.csv")
    print("=" * 68)


if __name__ == "__main__":
    main()
