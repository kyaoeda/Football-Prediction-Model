# main_xgb.py
# ─────────────────────────────────────────────────────────────────
# xG-form XGBoost pipeline.
#
#   1. Load football-data.co.uk PL results + Understat xG
#   2. Build leakage-safe rolling xG form features
#   3. Train XGBoost on all seasons before TEST_SEASON
#   4. Predict TEST_SEASON and compare vs Dixon-Coles and the market
#
# Run:  python main_xgb.py
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from main import load_seasons
from data.understat import load_understat_xg, merge_xg
from features.rolling_form import add_rolling_xg_features, ROLLING_FEATURES
from models.xgb_model import XGBResultModel, result_target
from models.dixon_coles import DixonColesModel
from evaluation.bookmaker import BookmakerComparison
from config import SEASONS, TEST_SEASON, DIXON_COLES_XI, CLOSING_BOOKMAKER

ROLLING_WINDOW = 5


# ── Metrics ───────────────────────────────────────────────────────

def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Ranked Probability Score. probs cols = [away, draw, home]."""
    n = len(outcomes)
    actual = np.zeros((n, 3))
    actual[np.arange(n), outcomes] = 1.0
    cum_p = np.cumsum(probs,  axis=1)
    cum_a = np.cumsum(actual, axis=1)
    return float(np.mean(np.sum((cum_p - cum_a) ** 2, axis=1) / 2))


def evaluate(name: str, probs: np.ndarray, outcomes: np.ndarray) -> dict:
    preds    = probs.argmax(axis=1)
    accuracy = float((preds == outcomes).mean())
    return {
        "model":    name,
        "accuracy": round(accuracy, 4),
        "rps":      round(rps(probs, outcomes), 4),
        "log_loss": round(float(log_loss(outcomes, probs, labels=[0, 1, 2])), 4),
    }


# ── Dixon-Coles baseline on the same split ────────────────────────

def dixon_coles_probs(train: pd.DataFrame, test: pd.DataFrame,
                      xi: float = DIXON_COLES_XI) -> np.ndarray:
    """Fit DC once on train, predict test. [away, draw, home] per row."""
    model = DixonColesModel(xi=xi).fit(train)

    probs = np.empty((len(test), 3))
    for k, (_, row) in enumerate(test.iterrows()):
        h, a = row["HomeTeam"], row["AwayTeam"]
        # Seed promoted/unseen teams with a prior, mirroring the backtester.
        for team in (h, a):
            if team not in model.team_to_idx_:
                model.register_promoted_team(team)
        p = model.predict(h, a)
        probs[k] = [p.away_win, p.draw, p.home_win]
    return probs


# ── Bookmaker (closing) probs on the same split ───────────────────

def bookmaker_probs(test: pd.DataFrame) -> np.ndarray | None:
    """
    Devigged closing-line probs aligned to test rows (order preserved),
    or None if the closing odds columns are absent.

    `test` must still carry the raw football-data columns (Date,
    HomeTeam, AwayTeam, B365C*), which it does — features are added
    without dropping the source columns.
    """
    bk = BookmakerComparison(margin_method="basic", bookmaker=CLOSING_BOOKMAKER)
    try:
        odds = bk.extract_odds(test)          # -> date/home_team/away_team + bk_*
    except KeyError:
        return None

    merged = test.merge(
        odds,
        left_on  = ["Date", "HomeTeam", "AwayTeam"],
        right_on = ["date", "home_team", "away_team"],
        how      = "left",
    )
    if merged["bk_home"].isna().all():
        return None
    return merged[["bk_away", "bk_draw", "bk_home"]].to_numpy()


# ── Main ──────────────────────────────────────────────────────────

def main():
    # 1. Load results + xG, merge, build features
    df = load_seasons(SEASONS)
    xg = load_understat_xg(SEASONS)
    df = merge_xg(df, xg)
    df = add_rolling_xg_features(df, n=ROLLING_WINDOW)

    # 2. Train/test split by season
    if TEST_SEASON not in set(df["season"]):
        raise ValueError(f"TEST_SEASON '{TEST_SEASON}' not in data")

    train = df[df["season"] != TEST_SEASON].reset_index(drop=True)
    test  = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    print(f"\nTrain: {len(train)} matches (< {TEST_SEASON})  "
          f"Test: {len(test)} matches ({TEST_SEASON})")
    print(f"Features ({len(ROLLING_FEATURES)}): {', '.join(ROLLING_FEATURES)}\n")

    outcomes = result_target(test)

    # 3. XGBoost
    xgb = XGBResultModel(features=ROLLING_FEATURES).fit(train)
    xgb_probs = xgb.predict_proba(test)

    # 4. Baselines on identical test matches
    dc_probs = dixon_coles_probs(train, test)

    n = len(test)
    home_rate = (outcomes == 2).mean()
    # Naive baseline = training class frequencies, broadcast to every match
    base_freq = np.array([
        (result_target(train) == 0).mean(),
        (result_target(train) == 1).mean(),
        (result_target(train) == 2).mean(),
    ])
    base_probs = np.tile(base_freq, (n, 1))

    rows = [
        evaluate("XGBoost (xG form)", xgb_probs,  outcomes),
        evaluate("Dixon-Coles",       dc_probs,   outcomes),
        evaluate("Baseline (freq)",   base_probs, outcomes),
    ]

    # Market closing line, if available on the test rows
    bkp = bookmaker_probs(test)
    if bkp is not None:
        rows.append(evaluate(f"Market ({CLOSING_BOOKMAKER})", bkp, outcomes))

    # ── Report ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"   RESULT MODEL COMPARISON — {TEST_SEASON} ({n} matches)")
    print("=" * 60)
    summary = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    print(summary.to_string(index=False))
    print(f"\n  (home-win base rate this season: {home_rate:.1%})")
    print("=" * 60)

    print("\nTop xG-form features by gain:")
    print(xgb.feature_importance().head(8).to_string(index=False))

    # 5. Save test predictions
    out = test[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    out["p_away"], out["p_draw"], out["p_home"] = (
        np.round(xgb_probs[:, 0], 4),
        np.round(xgb_probs[:, 1], 4),
        np.round(xgb_probs[:, 2], 4),
    )
    out.to_csv("outputs/xgb_predictions.csv", index=False)
    print("\nPredictions saved to outputs/xgb_predictions.csv")


if __name__ == "__main__":
    main()
