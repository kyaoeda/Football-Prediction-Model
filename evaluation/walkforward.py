# evaluation/walkforward.py
# ─────────────────────────────────────────────────────────────────
# Expanding-window, season-by-season cross-validation.
#
# A single train/test split on one season (380 matches) cannot
# distinguish a 0.005 RPS difference from noise. This module refits
# each model once per test season, always training only on EARLIER
# seasons, then pools the per-match scores across folds so model
# differences can be tested with a paired bootstrap.
#
# Model specs
# -----------
#   {'kind': 'xgb',     'features': [...]}  -> refit each fold
#   {'kind': 'columns', 'cols': [away, draw, home]} -> precomputed
#       probabilities already in the frame (e.g. out-of-fold Dixon-Coles
#       or devigged market odds); read, never refit.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from models.xgb_model import XGBResultModel, result_target


# ── Metrics ───────────────────────────────────────────────────────

def rps_per_match(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match Ranked Probability Score. probs cols = [away, draw, home]."""
    n = len(outcomes)
    actual = np.zeros((n, 3))
    actual[np.arange(n), outcomes] = 1.0
    cum_p = np.cumsum(probs,  axis=1)
    cum_a = np.cumsum(actual, axis=1)
    return np.sum((cum_p - cum_a) ** 2, axis=1) / 2


def paired_bootstrap(loss_a: np.ndarray,
                     loss_b: np.ndarray,
                     n_boot: int = 10000,
                     seed: int = 42) -> dict:
    """
    Paired bootstrap on two per-match loss arrays (lower = better).

    Resamples matches with replacement and recomputes the mean
    difference (a - b), so a negative difference means model A is
    better. Returns the observed difference, a 95% interval, and the
    share of resamples in which A wins.
    """
    rng  = np.random.default_rng(seed)
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    n    = len(diff)

    idx    = rng.integers(0, n, size=(n_boot, n))
    means  = diff[idx].mean(axis=1)

    return {
        "mean_diff":  float(diff.mean()),
        "ci_lo":      float(np.percentile(means, 2.5)),
        "ci_hi":      float(np.percentile(means, 97.5)),
        "p_a_better": float((means < 0).mean()),
    }


# ── Walk-forward driver ───────────────────────────────────────────

def _fold_probs(spec: dict, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Produce [away, draw, home] probabilities for one fold."""
    if spec["kind"] == "xgb":
        model = XGBResultModel(features=spec["features"],
                               **spec.get("params", {}))
        model.fit(train)
        return model.predict_proba(test)

    if spec["kind"] == "columns":
        return test[spec["cols"]].to_numpy(dtype=float)

    raise ValueError(f"Unknown spec kind: {spec['kind']}")


def run_walkforward(df: pd.DataFrame,
                    specs: dict,
                    seasons_order: list[str],
                    test_seasons: list[str],
                    season_col: str = "season") -> dict:
    """
    Refit and score each spec once per test season, training only on
    earlier seasons.

    Returns
    -------
    dict with:
        'per_fold' : DataFrame — metrics per (season, model)
        'pooled'   : DataFrame — metrics pooled over all folds
        'losses'   : dict[name] -> per-match RPS array (pooled, aligned)
        'outcomes' : pooled outcome array
    """
    rank = {s: i for i, s in enumerate(seasons_order)}

    fold_rows   = []
    collected   = {name: [] for name in specs}
    outcomes_all = []

    for season in test_seasons:
        train = df[df[season_col].map(rank) < rank[season]]
        test  = df[df[season_col] == season]
        if train.empty or test.empty:
            print(f"  [skip] {season}: train={len(train)} test={len(test)}")
            continue

        y = result_target(test)
        outcomes_all.append(y)
        print(f"  fold {season}: train={len(train):5d}  test={len(test):4d}")

        for name, spec in specs.items():
            probs = _fold_probs(spec, train, test)
            collected[name].append(probs)

            # Per-fold metrics ignore rows this model cannot score
            ok = np.isfinite(probs).all(axis=1)
            if ok.sum() == 0:
                fold_rows.append({"season": season, "model": name,
                                  "n": 0, "accuracy": np.nan,
                                  "rps": np.nan, "log_loss": np.nan})
                continue

            p, yy = probs[ok], y[ok]
            fold_rows.append({
                "season":   season,
                "model":    name,
                "n":        int(ok.sum()),
                "accuracy": float((p.argmax(axis=1) == yy).mean()),
                "rps":      float(rps_per_match(p, yy).mean()),
                "log_loss": float(log_loss(yy, p / p.sum(axis=1, keepdims=True),
                                           labels=[0, 1, 2])),
            })

    # ── Pool across folds ─────────────────────────────────────────
    outcomes = np.concatenate(outcomes_all)
    pooled_probs = {n: np.vstack(v) for n, v in collected.items()}

    # Compare only on matches every model could score
    valid = np.ones(len(outcomes), dtype=bool)
    for p in pooled_probs.values():
        valid &= np.isfinite(p).all(axis=1)

    losses = {}
    pooled_rows = []
    for name, p in pooled_probs.items():
        pv = p[valid]
        pv = pv / pv.sum(axis=1, keepdims=True)     # guard against drift
        yv = outcomes[valid]
        losses[name] = rps_per_match(pv, yv)
        pooled_rows.append({
            "model":    name,
            "n":        int(valid.sum()),
            "accuracy": round(float((pv.argmax(axis=1) == yv).mean()), 4),
            "rps":      round(float(losses[name].mean()), 4),
            "log_loss": round(float(log_loss(yv, pv, labels=[0, 1, 2])), 4),
        })

    return {
        "per_fold": pd.DataFrame(fold_rows),
        "pooled":   pd.DataFrame(pooled_rows).sort_values("rps").reset_index(drop=True),
        "losses":   losses,
        "outcomes": outcomes[valid],
    }
