# main_tune_xi.py
# ─────────────────────────────────────────────────────────────────
# Nested rolling-origin selection of the Dixon-Coles time-decay xi.
#
# The point of the nesting
# ------------------------
# Picking xi on the same folds you then report is selection bias: the
# reported score is no longer out-of-sample. Here, for each test season
# S the value xi* is chosen using ONLY seasons before S (an inner
# validation on the last VAL_SEASONS campaigns), and S is scored with
# that xi*. The outer scores never influenced the choice, so they stay
# unbiased — and the procedure is also what you'd have to do live.
#
# What to look at in the output
# -----------------------------
#   1. Are the per-fold xi* values STABLE? Stable optimum -> we are
#      estimating a real decay rate. Erratic -> fitting noise.
#   2. Is the validation curve FLAT? If RPS barely moves across xi,
#      tuning buys nothing regardless of significance.
#
# Both models are evaluated in the same "fit on all prior seasons,
# predict the season" mode, so the comparison isolates the effect of
# xi alone (these numbers are NOT comparable to main_cv.py, which
# refits walk-forward within each season).
#
# Run:  python main_tune_xi.py
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from main import load_seasons
from main_xgb import dixon_coles_probs
from models.xgb_model import result_target
from evaluation.walkforward import rps_per_match, paired_bootstrap
from config import SEASONS_PL_FULL, DIXON_COLES_XI

# Candidate decay rates. Half-life in days = ln(2)/xi, so this spans
# roughly 6 weeks (xi=0.015) to 3.8 years (xi=0.0005).
XI_GRID = [0.0005, 0.001, 0.002, 0.00325, 0.005, 0.0075, 0.010, 0.015]

FIRST_TEST_SEASON = "2019-20"
VAL_SEASONS       = 2      # inner validation campaigns per fold


def half_life_days(xi: float) -> float:
    return np.log(2) / xi


def _season_rps(df: pd.DataFrame, rank: dict, season: str, xi: float) -> np.ndarray:
    """Per-match RPS for one season, DC trained on all earlier seasons."""
    train = df[df["season"].map(rank) < rank[season]]
    test  = df[df["season"] == season]
    probs = dixon_coles_probs(train, test, xi=xi)
    return rps_per_match(probs, result_target(test))


def select_xi(df: pd.DataFrame, rank: dict, val_seasons: list[str]) -> tuple:
    """Choose xi on validation seasons only. Returns (xi*, curve dict)."""
    curve = {}
    for xi in XI_GRID:
        losses = [_season_rps(df, rank, vs, xi) for vs in val_seasons]
        curve[xi] = float(np.concatenate(losses).mean())
    best = min(curve, key=curve.get)
    return best, curve


def main():
    df = load_seasons(SEASONS_PL_FULL)

    seasons_order = [s for s in SEASONS_PL_FULL if s in set(df["season"])]
    rank = {s: i for i, s in enumerate(seasons_order)}
    test_seasons = [s for s in seasons_order if s >= FIRST_TEST_SEASON]

    print(f"\nSeasons: {seasons_order[0]} -> {seasons_order[-1]}  "
          f"({len(seasons_order)} total)")
    print(f"Outer test folds: {', '.join(test_seasons)}")
    print(f"Inner validation: {VAL_SEASONS} seasons before each fold")
    print(f"xi grid: {XI_GRID}\n")

    fold_rows   = []
    curves      = []
    tuned_loss  = []
    base_loss   = []

    for season in test_seasons:
        i = rank[season]
        val_seasons = seasons_order[max(0, i - VAL_SEASONS):i]
        if len(val_seasons) < VAL_SEASONS or rank[val_seasons[0]] == 0:
            print(f"  [skip] {season}: insufficient history for inner validation")
            continue

        xi_star, curve = select_xi(df, rank, val_seasons)
        curves.append(curve)

        # Outer evaluation: unbiased, xi* never saw this season
        l_tuned = _season_rps(df, rank, season, xi_star)
        l_base  = _season_rps(df, rank, season, DIXON_COLES_XI)
        tuned_loss.append(l_tuned)
        base_loss.append(l_base)

        fold_rows.append({
            "season":       season,
            "val_seasons":  "+".join(val_seasons),
            "xi_star":      xi_star,
            "half_life_d":  round(half_life_days(xi_star)),
            "rps_tuned":    round(float(l_tuned.mean()), 4),
            "rps_default":  round(float(l_base.mean()), 4),
        })
        print(f"  {season}: xi*={xi_star:<7} (half-life {half_life_days(xi_star):5.0f}d)  "
              f"rps_tuned={l_tuned.mean():.4f}  rps_default={l_base.mean():.4f}")

    folds = pd.DataFrame(fold_rows)

    # ── 1. Stability of the selected xi ───────────────────────────
    print("\n" + "=" * 68)
    print("   PER-FOLD SELECTION (is xi* stable?)")
    print("=" * 68)
    print(folds.to_string(index=False))

    chosen = folds["xi_star"].to_numpy()
    uniq   = sorted(set(chosen))
    print(f"\n  distinct xi* chosen : {uniq}")
    print(f"  mode                : {pd.Series(chosen).mode().iloc[0]}")
    print(f"  spread (min -> max) : {chosen.min()} -> {chosen.max()} "
          f"({chosen.max() / chosen.min():.1f}x)")

    # ── 2. Flatness of the validation curve ───────────────────────
    print("\n" + "=" * 68)
    print("   VALIDATION CURVE (mean inner RPS across folds)")
    print("=" * 68)
    mean_curve = {xi: float(np.mean([c[xi] for c in curves])) for xi in XI_GRID}
    best_xi    = min(mean_curve, key=mean_curve.get)
    worst      = max(mean_curve.values())
    best_val   = mean_curve[best_xi]
    for xi in XI_GRID:
        bar  = "#" * int(round((mean_curve[xi] - best_val) * 20000))
        mark = "  <- best" if xi == best_xi else ""
        print(f"  xi={xi:<7} half-life={half_life_days(xi):6.0f}d  "
              f"RPS={mean_curve[xi]:.4f}  {bar}{mark}")
    print(f"\n  curve range (worst - best) = {worst - best_val:.4f} RPS")

    # ── 3. Did tuning actually help, measured honestly? ───────────
    t = np.concatenate(tuned_loss)
    b = np.concatenate(base_loss)
    print("\n" + "=" * 68)
    print("   OUT-OF-SAMPLE EFFECT OF TUNING (nested, unbiased)")
    print("=" * 68)
    print(f"  Default xi={DIXON_COLES_XI}: RPS {b.mean():.4f}")
    print(f"  Nested-tuned xi        : RPS {t.mean():.4f}")

    bs = paired_bootstrap(t, b)
    if bs["ci_hi"] < 0:
        verdict = "tuning SIGNIFICANTLY helps"
    elif bs["ci_lo"] > 0:
        verdict = "tuning SIGNIFICANTLY hurts"
    else:
        verdict = "no significant difference"
    print(f"  diff = {bs['mean_diff']:+.5f}  "
          f"95% CI [{bs['ci_lo']:+.5f}, {bs['ci_hi']:+.5f}]")
    print(f"  P(tuned better) = {bs['p_a_better']:.1%}   -> {verdict}")
    print("=" * 68)

    folds.to_csv("outputs/xi_tuning_folds.csv", index=False)
    pd.DataFrame([{"xi": k, "val_rps": v} for k, v in mean_curve.items()]) \
        .to_csv("outputs/xi_tuning_curve.csv", index=False)
    print("\nSaved outputs/xi_tuning_folds.csv and outputs/xi_tuning_curve.csv")


if __name__ == "__main__":
    main()
