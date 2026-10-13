# evaluation/backtester.py
# ─────────────────────────────────────────────────────────────────
# Walk-forward backtester for the Dixon-Coles model.
#
# Strategy:
#   - Train on all matches up to date T
#   - Predict match at date T+1
#   - Roll forward one match at a time
#   - Evaluate predictions against actual results
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from models.dixon_coles import DixonColesModel
from config import (
    DIXON_COLES_XI,
    DIXON_COLES_MAX_GOALS,
    PROMOTED_RATING_METHOD,
    PROMOTED_N_WEAKEST,
)


class Backtester:
    """
    Walk-forward backtester for football prediction models.

    Rather than retraining on every single match (very slow),
    we retrain every N matches — a practical compromise that
    still prevents data leakage.

    Parameters
    ----------
    xi          : float — time decay for Dixon-Coles
    min_train   : int   — minimum matches before first prediction
                          (need enough data to estimate parameters)
    retrain_every : int — retrain model every N matches
                          1   = most accurate, very slow
                          10  = good balance ✅
                          38  = retrain once per matchweek
    handle_promoted : bool — if True, a team appearing for the first time
                          (i.e. promoted into the division) is seeded with a
                          prior rating so its match is predicted rather than
                          skipped. If False, such matches are skipped (old
                          behaviour).
    promoted_method : str — prior method passed to the model
                          ('weakest' | 'average' | 'percentile')
    n_weakest       : int — teams averaged when promoted_method='weakest'

    Example
    -------
    >>> bt = Backtester(min_train=380, retrain_every=10)
    >>> results = bt.run(df)
    >>> bt.summary(results)
    """

    def __init__(self,
                 xi              : float = DIXON_COLES_XI,
                 min_train       : int   = 380,
                 retrain_every   : int   = 10,
                 handle_promoted : bool  = True,
                 promoted_method : str   = PROMOTED_RATING_METHOD,
                 n_weakest       : int   = PROMOTED_N_WEAKEST,
                 totals_line     : float = 2.5):

        self.xi              = xi
        self.min_train       = min_train
        self.retrain_every   = retrain_every
        self.handle_promoted = handle_promoted
        self.promoted_method = promoted_method
        self.n_weakest       = n_weakest
        self.totals_line     = totals_line

    # ── Private Helpers ───────────────────────────────────────────

    def _get_result(self, row: pd.Series) -> int:
        """
        Convert scoreline to outcome integer.
        2 = home win, 1 = draw, 0 = away win

        We use 2/1/0 so that numpy indexing into
        probability arrays works cleanly.
        """
        if   row['FTHG'] > row['FTAG']: return 2   # home win
        elif row['FTHG'] == row['FTAG']: return 1   # draw
        else                           : return 0   # away win

    def _rps(self, probs: np.ndarray,
              outcome: int) -> float:
        """
        Ranked Probability Score for a single match.

        Lower is better.
        Perfect prediction = 0.0
        Random prediction  ≈ 0.333

        Parameters
        ----------
        probs   : array [P(away), P(draw), P(home)]
        outcome : int — 0=away, 1=draw, 2=home
        """
        # One-hot encode actual outcome
        actual     = np.zeros(3)
        actual[outcome] = 1.0

        # Cumulative probabilities
        cum_pred   = np.cumsum(probs)
        cum_actual = np.cumsum(actual)

        return float(np.sum((cum_pred - cum_actual) ** 2) / 2)

    # ── Main Method ───────────────────────────────────────────────

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the walk-forward backtest.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset sorted by date. Must contain:
            HomeTeam, AwayTeam, FTHG, FTAG, Date

        Returns
        -------
        pd.DataFrame — one row per predicted match with columns:
            Date, HomeTeam, AwayTeam,
            home_win, draw, away_win,   ← model probabilities
            outcome,                    ← actual result
            rps,                        ← ranked probability score
            log_loss                    ← log loss per match
        """
        data   = data.reset_index(drop=True)
        n      = len(data)
        results = []

        promoted_seen    = set()   # promoted teams given a prior rating
        promoted_matches = 0       # matches predicted via a prior
        skipped_matches  = 0       # matches skipped (promoted handling off)

        print(f"Backtesting {n - self.min_train} matches...")
        print(f"Training window starts at: {self.min_train} matches")
        print(f"Retraining every: {self.retrain_every} matches\n")

        model        = None
        last_trained = None

        for i in range(self.min_train, n):

            # ── Retrain if needed ──────────────────────────────────
            should_retrain = (
                model is None or
                (i - self.min_train) % self.retrain_every == 0
            )

            if should_retrain:
                train_data = data.iloc[:i]
                model      = DixonColesModel(
                    xi        = self.xi,
                    max_goals = DIXON_COLES_MAX_GOALS
                )
                model.fit(train_data)
                last_trained = i

                # Progress update every 100 matches
                if (i - self.min_train) % 100 == 0:
                    pct = (i - self.min_train) / (n - self.min_train) * 100
                    print(f"  Progress: {pct:.0f}% - match {i}/{n}")

            # ── Predict next match ─────────────────────────────────
            row       = data.iloc[i]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            # Handle teams absent from the training data — i.e. teams
            # promoted into the division that have no prior matches.
            unseen = [t for t in (home_team, away_team)
                      if t not in model.team_to_idx_]

            if unseen:
                if not self.handle_promoted:
                    # Old behaviour: skip the match entirely.
                    skipped_matches += 1
                    continue

                # Seed each promoted team with a prior rating so we can
                # still predict. On the next retrain these teams have real
                # matches in the training window and are fitted normally.
                for team in unseen:
                    model.register_promoted_team(
                        team,
                        method    = self.promoted_method,
                        n_weakest = self.n_weakest,
                    )
                    promoted_seen.add(team)
                promoted_matches += 1

            pred    = model.predict(home_team, away_team)
            outcome = self._get_result(row)

            # Total-goals market, priced off the same scoreline matrix
            p_over, p_under = model.predict_totals(
                home_team, away_team, self.totals_line)
            total_goals = row['FTHG'] + row['FTAG']

            # Probability array — must match outcome encoding
            # [P(away), P(draw), P(home)]
            probs = np.array([
                pred.away_win,
                pred.draw,
                pred.home_win
            ])

            # ── Compute metrics ────────────────────────────────────
            rps      = self._rps(probs, outcome)

            # Log loss for single match
            # sklearn expects shape (1, n_classes)
            ll = log_loss(
                [outcome],
                [probs],
                labels=[0, 1, 2]
            )

            results.append({
                'date'     : row['Date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_win' : round(pred.home_win, 4),
                'draw'     : round(pred.draw,     4),
                'away_win' : round(pred.away_win, 4),
                'exp_home' : round(pred.exp_home, 2),
                'exp_away' : round(pred.exp_away, 2),
                'outcome'  : outcome,   # 2=H, 1=D, 0=A
                'rps'      : round(rps, 4),
                'log_loss' : round(ll,  4),
                'p_over'   : round(p_over,  4),
                'p_under'  : round(p_under, 4),
                'total_goals'  : int(total_goals),
                'over_actual'  : int(total_goals > self.totals_line),
                'season'   : row.get('season', 'unknown')
            })

        print(f"\nDone - {len(results)} matches predicted")
        if self.handle_promoted:
            print(f"  Promoted-team matches predicted via prior : "
                  f"{promoted_matches} "
                  f"({len(promoted_seen)} teams: {', '.join(sorted(promoted_seen)) or 'none'})")
        else:
            print(f"  Matches skipped (unseen team)             : {skipped_matches}")
        print()
        return pd.DataFrame(results)

    def summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Print a clean summary of backtest performance.

        Parameters
        ----------
        results : pd.DataFrame — output of .run()

        Returns
        -------
        pd.DataFrame — summary metrics
        """
        if results.empty:
            print("No results to summarise.")
            return

        # ── Overall metrics ────────────────────────────────────────
        mean_rps = results['rps'].mean()
        mean_ll  = results['log_loss'].mean()
        n        = len(results)

        # ── Accuracy (did highest probability outcome occur?) ──────
        outcome_map  = {2: 'home_win', 1: 'draw', 0: 'away_win'}
        predicted    = results.apply(
            lambda r: max(
                ['home_win', 'draw', 'away_win'],
                key=lambda x: r[x]
            ), axis=1
        )
        actual       = results['outcome'].map(outcome_map)
        accuracy     = (predicted == actual).mean()

        # ── Baseline: always predict home win ─────────────────────
        home_win_rate = (results['outcome'] == 2).mean()

        print("=" * 50)
        print("       BACKTEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"  Matches predicted : {n}")
        print(f"  Mean RPS          : {mean_rps:.4f}  (lower = better)")
        print(f"  Mean Log Loss     : {mean_ll:.4f}  (lower = better)")
        print(f"  Accuracy          : {accuracy:.1%}")
        print(f"  Baseline accuracy : {home_win_rate:.1%}  (always home)")
        print("=" * 50)

        # ── Per season breakdown ───────────────────────────────────
        if 'season' in results.columns:
            print("\nPer Season:")
            season_summary = (
                results.groupby('season')
                .agg(
                    matches  = ('rps',      'count'),
                    mean_rps = ('rps',      'mean'),
                    mean_ll  = ('log_loss', 'mean')
                )
                .round(4)
            )
            print(season_summary.to_string())
            print()

        return pd.DataFrame([{
            'n'        : n,
            'mean_rps' : round(mean_rps, 4),
            'mean_ll'  : round(mean_ll,  4),
            'accuracy' : round(accuracy, 4),
            'baseline' : round(home_win_rate, 4)
        }])