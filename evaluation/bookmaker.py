# evaluation/bookmaker.py
# ─────────────────────────────────────────────────────────────────
# Bookmaker comparison tools.
#
# Answers three questions:
#   1. Is our model well calibrated?
#   2. Where does our model disagree with the bookmaker?
#   3. When we disagree, are we right? (CLV)
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class BookmakerComparison:
    """
    Compares model predictions against bookmaker odds.

    Requires football-data.co.uk data which includes
    Bet365 odds columns: B365H, B365D, B365A

    Parameters
    ----------
    margin_method : str
        How to remove bookmaker margin:
        'basic'  — divide by overround
        'power'  — power method (more accurate)

    Example
    -------
    >>> bk = BookmakerComparison()
    >>> bk_probs = bk.extract_odds(raw_data)
    >>> merged   = bk.merge_with_results(results, bk_probs)
    >>> bk.full_report(merged)
    """

    def __init__(self, margin_method: str = 'basic'):
        self.margin_method = margin_method

    # ── Odds Processing ───────────────────────────────────────────

    def extract_odds(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and normalise bookmaker odds from raw data.

        Removes the bookmaker margin (overround) to get
        true implied probabilities.

        Parameters
        ----------
        data : pd.DataFrame
            Raw football-data.co.uk data with B365H, B365D, B365A

        Returns
        -------
        pd.DataFrame with columns:
            date, home_team, away_team,
            bk_home, bk_draw, bk_away,   ← normalised probabilities
            overround                     ← bookmaker margin
        """
        # Drop rows with missing odds
        odds_data = data.dropna(subset=['B365H', 'B365D', 'B365A']).copy()

        # Raw implied probabilities (sum > 1 due to margin)
        raw_home = 1 / odds_data['B365H']
        raw_draw = 1 / odds_data['B365D']
        raw_away = 1 / odds_data['B365A']

        # Overround = how much bookmaker takes
        # e.g. 1.05 = 5% margin
        overround = raw_home + raw_draw + raw_away

        if self.margin_method == 'basic':
            # Divide each probability by overround
            # Simple but slightly inaccurate
            bk_home = raw_home / overround
            bk_draw = raw_draw / overround
            bk_away = raw_away / overround

        elif self.margin_method == 'power':
            # Power method — more accurate margin removal
            # Finds exponent k such that sum(p^(1/k)) = 1
            # Better preserves relative probabilities
            bk_home, bk_draw, bk_away = self._power_method(
                raw_home.values,
                raw_draw.values,
                raw_away.values
            )

        return pd.DataFrame({
            'date'      : odds_data['Date'].values,
            'home_team' : odds_data['HomeTeam'].values,
            'away_team' : odds_data['AwayTeam'].values,
            'bk_home'   : np.round(bk_home, 4),
            'bk_draw'   : np.round(bk_draw, 4),
            'bk_away'   : np.round(bk_away, 4),
            'overround' : np.round(overround, 4),
            'B365H'     : odds_data['B365H'].values,
            'B365D'     : odds_data['B365D'].values,
            'B365A'     : odds_data['B365A'].values,
        })

    def _power_method(self, raw_home, raw_draw, raw_away):
        """
        Power method for margin removal.

        Finds k such that:
            (raw_home)^k + (raw_draw)^k + (raw_away)^k = 1

        More accurate than basic normalisation because it
        preserves the relative shape of the probability
        distribution better.
        """
        from scipy.optimize import brentq

        bk_home = np.zeros(len(raw_home))
        bk_draw = np.zeros(len(raw_home))
        bk_away = np.zeros(len(raw_home))

        for i in range(len(raw_home)):
            h, d, a = raw_home[i], raw_draw[i], raw_away[i]

            def equation(k):
                return h**k + d**k + a**k - 1.0

            try:
                k = brentq(equation, 0.5, 2.0)
                bk_home[i] = h**k
                bk_draw[i] = d**k
                bk_away[i] = a**k
            except ValueError:
                # Fallback to basic if power method fails
                total      = h + d + a
                bk_home[i] = h / total
                bk_draw[i] = d / total
                bk_away[i] = a / total

        return bk_home, bk_draw, bk_away

    # ── Merging ───────────────────────────────────────────────────

    def merge_with_results(self, results: pd.DataFrame,
                           bk_probs: pd.DataFrame) -> pd.DataFrame:
        """
        Merge backtest results with bookmaker probabilities.

        Parameters
        ----------
        results  : output of Backtester.run()
        bk_probs : output of extract_odds()

        Returns
        -------
        Merged DataFrame with both model and bookmaker probs
        """
        # Standardise date format for merging
        results  = results.copy()
        bk_probs = bk_probs.copy()

        results['date']       = pd.to_datetime(results['date'])
        bk_probs['date']      = pd.to_datetime(bk_probs['date'])

        merged = results.merge(
            bk_probs,
            left_on  = ['date', 'home_team', 'away_team'],
            right_on = ['date', 'home_team', 'away_team'],
            how      = 'inner'
        )

        print(f"Merged {len(merged)} matches "
              f"({len(results) - len(merged)} dropped — missing odds)")

        return merged

    # ── Core Metrics ──────────────────────────────────────────────

    def closing_line_value(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Closing Line Value (CLV) — the gold standard metric.

        Measures whether your model finds value BEFORE
        the bookmaker prices it in.

        Positive CLV = your model is systematically smarter
                       than the bookmaker on those bets.

        CLV = (your_prob / bookmaker_prob) - 1

        e.g. CLV = +0.05 means your model gives 5% more
             probability than the bookmaker — a genuine edge.
        """
        merged = merged.copy()

        # CLV for each outcome
        merged['clv_home'] = (merged['home_win'] / merged['bk_home']) - 1
        merged['clv_draw'] = (merged['draw']     / merged['bk_draw']) - 1
        merged['clv_away'] = (merged['away_win'] / merged['bk_away']) - 1

        # CLV for the outcome your model favours most
        # (the bet you would actually place)
        def best_clv(row):
            """CLV of the outcome with highest model probability."""
            if   row['home_win'] >= row['draw'] and \
                 row['home_win'] >= row['away_win']:
                return row['clv_home'], 'home'
            elif row['draw'] >= row['home_win'] and \
                 row['draw'] >= row['away_win']:
                return row['clv_draw'], 'draw'
            else:
                return row['clv_away'], 'away'

        merged[['best_clv', 'best_bet']] = merged.apply(
            lambda r: pd.Series(best_clv(r)), axis=1
        )

        return merged

    def disagreement(self, merged: pd.DataFrame,
                     threshold: float = 0.05) -> pd.DataFrame:
        """
        Find matches where model significantly disagrees
        with the bookmaker.

        Parameters
        ----------
        threshold : float
            Minimum probability difference to flag as disagreement
            0.05 = model gives 5%+ more/less than bookmaker

        Returns
        -------
        DataFrame of matches where model disagrees with market
        """
        merged = merged.copy()

        # Absolute difference between model and bookmaker
        merged['diff_home'] = merged['home_win'] - merged['bk_home']
        merged['diff_draw'] = merged['draw']     - merged['bk_draw']
        merged['diff_away'] = merged['away_win'] - merged['bk_away']

        # Maximum disagreement across any outcome
        merged['max_diff'] = merged[
            ['diff_home', 'diff_draw', 'diff_away']
        ].abs().max(axis=1)

        # Flag significant disagreements
        disagreements = merged[merged['max_diff'] >= threshold].copy()

        return disagreements.sort_values('max_diff', ascending=False)

    def rps(self, probs: np.ndarray,
            outcomes: np.ndarray) -> float:
        """RPS for an array of predictions."""
        n      = len(outcomes)
        rps_sum = 0.0

        for i in range(n):
            actual          = np.zeros(3)
            actual[outcomes[i]] = 1.0
            cum_pred        = np.cumsum(probs[i])
            cum_actual      = np.cumsum(actual)
            rps_sum        += np.sum((cum_pred - cum_actual) ** 2) / 2

        return rps_sum / n

    # ── Calibration ───────────────────────────────────────────────

    def calibration_data(self, merged: pd.DataFrame,
                         n_bins: int = 10) -> pd.DataFrame:
        """
        Compute calibration data for the model.

        Groups predictions into probability bins and checks
        whether predicted probabilities match actual frequencies.

        Perfect calibration: when model says 60%, wins 60%.
        """
        bins    = np.linspace(0, 1, n_bins + 1)
        records = []

        # Check calibration for each outcome
        for outcome, col, bk_col in [
            ('Home Win', 'home_win', 'bk_home'),
            ('Draw',     'draw',     'bk_draw'),
            ('Away Win', 'away_win', 'bk_away')
        ]:
            actual = (merged['outcome'] == {'Home Win': 2,
                                             'Draw': 1,
                                             'Away Win': 0}[outcome]).astype(int)

            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                mask   = (merged[col] >= lo) & (merged[col] < hi)

                if mask.sum() < 5:   # skip bins with too few samples
                    continue

                records.append({
                    'outcome'       : outcome,
                    'bin_mid'       : (lo + hi) / 2,
                    'predicted_prob': merged.loc[mask, col].mean(),
                    'actual_freq'   : actual[mask].mean(),
                    'n'             : mask.sum()
                })

        return pd.DataFrame(records)

    # ── Full Report ───────────────────────────────────────────────

    def full_report(self, merged: pd.DataFrame):
        """
        Print complete bookmaker comparison report.
        """
        merged = self.closing_line_value(merged)

        # ── Model vs Bookmaker RPS ─────────────────────────────────
        model_probs = merged[['away_win', 'draw', 'home_win']].values
        bk_probs    = merged[['bk_away',  'bk_draw', 'bk_home']].values
        outcomes    = merged['outcome'].values

        model_rps   = self.rps(model_probs, outcomes)
        bk_rps      = self.rps(bk_probs,    outcomes)

        # ── CLV Summary ────────────────────────────────────────────
        mean_clv    = merged['best_clv'].mean()
        pos_clv_pct = (merged['best_clv'] > 0).mean()

        # ── Overround ──────────────────────────────────────────────
        mean_overround = merged['overround'].mean()

        print("=" * 55)
        print("        BOOKMAKER COMPARISON REPORT")
        print("=" * 55)
        print(f"\n  Matches compared    : {len(merged)}")
        print(f"  Mean overround      : {mean_overround:.3f} "
              f"({(mean_overround - 1) * 100:.1f}% margin)")

        print(f"\n  ── RPS Comparison ──────────────────────────")
        print(f"  Model RPS           : {model_rps:.4f}")
        print(f"  Bookmaker RPS       : {bk_rps:.4f}")
        delta = bk_rps - model_rps
        sign  = '✅ Model better' if delta > 0 else '❌ Bookmaker better'
        print(f"  Difference          : {delta:+.4f}  {sign}")

        print(f"\n  ── Closing Line Value ──────────────────────")
        print(f"  Mean CLV            : {mean_clv:+.4f} "
              f"({'positive ✅' if mean_clv > 0 else 'negative ❌'})")
        print(f"  % predictions +CLV  : {pos_clv_pct:.1%}")

        print(f"\n  ── Disagreement ────────────────────────────")
        disagree = self.disagreement(merged, threshold=0.05)
        print(f"  Matches >5% diff    : {len(disagree)} "
              f"({len(disagree)/len(merged):.1%})")

        # When model disagrees, is it right?
        if len(disagree) > 0:
            disagree_outcomes = disagree['outcome'].values
            disagree_model    = disagree[['away_win', 'draw', 'home_win']].values
            disagree_bk       = disagree[['bk_away', 'bk_draw', 'bk_home']].values

            disagree_model_rps = self.rps(disagree_model, disagree_outcomes)
            disagree_bk_rps    = self.rps(disagree_bk,    disagree_outcomes)

            print(f"  Model RPS (disagree): {disagree_model_rps:.4f}")
            print(f"  Bookie RPS (disagree): {disagree_bk_rps:.4f}")
            sign = '✅' if disagree_model_rps < disagree_bk_rps else '❌'
            print(f"  Model wins on disagreements: {sign}")

        print("=" * 55)

    # ── Visualisation ─────────────────────────────────────────────

    def plot(self, merged: pd.DataFrame):
        """
        Four-panel visualisation:
        1. Calibration curve
        2. CLV distribution
        3. Model vs bookmaker probability scatter
        4. RPS over time
        """
        merged   = self.closing_line_value(merged)
        cal_data = self.calibration_data(merged)

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig)

        # ── Panel 1: Calibration Curve ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        colours = {'Home Win': 'blue', 'Draw': 'grey', 'Away Win': 'red'}

        for outcome in ['Home Win', 'Draw', 'Away Win']:
            subset = cal_data[cal_data['outcome'] == outcome]
            ax1.scatter(
                subset['predicted_prob'],
                subset['actual_freq'],
                label  = outcome,
                color  = colours[outcome],
                s      = subset['n'] * 2,   # size = sample size
                alpha  = 0.7
            )

        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Actual Frequency')
        ax1.set_title('Calibration Curve')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: CLV Distribution ──────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(
            merged['best_clv'],
            bins    = 40,
            color   = 'steelblue',
            edgecolor = 'white',
            alpha   = 0.8
        )
        ax2.axvline(0, color='red', linestyle='--', label='Zero CLV')
        ax2.axvline(
            merged['best_clv'].mean(),
            color     = 'green',
            linestyle = '-',
            label     = f"Mean CLV: {merged['best_clv'].mean():+.3f}"
        )
        ax2.set_xlabel('Closing Line Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CLV Distribution')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: Model vs Bookmaker Scatter ────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(
            merged['bk_home'],
            merged['home_win'],
            alpha = 0.3,
            s     = 10,
            color = 'blue',
            label = 'Home Win'
        )
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect agreement')
        ax3.set_xlabel('Bookmaker Probability')
        ax3.set_ylabel('Model Probability')
        ax3.set_title('Model vs Bookmaker (Home Win)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── Panel 4: Rolling RPS Over Time ─────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])

        merged_sorted = merged.sort_values('date')
        window        = 38   # one season rolling window

        rolling_model_rps = (
            merged_sorted['rps']
            .rolling(window, min_periods=10)
            .mean()
        )

        # Bookmaker rolling RPS
        bk_rps_series = merged_sorted.apply(
            lambda r: self.rps(
                np.array([[r['bk_away'], r['bk_draw'], r['bk_home']]]),
                np.array([r['outcome']])
            ), axis=1
        )
        rolling_bk_rps = bk_rps_series.rolling(window, min_periods=10).mean()

        ax4.plot(
            merged_sorted['date'],
            rolling_model_rps,
            label = 'Model RPS',
            color = 'blue'
        )
        ax4.plot(
            merged_sorted['date'],
            rolling_bk_rps,
            label     = 'Bookmaker RPS',
            color     = 'red',
            linestyle = '--'
        )
        ax4.set_xlabel('Date')
        ax4.set_ylabel(f'Rolling RPS ({window} match window)')
        ax4.set_title('Model vs Bookmaker RPS Over Time')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Bookmaker Comparison Report', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('outputs/bookmaker_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved to outputs/bookmaker_comparison.png")