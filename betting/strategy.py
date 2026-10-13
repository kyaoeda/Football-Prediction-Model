# betting/strategy.py
# ─────────────────────────────────────────────────────────────────
# Betting strategy — finds value bets and tracks profit.
#
# Supports:
#   - Flat staking
#   - Kelly criterion staking
#   - Edge-proportional staking
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import BANK, STAKE, MIN_EDGE, MAX_EDGE, MIN_ODDS, MAX_ODDS


class BettingStrategy:
    """
    Identifies value bets from backtest results and
    tracks profit/loss across a season.

    Parameters
    ----------
    bank       : float — starting bankroll
    stake      : float — flat stake per bet
    min_edge   : float — minimum edge required to bet
                         0.05 = only bet when model gives
                         5%+ more probability than bookmaker
    max_edge   : float — maximum edge to consider
    min_odds   : float — minimum odds to consider
    max_odds   : float — maximum odds to consider
    method     : str   — staking method
                         'flat'  = same stake every bet
                         'kelly' = Kelly criterion
                         'edge'  = proportional to edge size

    Example
    -------
    >>> strategy = BettingStrategy(bank=1000, stake=10, min_edge=0.05)
    >>> bets     = strategy.find_value_bets(merged)
    >>> strategy.summary(bets)
    >>> strategy.plot(bets)
    """

    def __init__(self,
                 bank     : float = BANK,
                 stake    : float = STAKE,
                 min_edge : float = MIN_EDGE,
                 max_edge : float = MAX_EDGE,
                 min_odds : float = MIN_ODDS,
                 max_odds : float = MAX_ODDS,
                 method   : str   = 'flat'):

        self.bank     = bank
        self.stake    = stake
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.method   = method

    # ── Private Helpers ───────────────────────────────────────────

    def _kelly_stake(self, model_prob: float,
                     odds: float,
                     bank: float,
                     fraction: float = 0.25) -> float:
        """
        Kelly Criterion stake.

        Full Kelly is theoretically optimal but very aggressive
        in practice. We use fractional Kelly (default 25%)
        which is much safer.

        Formula:
            f = (b*p - q) / b
            where:
                b = decimal odds - 1
                p = model probability   (our estimate of P(win))
                q = 1 - p

        Parameters
        ----------
        model_prob : our probability for this outcome (P(win))
        odds       : decimal odds we would take
        bank       : current bankroll
        fraction   : Kelly fraction (0.25 = quarter Kelly)
        """
        b = odds - 1
        p = model_prob
        q = 1 - p

        kelly = (b * p - q) / b

        # Never bet more than 5% of bank on one bet
        kelly = max(0.0, min(kelly, 0.05))

        return round(bank * kelly * fraction, 2)

    def _edge_stake(self, edge: float,
                    base_stake: float) -> float:
        """
        Edge-proportional staking.
        Larger edge = larger stake, linearly scaled.

        e.g. edge=0.05 → 1x base stake
             edge=0.10 → 2x base stake
             edge=0.15 → 3x base stake (capped at 3x)
        """
        multiplier = min(edge / self.min_edge, 3.0)
        return round(base_stake * multiplier, 2)

    def _get_stake(self, edge: float,
                   odds: float,
                   bank: float,
                   model_prob: float) -> float:
        """Return stake based on chosen staking method."""
        if   self.method == 'flat' : return self.stake
        elif self.method == 'kelly': return self._kelly_stake(model_prob, odds, bank)
        elif self.method == 'edge' : return self._edge_stake(edge, self.stake)
        else                       : return self.stake

    def _outcome_name(self, outcome: int) -> str:
        """Convert outcome integer to string."""
        return {2: 'home', 1: 'draw', 0: 'away'}[outcome]

    # ── Core Method ───────────────────────────────────────────────

    def find_value_bets(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Scan all predictions for value bets and simulate
        profit/loss across the dataset.

        Parameters
        ----------
        merged : pd.DataFrame
            Output of BookmakerComparison.merge_with_results()
            Must contain model probs AND bookmaker probs.

        Returns
        -------
        pd.DataFrame — one row per VALUE BET found, with columns:
            date, home_team, away_team,
            bet_on,          ← which outcome we bet on
            model_prob,      ← our probability
            bk_prob,         ← bookmaker implied probability
            odds,            ← decimal odds we get
            edge,            ← model_prob - bk_prob
            stake,           ← amount staked
            outcome,         ← actual result
            won,             ← did we win?
            profit,          ← profit/loss on this bet
            bank,            ← running bankroll
            roi              ← return on investment
        """
        merged = merged.copy().sort_values('date').reset_index(drop=True)

        bets        = []
        bank        = self.bank

        for _, row in merged.iterrows():

            # ── Check each outcome for value ──────────────────────
            outcomes = [
                ('home', row['home_win'], row['bk_home'], row['odds_home']),
                ('draw', row['draw'],     row['bk_draw'], row['odds_draw']),
                ('away', row['away_win'], row['bk_away'], row['odds_away']),
            ]

            # Closing odds per outcome (present only if a closing reference
            # was supplied to extract_odds); used for true CLV grading.
            close_odds = {
                'home': row.get('close_odds_home', np.nan),
                'draw': row.get('close_odds_draw', np.nan),
                'away': row.get('close_odds_away', np.nan),
            }

            for bet_on, model_prob, bk_prob, odds in outcomes:

                # Skip if odds out of range
                if not (self.min_odds <= odds <= self.max_odds):
                    continue

                # Calculate edge
                edge = model_prob - bk_prob

                # Only bet if edge exceeds threshold
                if edge < self.min_edge:
                    continue

                # Skip if edge exceeds maximum
                if edge > self.max_edge:
                    continue

                # Calculate stake
                stake = self._get_stake(edge, odds, bank, model_prob)

                # Skip if stake would exceed bank
                if stake > bank:
                    continue

                # Did we win?
                actual     = self._outcome_name(row['outcome'])
                won        = (actual == bet_on)
                profit     = round((odds - 1) * stake if won else -stake, 2)
                bank       = round(bank + profit, 2)

                # True Closing Line Value: did the price we took (opening)
                # beat the closing line? clv > 0 means we got better odds
                # than the market's sharpest estimate — the real edge signal.
                c_odds = close_odds[bet_on]
                if pd.notna(c_odds) and c_odds > 0:
                    clv        = round(odds / c_odds - 1, 4)
                    beat_close = bool(odds > c_odds)
                else:
                    clv, beat_close = np.nan, None

                bets.append({
                    'date'      : row['date'],
                    'home_team' : row['home_team'],
                    'away_team' : row['away_team'],
                    'bet_on'    : bet_on,
                    'model_prob': round(model_prob, 4),
                    'bk_prob'   : round(bk_prob,    4),
                    'odds'      : round(odds,        2),
                    'close_odds': round(c_odds, 2) if pd.notna(c_odds) else np.nan,
                    'edge'      : round(edge,        4),
                    'clv'       : clv,
                    'beat_close': beat_close,
                    'stake'     : stake,
                    'outcome'   : actual,
                    'won'       : won,
                    'profit'    : profit,
                    'bank'      : bank,
                    'roi'       : round(profit / stake, 4),
                    'season'    : row.get('season', 'unknown')
                })

        return pd.DataFrame(bets)

    # ── Summary ───────────────────────────────────────────────────

    def summary(self, bets: pd.DataFrame):
        """
        Print a complete betting performance summary.
        """
        if bets.empty:
            print("No value bets found. Try lowering min_edge.")
            return

        total_bets    = len(bets)
        total_staked  = bets['stake'].sum()
        total_profit  = bets['profit'].sum()
        win_rate      = bets['won'].mean()
        roi           = total_profit / total_staked
        final_bank    = bets['bank'].iloc[-1]
        bank_growth   = (final_bank - self.bank) / self.bank

        # Max drawdown
        peak          = bets['bank'].cummax()
        drawdown      = (bets['bank'] - peak) / peak
        max_drawdown  = drawdown.min()

        # Longest losing streak
        losing_streak = 0
        max_streak    = 0
        for won in bets['won']:
            if not won:
                losing_streak += 1
                max_streak = max(max_streak, losing_streak)
            else:
                losing_streak = 0

        print("=" * 55)
        print("          BETTING PERFORMANCE SUMMARY")
        print("=" * 55)
        print(f"\n  Staking method    : {self.method}")
        print(f"  Min edge required : {self.min_edge:.0%}")
        print(f"  Starting bank     : GBP {self.bank:,.2f}")
        print(f"  Final bank        : GBP {final_bank:,.2f}")

        print(f"\n  -- Volume ----------------------------------")
        print(f"  Total bets        : {total_bets}")
        print(f"  Total staked      : GBP {total_staked:,.2f}")

        # Breakdown by outcome type
        for outcome in ['home', 'draw', 'away']:
            n = (bets['bet_on'] == outcome).sum()
            print(f"  Bets on {outcome:<5}    : {n}")

        print(f"\n  -- Performance -----------------------------")
        print(f"  Win rate          : {win_rate:.1%}")
        print(f"  Total profit      : GBP {total_profit:+,.2f}")
        print(f"  ROI               : {roi:+.2%}")
        print(f"  Bank growth       : {bank_growth:+.2%}")

        # ── Closing Line Value (real edge signal) ──────────────────
        if 'clv' in bets.columns:
            clv_series = bets['clv'].dropna()
            if len(clv_series):
                print(f"\n  -- Closing Line Value ----------------------")
                print(f"  Bets graded vs close: {len(clv_series)} of {total_bets}")
                print(f"  Mean CLV            : {clv_series.mean():+.2%}")
                print(f"  % beating close     : {(clv_series > 0).mean():.1%}")

        print(f"\n  -- Risk ------------------------------------")
        print(f"  Max drawdown      : {max_drawdown:.2%}")
        print(f"  Longest loss run  : {max_streak} bets")

        # Per season breakdown
        if 'season' in bets.columns:
            print(f"\n  -- Per Season ------------------------------")
            season_summary = (
                bets.groupby('season')
                .agg(
                    bets       = ('profit', 'count'),
                    staked     = ('stake',  'sum'),
                    profit     = ('profit', 'sum'),
                    win_rate   = ('won',    'mean'),
                )
            )
            season_summary['roi'] = (
                season_summary['profit'] / season_summary['staked']
            )
            season_summary = season_summary.round(3)
            print(season_summary.to_string())

        print("=" * 55)

    # ── Visualisation ─────────────────────────────────────────────

    def plot(self, bets: pd.DataFrame):
        """
        Four-panel betting performance visualisation:
        1. Cumulative profit over time
        2. Profit distribution per bet
        3. ROI by edge bucket
        4. Win rate by odds bucket
        """
        if bets.empty:
            print("No bets to plot.")
            return

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig)

        # ── Panel 1: Cumulative Profit ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])

        cumulative_profit = bets['profit'].cumsum()
        ax1.plot(
            range(len(bets)),
            cumulative_profit,
            color     = 'blue',
            linewidth = 1.5,
            label     = 'Cumulative Profit'
        )
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax1.fill_between(
            range(len(bets)),
            cumulative_profit,
            0,
            where  = cumulative_profit >= 0,
            alpha  = 0.2,
            color  = 'green',
            label  = 'Profit'
        )
        ax1.fill_between(
            range(len(bets)),
            cumulative_profit,
            0,
            where  = cumulative_profit < 0,
            alpha  = 0.2,
            color  = 'red',
            label  = 'Loss'
        )
        ax1.set_xlabel('Bet Number')
        ax1.set_ylabel('Cumulative Profit (£)')
        ax1.set_title('Cumulative Profit Over Time')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Profit Distribution ──────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])

        colours = ['green' if p > 0 else 'red' for p in bets['profit']]
        ax2.bar(range(len(bets)), bets['profit'],
                color=colours, alpha=0.6, width=1.0)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Bet Number')
        ax2.set_ylabel('Profit (£)')
        ax2.set_title('Profit Per Bet')
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: ROI By Edge Bucket ────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])

        bets['edge_bucket'] = pd.cut(
            bets['edge'],
            bins   = [0.05, 0.10, 0.15, 0.20, 0.30, 1.0],
            labels = ['5-10%', '10-15%', '15-20%', '20-30%', '>30%']
        )

        edge_roi = (
            bets.groupby('edge_bucket', observed=True)
            .apply(lambda x: x['profit'].sum() / x['stake'].sum(),
                   include_groups=False)
            .reset_index()
        )
        edge_roi.columns = ['edge_bucket', 'roi']

        colours = ['green' if r > 0 else 'red'
                   for r in edge_roi['roi']]
        ax3.bar(
            edge_roi['edge_bucket'].astype(str),
            edge_roi['roi'] * 100,
            color = colours,
            alpha = 0.7
        )
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_xlabel('Edge Size')
        ax3.set_ylabel('ROI (%)')
        ax3.set_title('ROI by Edge Size')
        ax3.grid(True, alpha=0.3, axis='y')

        # ── Panel 4: Win Rate By Odds Bucket ──────────────────────
        ax4 = fig.add_subplot(gs[1, 1])

        bets['odds_bucket'] = pd.cut(
            bets['odds'],
            bins   = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0],
            labels = ['1.0-1.5', '1.5-2.0', '2.0-2.5',
                      '2.5-3.0', '3.0-5.0', '5.0-10.0']
        )

        odds_stats = (
            bets.groupby('odds_bucket', observed=True)
            .agg(
                win_rate     = ('won',    'mean'),
                implied_prob = ('bk_prob', 'mean'),
                n            = ('won',    'count')
            )
            .reset_index()
        )

        x     = range(len(odds_stats))
        width = 0.35

        ax4.bar(
            [i - width/2 for i in x],
            odds_stats['win_rate'] * 100,
            width  = width,
            label  = 'Actual Win Rate',
            color  = 'blue',
            alpha  = 0.7
        )
        ax4.bar(
            [i + width/2 for i in x],
            odds_stats['implied_prob'] * 100,
            width  = width,
            label  = 'Bookmaker Implied',
            color  = 'orange',
            alpha  = 0.7
        )

        ax4.set_xticks(list(x))
        ax4.set_xticklabels(
            odds_stats['odds_bucket'].astype(str),
            rotation = 45,
            ha       = 'right'
        )
        ax4.set_ylabel('Probability (%)')
        ax4.set_title('Actual Win Rate vs Bookmaker Implied')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle(
            f'Betting Performance — {self.method.title()} Staking '
            f'(min edge: {self.min_edge:.0%})',
            fontsize = 14
        )
        plt.tight_layout()
        plt.savefig('outputs/betting_performance.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved to outputs/betting_performance.png")