# evaluation/totals.py
# ─────────────────────────────────────────────────────────────────
# Over/Under 2.5 goals: market comparison and value betting.
#
# The 1X2 tooling assumes three outcomes, so totals get their own
# (simpler) two-way version here: devig a pair of odds, compare the
# model to the market, then bet at the best available price and grade
# every bet against the closing line.
#
# Two-way markets are usually priced with a thinner margin than 1X2,
# and the model gets them for free from the Dixon-Coles scoreline
# matrix — no separate model is fitted.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

# (over_column, under_column) for the 2.5 goals line.
# football-data carries these from 2019-20 onward.
TOTALS_BOOKMAKERS = {
    'bet365':             ('B365>2.5',  'B365<2.5'),
    'bet365_closing':     ('B365C>2.5', 'B365C<2.5'),
    'pinnacle':           ('P>2.5',     'P<2.5'),
    'pinnacle_closing':   ('PC>2.5',    'PC<2.5'),
    'market_avg':         ('Avg>2.5',   'Avg<2.5'),
    'market_max':         ('Max>2.5',   'Max<2.5'),
    'market_avg_closing': ('AvgC>2.5',  'AvgC<2.5'),
    'market_max_closing': ('MaxC>2.5',  'MaxC<2.5'),
}


def devig_totals(data: pd.DataFrame, bookmaker: str) -> pd.DataFrame:
    """
    Devig a two-way Over/Under market into fair probabilities.

    Returns date/home/away keys plus:
        p_over_bk, p_under_bk  -- margin-free probabilities
        odds_over, odds_under  -- raw decimal odds
        overround
    """
    if bookmaker not in TOTALS_BOOKMAKERS:
        raise ValueError(f"Unknown totals bookmaker '{bookmaker}'. "
                         f"Choose from: {', '.join(TOTALS_BOOKMAKERS)}")

    o_col, u_col = TOTALS_BOOKMAKERS[bookmaker]
    missing = {o_col, u_col} - set(data.columns)
    if missing:
        raise KeyError(f"Missing totals columns for '{bookmaker}': {sorted(missing)}")

    d = data.dropna(subset=[o_col, u_col]).copy()

    raw_over  = 1.0 / d[o_col]
    raw_under = 1.0 / d[u_col]
    overround = raw_over + raw_under

    return pd.DataFrame({
        'date':       d['Date'].values,
        'home_team':  d['HomeTeam'].values,
        'away_team':  d['AwayTeam'].values,
        'p_over_bk':  np.round(raw_over / overround, 4),
        'p_under_bk': np.round(raw_under / overround, 4),
        'odds_over':  d[o_col].values,
        'odds_under': d[u_col].values,
        'overround':  np.round(overround, 4),
    })


def brier(prob_over: np.ndarray, over_actual: np.ndarray) -> float:
    """Brier score for the two-way market (lower = better)."""
    return float(np.mean((np.asarray(prob_over) - np.asarray(over_actual)) ** 2))


def log_loss_binary(prob_over: np.ndarray, over_actual: np.ndarray) -> float:
    p = np.clip(np.asarray(prob_over, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(over_actual, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def find_totals_bets(merged: pd.DataFrame,
                     min_edge: float = 0.05,
                     max_edge: float = 0.15,
                     min_odds: float = 1.50,
                     max_odds: float = 4.00,
                     stake: float = 10.0,
                     bank: float = 1000.0) -> pd.DataFrame:
    """
    Bet Over or Under whenever the model's edge over the devigged
    best-available price clears min_edge, then grade the taken price
    against the closing line (true CLV).

    `merged` must contain model probs (p_over/p_under), market probs
    (p_over_bk/p_under_bk), taken odds (odds_over/odds_under) and,
    optionally, closing odds (close_over/close_under).
    """
    merged = merged.sort_values('date').reset_index(drop=True)
    bets = []

    for _, row in merged.iterrows():
        sides = [
            ('over',  row['p_over'],  row['p_over_bk'],  row['odds_over'],
             row.get('close_over',  np.nan)),
            ('under', row['p_under'], row['p_under_bk'], row['odds_under'],
             row.get('close_under', np.nan)),
        ]

        for side, model_p, bk_p, odds, c_odds in sides:
            if not (min_odds <= odds <= max_odds):
                continue

            edge = model_p - bk_p
            if edge < min_edge or edge > max_edge:
                continue

            won    = (row['over_actual'] == 1) if side == 'over' else (row['over_actual'] == 0)
            profit = round((odds - 1) * stake if won else -stake, 2)
            bank   = round(bank + profit, 2)

            if pd.notna(c_odds) and c_odds > 0:
                clv = round(odds / c_odds - 1, 4)
            else:
                clv = np.nan

            bets.append({
                'date':       row['date'],
                'home_team':  row['home_team'],
                'away_team':  row['away_team'],
                'side':       side,
                'model_prob': round(model_p, 4),
                'bk_prob':    round(bk_p, 4),
                'odds':       round(odds, 2),
                'close_odds': round(c_odds, 2) if pd.notna(c_odds) else np.nan,
                'edge':       round(edge, 4),
                'clv':        clv,
                'stake':      stake,
                'won':        won,
                'profit':     profit,
                'bank':       bank,
                'season':     row.get('season', 'unknown'),
            })

    return pd.DataFrame(bets)


def summarise_totals_bets(bets: pd.DataFrame, start_bank: float = 1000.0):
    """Print a betting summary for the totals market."""
    if bets.empty:
        print("No value bets found on the totals market.")
        return

    staked = bets['stake'].sum()
    profit = bets['profit'].sum()

    print(f"  Total bets        : {len(bets)}  "
          f"(over={int((bets['side'] == 'over').sum())}, "
          f"under={int((bets['side'] == 'under').sum())})")
    print(f"  Total staked      : GBP {staked:,.2f}")
    print(f"  Win rate          : {bets['won'].mean():.1%}")
    print(f"  Total profit      : GBP {profit:+,.2f}")
    print(f"  ROI               : {profit / staked:+.2%}")

    clv = bets['clv'].dropna()
    if len(clv):
        print(f"  Bets graded vs close: {len(clv)} of {len(bets)}")
        print(f"  Mean CLV          : {clv.mean():+.2%}")
        print(f"  % beating close   : {(clv > 0).mean():.1%}")
