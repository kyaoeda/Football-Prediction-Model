# config.py
# ─────────────────────────────────────────────────────────────────
# Central configuration for the football prediction project.
# Change values here — they update everywhere automatically.
# ─────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent   # folder where config.py lives

# ── Season URLs (football-data.co.uk) ──────────────────────────────
# E0 = Premier League, E1 = Championship. Pick the active league via
# SEASONS below — that's the only line to change to switch division.
SEASONS_PL = {
    '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
    '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
    '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
    '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
    '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
}

SEASONS_CH = {
    '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/E1.csv',
    '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E1.csv',
    '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E1.csv',
    '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E1.csv',
    '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E1.csv',
}

# Deeper Premier League history for cross-validation / model training.
# Understat xG starts at 2014-15, so that is the earliest usable season.
# NOTE: best-price (Max*) and closing (*C*) odds columns only exist from
# 2019-20 onward — earlier seasons carry Bet365 pre-match odds only.
SEASONS_PL_FULL = {
    f'{y}-{str(y + 1)[-2:]}':
        f'https://www.football-data.co.uk/mmz4281/{str(y)[-2:]}{str(y + 1)[-2:]}/E0.csv'
    for y in range(2014, 2026)
}

# Active league for the run:
SEASONS = SEASONS_PL

# ── Promoted Teams Per Season ──────────────────────────────────────
PROMOTED_TEAMS = {
    '2021-22': ['Norwich', 'Watford', 'Brentford'],
    '2022-23': ['Fulham', 'Bournemouth', 'Nott\'m\' Forest'],
    '2023-24': ['Burnley', 'Sheffield United', 'Luton Town'],
    '2024-25': ['Leicester City', 'Ipswich Town', 'Southampton'],
    '2025-26': ['Sunderland', 'Burnley', 'Leeds United'],
}
# ── Dixon-Coles Parameters ─────────────────────────────────────────
DIXON_COLES_XI        = 0.00325  # time decay rate
DIXON_COLES_MAX_GOALS = 10        # max goals in scoreline matrix

# ── Promoted Team Handling ─────────────────────────────────────────
# When a promoted team first appears mid-backtest it has no history,
# so we seed it with a prior rating instead of skipping the match.
#   'weakest'    — average of the N weakest fitted teams (recommended)
#   'average'    — league-average rating
#   'percentile' — 25th percentile of fitted ratings
PROMOTED_RATING_METHOD = 'weakest'
PROMOTED_N_WEAKEST     = 3        # teams used when method = 'weakest'

# ── Backtest Split ─────────────────────────────────────────────────
TEST_SEASON = '2025-26'          # predict this season; train on all seasons before it

# Genuine value test: bet at the BEST AVAILABLE price across bookmakers
# (Max*), then grade every bet against the best price at CLOSING. Comparing
# best-price-now vs best-price-at-close is apples-to-apples: it isolates model
# skill from the mechanical advantage of shopping across books.
#
# Why Max over B365: mean overround is 1.023 vs 1.056 — shopping saves ~3.3
# points of margin per bet, which dwarfs most modelling gains.
# Set CLOSING_BOOKMAKER = None to disable CLV grading.
BOOKMAKER         = 'market_max'          # best available price you bet at
CLOSING_BOOKMAKER = 'market_max_closing'  # best price at close, for CLV grading

# ── Betting Parameters ─────────────────────────────────────────────
BANK                  = 1000.0   # starting bankroll
STAKE                 = 10.0     # flat stake per bet
MIN_EDGE              = 0.05     # minimum edge to place bet (5%)
MAX_EDGE              = 0.15     # maximum edge to place bet (15%)
MIN_ODDS              = 1.50     # ignore very short odds
MAX_ODDS              = 4.0     # ignore very long odds (too noisy)