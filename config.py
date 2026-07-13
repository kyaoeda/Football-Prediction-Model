# config.py
# ─────────────────────────────────────────────────────────────────
# Central configuration for the football prediction project.
# Change values here — they update everywhere automatically.
# ─────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent   # folder where config.py lives

# ── Seasons ────────────────────────────────────────────────────────
SEASONS = {
    '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
    '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
    '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
    '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
    '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
}

# ── Dixon-Coles Parameters ─────────────────────────────────────────
DIXON_COLES_XI        = 0.00325  # time decay rate
DIXON_COLES_MAX_GOALS = 10        # max goals in scoreline matrix

# ── Betting Parameters ─────────────────────────────────────────────
BANK                  = 1000.0   # starting bankroll
STAKE                 = 10.0     # flat stake per bet
MIN_EDGE              = 0.05     # minimum edge to place bet (5%)
MAX_EDGE              = 0.15     # maximum edge to place bet (15%)
MIN_ODDS              = 1.50     # ignore very short odds
MAX_ODDS              = 4.0     # ignore very long odds (too noisy)