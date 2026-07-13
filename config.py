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
DIXON_COLES_XI        = 0.002  # time decay rate
DIXON_COLES_MAX_GOALS = 10        # max goals in scoreline matrix