# models/dixon_coles.py

import numpy as np
import pandas as pd
import warnings
from scipy.stats             import poisson
from scipy.optimize          import minimize, LinearConstraint
from dataclasses             import dataclass
from config                  import DIXON_COLES_XI, DIXON_COLES_MAX_GOALS


@dataclass
class MatchPrediction:
    """Clean container for model predictions."""
    home_win  : float
    draw      : float
    away_win  : float
    exp_home  : float
    exp_away  : float

    def __repr__(self):
        return (
            f"Home Win : {self.home_win:.1%}\n"
            f"Draw     : {self.draw:.1%}\n"
            f"Away Win : {self.away_win:.1%}\n"
            f"Exp Home : {self.exp_home:.2f}\n"
            f"Exp Away : {self.exp_away:.2f}"
        )

    def to_dict(self):
        return {
            'home_win': round(self.home_win, 4),
            'draw'    : round(self.draw,     4),
            'away_win': round(self.away_win, 4),
            'exp_home': round(self.exp_home, 4),
            'exp_away': round(self.exp_away, 4),
        }


class DixonColesModel:
    """
    Dixon-Coles football match prediction model.

    Uses a sum-to-zero constraint on attack parameters
    to ensure a unique solution and stable optimisation.

    Parameters
    ----------
    xi        : float — time decay rate
    max_goals : int   — max goals in scoreline matrix
    """

    def __init__(self,
                 xi        : float = DIXON_COLES_XI,
                 max_goals : int   = DIXON_COLES_MAX_GOALS):
        self.xi        = xi
        self.max_goals = max_goals
        self.params_   = None
        self.teams_    = None
        self.fitted_   = False

    # ── Private Helpers ───────────────────────────────────────────

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError(
                "Model not fitted. Call .fit(data) first."
            )

    def _check_teams(self, home_team: str, away_team: str):
        for team in [home_team, away_team]:
            if team not in self.teams_:
                raise ValueError(
                    f"Unknown team: '{team}'.\n"
                    f"Available: {sorted(self.teams_)}"
                )

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        """Exponential decay — recent matches weighted more."""
        days_ago = (dates.max() - dates).dt.days
        return np.exp(-self.xi * days_ago)

    def _tau(self, x: int, y: int,
             lam: float, mu: float,
             rho: float) -> float:
        """Dixon-Coles low-score correction."""
        if   x == 0 and y == 0: return 1 - lam * mu * rho
        elif x == 0 and y == 1: return 1 + lam * rho
        elif x == 1 and y == 0: return 1 + mu  * rho
        elif x == 1 and y == 1: return 1 - rho
        else                  : return 1.0

    def _unpack_params(self, params: np.ndarray) -> dict:
        """
        Convert flat parameter array to named dictionary.

        Parameter array layout:
        [ attack_0 ... attack_n,
          defence_0 ... defence_n,
          home_adv,
          rho ]
        """
        n = len(self.teams_)

        return {
            'attack'  : dict(zip(self.teams_, params[:n])),
            'defence' : dict(zip(self.teams_, params[n:2*n])),
            'home_adv': float(params[-2]),
            'rho'     : float(params[-1])
        }

    def _initial_guess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Data-driven initial parameter guess.

        Uses each team's actual scoring/conceding rates
        relative to the league average as starting values.
        Pre-centres attack params so they already sum to ~0.
        This gives the optimiser a near-optimal starting point
        and dramatically reduces warnings.
        """
        n            = len(self.teams_)
        league_avg   = (data['FTHG'].mean() + data['FTAG'].mean()) / 2

        attack_init  = []
        defence_init = []

        for team in self.teams_:
            home_scored   = data[data['HomeTeam'] == team]['FTHG']
            away_scored   = data[data['AwayTeam'] == team]['FTAG']
            home_conceded = data[data['HomeTeam'] == team]['FTAG']
            away_conceded = data[data['AwayTeam'] == team]['FTHG']

            avg_scored    = pd.concat([home_scored, away_scored]).mean()
            avg_conceded  = pd.concat([home_conceded, away_conceded]).mean()

            # Log scale relative to league average
            attack_init.append(
                np.log(max(avg_scored,   0.5) / league_avg)
            )
            defence_init.append(
                np.log(max(avg_conceded, 0.5) / league_avg)
            )

        attack_init  = np.array(attack_init)
        defence_init = np.array(defence_init)

        # Pre-centre attack so it already sums close to zero
        # This puts the optimiser very close to the solution
        attack_init -= attack_init.mean()

        return np.concatenate([
            attack_init,
            defence_init,
            [0.25],    # home advantage
            [-0.1]     # rho
        ])

    def _neg_log_likelihood(self, params: np.ndarray,
                             data: pd.DataFrame,
                             weights: np.ndarray) -> float:
        """
        Vectorised negative log likelihood.
        Minimising this = maximising log likelihood.
        """
        p = self._unpack_params(params)

        # ── Expected goals ─────────────────────────────────────────
        attack_home  = data['HomeTeam'].map(p['attack']).values
        defence_home = data['HomeTeam'].map(p['defence']).values
        attack_away  = data['AwayTeam'].map(p['attack']).values
        defence_away = data['AwayTeam'].map(p['defence']).values

        lam = np.exp(attack_home - defence_away + p['home_adv'])
        mu  = np.exp(attack_away - defence_home)

        # Clip to prevent numerical overflow
        lam = np.clip(lam, 1e-6, 15.0)
        mu  = np.clip(mu,  1e-6, 15.0)

        hg  = data['FTHG'].values.astype(int)
        ag  = data['FTAG'].values.astype(int)

        # ── Log probabilities ──────────────────────────────────────
        log_p_home = np.clip(poisson.logpmf(hg, lam), -20, 0)
        log_p_away = np.clip(poisson.logpmf(ag, mu),  -20, 0)

        # ── Tau correction ─────────────────────────────────────────
        rho  = p['rho']
        tau  = np.ones(len(data))

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10]  * rho
        tau[m11] = 1 - rho

        # Guard against invalid tau
        if np.any(tau <= 0):
            return np.inf

        # ── Weighted log likelihood ────────────────────────────────
        ll = weights * (
            np.log(tau)
            + log_p_home
            + log_p_away
        )

        # Guard against any remaining nan/inf
        if not np.isfinite(ll).all():
            return np.inf

        return -ll.sum()

    # ── Public Interface ──────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> 'DixonColesModel':
        """
        Fit the model using sum-to-zero constraint on attack params.
        """
        data        = data.reset_index(drop=True)
        self.teams_ = sorted(
            set(data['HomeTeam']) | set(data['AwayTeam'])
        )
        n = len(self.teams_)

        # ── Initial guess ──────────────────────────────────────────
        x0 = self._initial_guess(data)

        # ── Time weights ───────────────────────────────────────────
        weights = self._time_weights(data['Date'])
        weights = weights / weights.sum() * len(weights)

        # ── Bounds ────────────────────────────────────────────────
        bounds = (
            [(-3.0, 3.0)] * n +    # attack params
            [(-3.0, 3.0)] * n +    # defence params
            [(0.0,  0.5)]  +       # home advantage
            [(-0.5, 0.0)]          # rho
        )

        # ── Sum-to-zero constraint ─────────────────────────────────
        # Build constraint matrix A of shape (1, n_params)
        # A @ x = 0  enforces  sum(attack params) = 0
        #
        # A looks like:
        # [1, 1, 1, ..., 1, 0, 0, ..., 0, 0, 0]
        #  ↑ attack params ↑  ↑ defence ↑  ↑ home, rho
        n_params = len(x0)
        A        = np.zeros((1, n_params))
        A[0, :n] = 1.0              # only attack params sum to zero

        sum_to_zero = LinearConstraint(
            A,
            lb = 0.0,   # lower bound
            ub = 0.0    # upper bound — lb == ub enforces equality
        )

        # ── Optimise ───────────────────────────────────────────────
        # Must use SLSQP or trust-constr — L-BFGS-B doesn't support
        # equality constraints
        result = minimize(
            fun         = self._neg_log_likelihood,
            x0          = x0,
            args        = (data, weights),
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = sum_to_zero,
            options     = {
                'maxiter': 2000,
                'ftol'   : 1e-12
            }
        )

        if not result.success:
            warnings.warn(
                f"Optimisation warning: {result.message}"
            )

        self.params_ = self._unpack_params(result.x)
        self.fitted_ = True
        return self

    def predict(self, home_team: str,
                away_team: str) -> MatchPrediction:
        """Predict match outcome probabilities."""
        self._check_fitted()
        self._check_teams(home_team, away_team)

        p   = self.params_
        lam = np.exp(
            p['attack'][home_team]
            - p['defence'][away_team]
            + p['home_adv']
        )
        mu  = np.exp(
            p['attack'][away_team]
            - p['defence'][home_team]
        )

        # Scoreline matrix
        home_probs = [poisson.pmf(i, lam) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(j, mu)  for j in range(self.max_goals + 1)]
        matrix     = np.outer(home_probs, away_probs)

        # Dixon-Coles correction
        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self._tau(i, j, lam, mu, p['rho'])

        matrix /= matrix.sum()

        return MatchPrediction(
            home_win = float(np.tril(matrix, -1).sum()),
            draw     = float(np.trace(matrix)),
            away_win = float(np.triu(matrix,  1).sum()),
            exp_home = lam,
            exp_away = mu
        )

    def get_ratings(self) -> pd.DataFrame:
        """Return team ratings sorted by attack strength."""
        self._check_fitted()

        return pd.DataFrame({
            'team'   : self.teams_,
            'attack' : [self.params_['attack'][t]  for t in self.teams_],
            'defence': [self.params_['defence'][t] for t in self.teams_]
        }).sort_values('attack', ascending=False).reset_index(drop=True)

    def predict_batch(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Predict multiple fixtures at once."""
        self._check_fitted()

        predictions = []
        for _, row in fixtures.iterrows():
            pred = self.predict(row['HomeTeam'], row['AwayTeam'])
            predictions.append(pred.to_dict())

        return pd.concat(
            [fixtures.reset_index(drop=True),
             pd.DataFrame(predictions)],
            axis=1
        )

    def __repr__(self):
        status = "fitted" if self.fitted_ else "not fitted"
        teams  = len(self.teams_) if self.teams_ else 0
        return (
            f"DixonColesModel("
            f"xi={self.xi}, "
            f"max_goals={self.max_goals}, "
            f"status={status}, "
            f"teams={teams})"
        )