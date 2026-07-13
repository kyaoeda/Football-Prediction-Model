import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from scipy.special import gammaln
from dataclasses import dataclass
from config import DIXON_COLES_XI, DIXON_COLES_MAX_GOALS


@dataclass
class MatchPrediction:
    home_win: float
    draw: float
    away_win: float
    exp_home: float
    exp_away: float

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
            "home_win": round(self.home_win, 4),
            "draw": round(self.draw, 4),
            "away_win": round(self.away_win, 4),
            "exp_home": round(self.exp_home, 4),
            "exp_away": round(self.exp_away, 4),
        }


class DixonColesModel:
    """
    Dixon-Coles football model using unconstrained L-BFGS-B.

    Identification strategy
    -----------------------
    Estimate only n_teams - 1 attack parameters, then reconstruct the final
    attack parameter so that attacks sum to zero automatically.

    Benefits
    --------
    - avoids expensive equality constraints
    - works with L-BFGS-B
    - typically much faster in rolling backtests
    - warm-start friendly
    """

    def __init__(self, xi: float = DIXON_COLES_XI, max_goals: int = DIXON_COLES_MAX_GOALS):
        self.xi = xi
        self.max_goals = max_goals
        self.params_ = None
        self.teams_ = None
        self.team_to_idx_ = None
        self.fitted_ = False
        self.result_ = None
        self._last_theta_ = None

    # ── Validation helpers ───────────────────────────────────────

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit(data) first.")

    def _check_teams(self, home_team: str, away_team: str):
        for team in [home_team, away_team]:
            if team not in self.teams_:
                raise ValueError(
                    f"Unknown team: '{team}'.\nAvailable: {sorted(self.teams_)}"
                )

    # ── Low-level math helpers ───────────────────────────────────

    @staticmethod
    def _log_factorial(k: np.ndarray) -> np.ndarray:
        return gammaln(np.asarray(k) + 1.0)

    @staticmethod
    def _tau_vectorized(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones_like(lam, dtype=float)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return tau

    @staticmethod
    def _tau_scalar(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lam * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        days_ago = (dates.max() - dates).dt.days.to_numpy()
        return np.exp(-self.xi * days_ago)

    # ── Team setup / encoding ────────────────────────────────────

    def _build_teams(self, data: pd.DataFrame):
        self.teams_ = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        self.team_to_idx_ = {team: i for i, team in enumerate(self.teams_)}

    def _encode_matches(self, data: pd.DataFrame):
        home_idx = data["HomeTeam"].map(self.team_to_idx_).to_numpy(dtype=int)
        away_idx = data["AwayTeam"].map(self.team_to_idx_).to_numpy(dtype=int)
        hg = data["FTHG"].to_numpy(dtype=int)
        ag = data["FTAG"].to_numpy(dtype=int)
        return home_idx, away_idx, hg, ag

    # ── Parameter handling ────────────────────────────────────────

    def _theta_size(self) -> int:
        n = len(self.teams_)
        return (n - 1) + n + 2

    def _unpack_theta(self, theta: np.ndarray):
        n = len(self.teams_)
        n_attack_free = n - 1

        attack_free = theta[:n_attack_free]
        defence = theta[n_attack_free:n_attack_free + n]
        home_adv = float(theta[-2])
        rho = float(theta[-1])

        last_attack = -attack_free.sum()
        attack = np.concatenate([attack_free, [last_attack]])
        return attack, defence, home_adv, rho

    def _pack_params_dict(self, theta: np.ndarray) -> dict:
        attack, defence, home_adv, rho = self._unpack_theta(theta)
        return {
            "attack": dict(zip(self.teams_, attack)),
            "defence": dict(zip(self.teams_, defence)),
            "home_adv": home_adv,
            "rho": rho,
        }

    def _theta_from_params_dict(self, params: dict) -> np.ndarray:
        n = len(self.teams_)
        attack = np.array([params["attack"][t] for t in self.teams_], dtype=float)
        defence = np.array([params["defence"][t] for t in self.teams_], dtype=float)
        attack = attack - attack.mean()
        attack_free = attack[:-1]
        return np.concatenate([attack_free, defence, [params["home_adv"]], [params["rho"]]])

    def _initial_guess(self, data: pd.DataFrame, init_params: np.ndarray | None = None) -> np.ndarray:
        n = len(self.teams_)
        expected_size = self._theta_size()

        if init_params is not None:
            init_params = np.asarray(init_params, dtype=float)
            if init_params.shape[0] != expected_size:
                raise ValueError(
                    f"init_params length {init_params.shape[0]} does not match expected {expected_size}"
                )
            return init_params.copy()

        if self._last_theta_ is not None and self._last_theta_.shape[0] == expected_size:
            return self._last_theta_.copy()

        league_avg = (data["FTHG"].mean() + data["FTAG"].mean()) / 2.0

        attack_init = []
        defence_init = []
        for team in self.teams_:
            home_scored = data.loc[data["HomeTeam"] == team, "FTHG"]
            away_scored = data.loc[data["AwayTeam"] == team, "FTAG"]
            home_conceded = data.loc[data["HomeTeam"] == team, "FTAG"]
            away_conceded = data.loc[data["AwayTeam"] == team, "FTHG"]

            avg_scored = pd.concat([home_scored, away_scored]).mean()
            avg_conceded = pd.concat([home_conceded, away_conceded]).mean()

            attack_init.append(np.log(max(avg_scored, 0.5) / league_avg))
            defence_init.append(np.log(max(avg_conceded, 0.5) / league_avg))

        attack_init = np.asarray(attack_init, dtype=float)
        defence_init = np.asarray(defence_init, dtype=float)

        attack_init -= attack_init.mean()
        attack_free = attack_init[:-1]

        return np.concatenate([
            attack_free,
            defence_init,
            [0.25],
            [-0.1],
        ])

    # ── Likelihood ────────────────────────────────────────────────

    def _neg_log_likelihood(
        self,
        theta: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        hg: np.ndarray,
        ag: np.ndarray,
        weights: np.ndarray,
        ridge: float = 0.0,
    ) -> float:
        attack, defence, home_adv, rho = self._unpack_theta(theta)

        lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] - defence[home_idx])

        lam = np.clip(lam, 1e-8, 15.0)
        mu = np.clip(mu, 1e-8, 15.0)

        tau = self._tau_vectorized(hg, ag, lam, mu, rho)
        if np.any(tau <= 0.0) or not np.isfinite(tau).all():
            return 1e12

        log_p = (
            np.log(tau)
            - lam
            + hg * np.log(lam)
            - self._log_factorial(hg)
            - mu
            + ag * np.log(mu)
            - self._log_factorial(ag)
        )

        if not np.isfinite(log_p).all():
            return 1e12

        nll = -np.sum(weights * log_p)

        if ridge > 0.0:
            nll += ridge * (
                np.sum(attack ** 2) +
                np.sum(defence ** 2) +
                home_adv ** 2 +
                rho ** 2
            )

        return float(nll)

    # ── Public API ────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        init_params: np.ndarray | None = None,
        ridge: float = 0.0,
        rho_bounds: tuple[float, float] = (-0.5, 0.0),
        maxiter: int = 500,
    ) -> "DixonColesModel":
        data = data.reset_index(drop=True)
        self._build_teams(data)

        home_idx, away_idx, hg, ag = self._encode_matches(data)
        weights = self._time_weights(data["Date"])
        weights = weights / weights.sum() * len(weights)

        theta0 = self._initial_guess(data, init_params=init_params)

        total_params = self._theta_size()
        bounds = [(None, None)] * total_params
        bounds[-2] = (0.0, 0.5)
        bounds[-1] = rho_bounds

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=theta0,
            args=(home_idx, away_idx, hg, ag, weights, ridge),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )

        if not result.success:
            warnings.warn(f"Optimisation warning: {result.message}")

        self.result_ = result
        self._last_theta_ = result.x.copy()
        self.params_ = self._pack_params_dict(result.x)
        self.fitted_ = True
        return self

    def _expected_goals_arrays(self, home_teams: pd.Series | np.ndarray, away_teams: pd.Series | np.ndarray):
        self._check_fitted()

        home_idx = pd.Series(home_teams).map(self.team_to_idx_)
        away_idx = pd.Series(away_teams).map(self.team_to_idx_)

        if home_idx.isna().any() or away_idx.isna().any():
            unknown = set(pd.Series(home_teams)[home_idx.isna()].tolist()) | set(pd.Series(away_teams)[away_idx.isna()].tolist())
            raise ValueError(f"Unknown team(s) in fixtures: {sorted(unknown)}")

        attack = np.array([self.params_["attack"][t] for t in self.teams_], dtype=float)
        defence = np.array([self.params_["defence"][t] for t in self.teams_], dtype=float)
        home_adv = self.params_["home_adv"]

        h = home_idx.to_numpy(dtype=int)
        a = away_idx.to_numpy(dtype=int)
        lam = np.exp(attack[h] - defence[a] + home_adv)
        mu = np.exp(attack[a] - defence[h])
        lam = np.clip(lam, 1e-8, 15.0)
        mu = np.clip(mu, 1e-8, 15.0)
        return lam, mu

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        self._check_fitted()
        self._check_teams(home_team, away_team)

        lam, mu = self._expected_goals_arrays([home_team], [away_team])
        lam = float(lam[0])
        mu = float(mu[0])
        rho = self.params_["rho"]

        hg = np.arange(self.max_goals + 1)
        ag = np.arange(self.max_goals + 1)
        home_p = np.exp(-lam + hg * np.log(lam) - self._log_factorial(hg))
        away_p = np.exp(-mu + ag * np.log(mu) - self._log_factorial(ag))
        matrix = np.outer(home_p, away_p)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self._tau_scalar(i, j, lam, mu, rho)

        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()

        return MatchPrediction(
            home_win=float(np.tril(matrix, -1).sum()),
            draw=float(np.trace(matrix)),
            away_win=float(np.triu(matrix, 1).sum()),
            exp_home=lam,
            exp_away=mu,
        )

    def predict_batch(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized batch prediction.

        This avoids Python-level loops for expected goals. It still builds one
        score matrix per fixture because each match has different lambdas/mus,
        but the team lookup and expected-goal computation are vectorized.
        """
        self._check_fitted()

        if not {"HomeTeam", "AwayTeam"}.issubset(fixtures.columns):
            raise ValueError("fixtures must contain 'HomeTeam' and 'AwayTeam' columns")

        fixtures = fixtures.reset_index(drop=True).copy()
        lam, mu = self._expected_goals_arrays(fixtures["HomeTeam"], fixtures["AwayTeam"])
        rho = self.params_["rho"]

        goals = np.arange(self.max_goals + 1)
        log_fact = self._log_factorial(goals)

        home_win = np.empty(len(fixtures), dtype=float)
        draw = np.empty(len(fixtures), dtype=float)
        away_win = np.empty(len(fixtures), dtype=float)

        for k, (lk, mk) in enumerate(zip(lam, mu)):
            home_p = np.exp(-lk + goals * np.log(lk) - log_fact)
            away_p = np.exp(-mk + goals * np.log(mk) - log_fact)
            matrix = np.outer(home_p, away_p)

            matrix[0, 0] *= 1.0 - lk * mk * rho
            matrix[0, 1] *= 1.0 + lk * rho
            matrix[1, 0] *= 1.0 + mk * rho
            matrix[1, 1] *= 1.0 - rho

            matrix = np.clip(matrix, 0.0, None)
            matrix /= matrix.sum()

            home_win[k] = np.tril(matrix, -1).sum()
            draw[k] = np.trace(matrix)
            away_win[k] = np.triu(matrix, 1).sum()

        preds = pd.DataFrame({
            "home_win": np.round(home_win, 4),
            "draw": np.round(draw, 4),
            "away_win": np.round(away_win, 4),
            "exp_home": np.round(lam, 4),
            "exp_away": np.round(mu, 4),
        })

        return pd.concat([fixtures, preds], axis=1)

    def get_ratings(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame({
            "team": self.teams_,
            "attack": [self.params_["attack"][t] for t in self.teams_],
            "defence": [self.params_["defence"][t] for t in self.teams_],
        }).sort_values("attack", ascending=False).reset_index(drop=True)

    def __repr__(self):
        status = "fitted" if self.fitted_ else "not fitted"
        teams = len(self.teams_) if self.teams_ else 0
        return (
            f"DixonColesModel(xi={self.xi}, max_goals={self.max_goals}, "
            f"status={status}, teams={teams})"
        )
