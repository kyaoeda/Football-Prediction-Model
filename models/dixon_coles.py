import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
from config import DIXON_COLES_XI, DIXON_COLES_MAX_GOALS

@dataclass
class MatchPrediction:
    """
    Clean container for model predictions.
    Returned by every call to .predict()
    """
    home_win  : float
    draw      : float
    away_win  : float
    exp_home  : float    # lambda — expected home goals
    exp_away  : float    # mu    — expected away goals

    def __repr__(self):
        """
        What prints when you do: print(prediction)
        """
        return (
            f"Home Win : {self.home_win:.1%}\n"
            f"Draw     : {self.draw:.1%}\n"
            f"Away Win : {self.away_win:.1%}\n"
            f"Exp Home : {self.exp_home:.2f}\n"
            f"Exp Away : {self.exp_away:.2f}"
        )

    def to_dict(self):
        """
        Convert to dictionary — useful later for
        storing predictions in a DataFrame.
        """
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

    Estimates per-team attack and defence parameters
    using Maximum Likelihood Estimation (MLE), with
    optional exponential time decay weighting and
    low-score correction via the rho parameter.

    Parameters
    ----------
    xi        : float
        Time decay rate.
        0.0     = all matches weighted equally
        0.00325 = optimal for 5+ seasons of data
                  (roughly halves weight every 6 months)

    max_goals : int
        Maximum goals considered in the scoreline matrix.
        10 is sufficient — P(X > 10) is negligible.

    Example
    -------
    >>> model = DixonColesModel(xi=0.00325)
    >>> model.fit(df)
    >>> pred = model.predict('Arsenal', 'Chelsea')
    >>> print(pred)
    Home Win : 48.2%
    Draw     : 26.1%
    Away Win : 25.7%
    """

    def __init__(self, xi: float = DIXON_COLES_XI, max_goals: int = DIXON_COLES_MAX_GOALS):

        # ── Hyperparameters (set by user) ──────────────────────────
        self.xi        = xi
        self.max_goals = max_goals

        # ── Learned attributes (set by .fit()) ────────────────────
        # These don't exist until the model is trained.
        # Convention: trailing underscore = set during fitting.
        self.params_   = None   # dict of attack/defence/home_adv/rho
        self.teams_    = None   # sorted list of all teams
        self.fitted_   = False  # guard flag

    def _check_fitted(self):
        """
        Guard method — called at the start of predict().
        Raises a clear error if user forgets to call .fit() first.

        The underscore prefix _ signals this is a private method —
        it's internal machinery, not part of the public interface.
        """
        if not self.fitted_:
            raise RuntimeError(
                "Model is not fitted yet. Call .fit(data) first."
            )

    def _check_teams(self, home_team: str, away_team: str):
        """
        Validates that both teams were in the training data.
        Raises a clear, informative error if not.
        """
        for team in [home_team, away_team]:
            if team not in self.teams_:
                raise ValueError(
                    f"Unknown team: '{team}'.\n"
                    f"Available teams: {sorted(self.teams_)}"
                )

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Compute exponential decay weights for each match.

        Recent matches get weight close to 1.0.
        Older matches get weight close to 0.0.

        Formula: w = exp(-xi * days_ago)

        If xi = 0.0, all weights = 1.0 (no decay).
        """
        days_ago = (dates.max() - dates).dt.days
        return np.exp(-self.xi * days_ago)

    def _tau(self, x: int, y: int,
             lam: float, mu: float,
             rho: float) -> float:
        """
        Dixon-Coles low-score correction factor.

        Adjusts the probability of four scorelines
        that Poisson systematically mispredicts:
            (0,0), (1,0), (0,1), (1,1)

        rho < 0 : reduces 0-0 and 1-1, increases 1-0 and 0-1
        rho = 0 : no correction (pure Poisson)

        Parameters
        ----------
        x   : home goals
        y   : away goals
        lam : expected home goals
        mu  : expected away goals
        rho : correlation parameter (typically ~ -0.13)
        """
        if   x == 0 and y == 0: return 1 - lam * mu * rho
        elif x == 0 and y == 1: return 1 + lam * rho
        elif x == 1 and y == 0: return 1 + mu  * rho
        elif x == 1 and y == 1: return 1 - rho
        else                  : return 1.0

    def _unpack_params(self, params: np.ndarray) -> dict:
        """
        Convert the flat parameter array (what scipy.minimize
        works with) into a named dictionary (what we work with).

        Parameter array layout:
        [ attack_0, attack_1, ..., attack_n,
          defence_0, defence_1, ..., defence_n,
          home_advantage,
          rho ]
        """
        n = len(self.teams_)
        return {
            'attack'  : dict(zip(self.teams_, params[:n])),
            'defence' : dict(zip(self.teams_, params[n:2*n])),
            'home_adv': float(params[-2]),
            'rho'     : float(params[-1])
        }

    def _neg_log_likelihood(self, params: np.ndarray,
                         data: pd.DataFrame,
                         weights: np.ndarray) -> float:
        """
        Vectorised version — much faster than iterrows().
        Computes log likelihood for all matches simultaneously.
        """
        p = self._unpack_params(params)

        # ── Vectorised expected goals ──────────────────────────────
        # Map team names to parameter values for entire column at once
        attack_home  = data['HomeTeam'].map(p['attack']).values
        defence_home = data['HomeTeam'].map(p['defence']).values
        attack_away  = data['AwayTeam'].map(p['attack']).values
        defence_away = data['AwayTeam'].map(p['defence']).values

        lam = np.exp(attack_home - defence_away + p['home_adv'])
        mu  = np.exp(attack_away - defence_home)
        
        # ── Guard 1: clip expected goals to sensible range ─────────────
        # Prevents exp() exploding to inf
        lam = np.clip(lam, 1e-6, 15.0)
        mu  = np.clip(mu,  1e-6, 15.0)

        hg  = data['FTHG'].values.astype(int)
        ag  = data['FTAG'].values.astype(int)

        # ── Vectorised Poisson log probabilities ───────────────────
        log_p_home = poisson.logpmf(hg, lam)
        log_p_away = poisson.logpmf(ag, mu)
    
        # ── Guard 2: clip log probs to prevent -inf ────────────────────
        log_p_home = np.clip(log_p_home, -20, 0)
        log_p_away = np.clip(log_p_away, -20, 0)

        # ── Vectorised tau correction ──────────────────────────────
        # Compute tau for every match simultaneously
        rho  = p['rho']
        tau  = np.ones(len(data))

        # Each condition applied as a boolean mask
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10]  * rho
        tau[m11] = 1 - rho

        # Guard against invalid tau values
        if np.any(tau <= 0):
            return np.inf

        # ── Weighted log likelihood ────────────────────────────────
        ll = weights * (
            np.log(tau)
            + log_p_home
            + log_p_away
        )

        return -ll.sum()

    def fit(self, data: pd.DataFrame) -> 'DixonColesModel':
        """
        Fit the model to historical match data.

        Finds the attack/defence parameters for every team
        that maximise the likelihood of the observed scorelines.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns:
                'HomeTeam' : str  — home team name
                'AwayTeam' : str  — away team name
                'FTHG'     : int  — full time home goals
                'FTAG'     : int  — full time away goals
                'Date'     : datetime — match date

        Returns
        -------
        self — allows method chaining: model.fit(df).predict(...)
        """

        # Reset index so weights align correctly with iterrows()
        data = data.reset_index(drop=True)

        # All unique teams, sorted alphabetically for consistency
        self.teams_ = sorted(
            set(data['HomeTeam']) | set(data['AwayTeam'])
        )
        n = len(self.teams_)

        # ── Initial parameter guess ────────────────────────────────
        # All attack/defence start at 1.0 (neutral)
        # home_adv starts at 0.25 (reasonable prior)
        # rho starts at -0.1 (slight negative correlation)
        x0 = np.concatenate([
            np.ones(n),    # attack parameters
            np.ones(n),    # defence parameters
            [0.25],        # home advantage
            [-0.1]         # rho
        ])

        # ── Time decay weights ─────────────────────────────────────
        weights = self._time_weights(data['Date'])

        # Normalise so weights average to 1.0
        # (keeps log likelihood on same scale regardless of xi)
        weights = weights / weights.sum() * len(weights)

        # ── Add bounds ─────────────────────────────────────────────────
        # Prevents optimiser wandering into numerical nonsense
        attack_bounds  = [(-3.0, 3.0)] * n    # attack params
        defence_bounds = [(-3.0, 3.0)] * n    # defence params
        home_bounds    = [(0.0,  1.0)]        # home advantage
        rho_bounds     = [(-1.0, 0.0)]        # rho (always negative)

        bounds = attack_bounds + defence_bounds + home_bounds + rho_bounds

        # ── Optimisation ───────────────────────────────────────────
        result = minimize(
            fun     = self._neg_log_likelihood,
            x0      = x0,
            args    = (data, weights),
            method  = 'L-BFGS-B',   # efficient for many parameters
            bounds  = bounds, 
            options = {'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            warnings.warn(
                f"Optimisation may not have converged:\n{result.message}"
            )

        # Store learned parameters
        self.params_ = self._unpack_params(result.x)
        self.fitted_ = True

        return self   # ← enables method chaining

    def predict(self, home_team: str,
                away_team: str) -> MatchPrediction:
        """
        Predict match outcome probabilities.

        Builds a full (max_goals x max_goals) scoreline
        probability matrix, applies Dixon-Coles correction,
        then sums regions for home win / draw / away win.

        Parameters
        ----------
        home_team : str — must have been in training data
        away_team : str — must have been in training data

        Returns
        -------
        MatchPrediction dataclass
        """
        # Guards — fail loudly with clear messages
        self._check_fitted()
        self._check_teams(home_team, away_team)

        p = self.params_

        # Expected goals for each team
        lam = np.exp(
            p['attack'][home_team]
            - p['defence'][away_team]
            + p['home_adv']
        )
        mu = np.exp(
            p['attack'][away_team]
            - p['defence'][home_team]
        )

        # ── Build scoreline probability matrix ─────────────────────
        # matrix[i][j] = P(home scores i, away scores j)
        home_probs = [poisson.pmf(i, lam) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(j, mu)  for j in range(self.max_goals + 1)]

        # outer product gives joint probabilities
        # (assumes independence before DC correction)
        matrix = np.outer(home_probs, away_probs)

        # ── Apply Dixon-Coles correction ───────────────────────────
        # Only affects the 2x2 top-left corner of the matrix
        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self._tau(i, j, lam, mu, p['rho'])

        # Renormalise — correction slightly shifts total probability
        matrix /= matrix.sum()

        # ── Sum regions of matrix ──────────────────────────────────
        #
        #        Away goals →
        #        0    1    2    3
        #   0  [draw  A    A    A ]
        #   1  [ H   draw  A    A ]  ← Home goals
        #   2  [ H    H   draw  A ]
        #   3  [ H    H    H   draw]
        #
        # np.tril(matrix, -1) = lower triangle (home wins)
        # np.trace(matrix)    = diagonal (draws)
        # np.triu(matrix,  1) = upper triangle (away wins)

        home_win = float(np.tril(matrix, -1).sum())
        draw     = float(np.trace(matrix))
        away_win = float(np.triu(matrix,  1).sum())

        return MatchPrediction(
            home_win = home_win,
            draw     = draw,
            away_win = away_win,
            exp_home = lam,
            exp_away = mu
        )

    def get_ratings(self) -> pd.DataFrame:
        """
        Return team attack and defence ratings as a DataFrame.
        Sorted by attack strength descending.

        Higher attack  = more goals scored
        Lower defence  = fewer goals conceded
                         (note: lower is BETTER for defence
                          in this parameterisation)
        """
        self._check_fitted()

        df = pd.DataFrame({
            'team'   : self.teams_,
            'attack' : [self.params_['attack'][t]  for t in self.teams_],
            'defence': [self.params_['defence'][t] for t in self.teams_]
        })

        return df.sort_values('attack', ascending=False)\
                 .reset_index(drop=True)

    def predict_batch(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Predict multiple fixtures at once.

        Parameters
        ----------
        fixtures : pd.DataFrame
            Must contain 'HomeTeam' and 'AwayTeam' columns.

        Returns
        -------
        Original DataFrame with prediction columns added.

        Example
        -------
        >>> upcoming = pd.DataFrame({
        ...     'HomeTeam': ['Arsenal', 'Liverpool'],
        ...     'AwayTeam': ['Chelsea', 'Man City']
        ... })
        >>> model.predict_batch(upcoming)
        """
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
        """
        What prints when you type the model object in the console.
        """
        status = "fitted" if self.fitted_ else "not fitted"
        teams  = len(self.teams_) if self.teams_ else 0
        return (
            f"DixonColesModel("
            f"xi={self.xi}, "
            f"max_goals={self.max_goals}, "
            f"status={status}, "
            f"teams={teams})"
        )