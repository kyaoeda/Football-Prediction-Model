# models/xgb_model.py
# ─────────────────────────────────────────────────────────────────
# XGBoost 1X2 result classifier.
#
# Predicts match outcome from pre-match features (e.g. rolling xG
# form). Outcome encoding matches the rest of the project:
#     0 = away win, 1 = draw, 2 = home win
# so predict_proba columns line up as [P(away), P(draw), P(home)].
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def result_target(df: pd.DataFrame) -> np.ndarray:
    """Encode FTHG/FTAG into 0=away, 1=draw, 2=home."""
    hg = df["FTHG"].to_numpy()
    ag = df["FTAG"].to_numpy()
    return np.where(hg > ag, 2, np.where(hg == ag, 1, 0))


class XGBResultModel:
    """
    Thin wrapper around XGBClassifier for 1X2 prediction.

    Parameters
    ----------
    features : list[str]
        Column names to use as inputs. NaNs are allowed — XGBoost
        learns a default split direction for missing values.
    Remaining kwargs are passed through to XGBClassifier; the defaults
    below are conservative to limit overfitting on ~1.5k matches.
    """

    def __init__(self,
                 features: list[str],
                 n_estimators: int = 300,
                 max_depth: int = 3,
                 learning_rate: float = 0.03,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_weight: float = 5.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):

        self.features = list(features)
        self.model = XGBClassifier(
            objective        = "multi:softprob",
            num_class        = 3,
            eval_metric      = "mlogloss",
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            reg_lambda       = reg_lambda,
            random_state     = random_state,
            n_jobs           = -1,
        )
        self.fitted_ = False

    # ── Training ──────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "XGBResultModel":
        X = train[self.features]
        y = result_target(train)
        self.model.fit(X, y)
        self.fitted_ = True
        return self

    # ── Prediction ────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of shape (n, 3): columns [away, draw, home]."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit(train) first.")
        # XGBClassifier orders columns by sorted class label -> [0,1,2]
        return self.model.predict_proba(df[self.features])

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach model probabilities to the input rows using the same
        column names the rest of the pipeline expects.
        """
        probs = self.predict_proba(df)
        out = df.copy()
        out["away_win"] = np.round(probs[:, 0], 4)
        out["draw"]     = np.round(probs[:, 1], 4)
        out["home_win"] = np.round(probs[:, 2], 4)
        return out

    # ── Introspection ─────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based feature importance, most important first."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit(train) first.")
        booster = self.model.get_booster()
        gain = booster.get_score(importance_type="gain")
        # When fit on a DataFrame, the booster keys features by column name;
        # fall back to the f0/f1 positional names otherwise.
        rows = []
        for i, feat in enumerate(self.features):
            g = gain.get(feat, gain.get(f"f{i}", 0.0))
            rows.append({"feature": feat, "gain": round(g, 2)})
        return (pd.DataFrame(rows)
                .sort_values("gain", ascending=False)
                .reset_index(drop=True))
