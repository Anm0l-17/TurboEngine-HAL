"""Quantile regression for direct prediction interval estimation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class QuantileSurrogate:
    """Quantile-regression surrogate that directly outputs prediction intervals.

    Trains separate models for the median (q=0.5), lower (q=alpha/2), and
    upper (q=1-alpha/2) quantiles for each target, then combines them into
    calibrated prediction intervals.
    """

    def __init__(
        self,
        base_estimator: str = "hist_gradient_boosting",
        alpha: float = 0.1,
        n_estimators: int = 300,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.seed = seed
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.models: dict[str, dict[str, object]] = {}

    def _make_quantile_model(self, quantile: float) -> object:
        """Create a quantile-aware HistGradientBoostingRegressor."""
        if self.base_estimator == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=self.n_estimators,
                max_depth=5,
                random_state=self.seed,
                early_stopping=False,
            )
        raise ValueError(f"Unsupported base estimator: {self.base_estimator}")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "QuantileSurrogate":
        """Fit quantile models for every target at q_low, q_mid, q_high."""
        target_names = y.columns if hasattr(y, "columns") else [f"target_{i}" for i in range(y.shape[1])]
        y_values = y.values if hasattr(y, "values") else np.asarray(y)
        quantiles = {"low": self.alpha / 2, "mid": 0.5, "high": 1.0 - self.alpha / 2}
        for name in quantiles:
            self.models[name] = {}
        for i, tgt in enumerate(target_names):
            for q_name, q_val in quantiles.items():
                model = self._make_quantile_model(q_val)
                model.fit(X, y_values[:, i])
                self.models[q_name][tgt] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return median predictions."""
        target_names = list(self.models["mid"].keys())
        median = np.column_stack([self.models["mid"][t].predict(X) for t in target_names])
        return pd.DataFrame(median, columns=target_names, index=X.index)

    def predict_interval(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (median, lower, upper) prediction DataFrames."""
        target_names = list(self.models["mid"].keys())
        median = np.column_stack([self.models["mid"][t].predict(X) for t in target_names])
        lower = np.column_stack([self.models["low"][t].predict(X) for t in target_names])
        upper = np.column_stack([self.models["high"][t].predict(X) for t in target_names])
        median_df = pd.DataFrame(median, columns=target_names, index=X.index)
        lower_df = pd.DataFrame(lower, columns=target_names, index=X.index)
        upper_df = pd.DataFrame(upper, columns=target_names, index=X.index)
        return median_df, lower_df, upper_df
