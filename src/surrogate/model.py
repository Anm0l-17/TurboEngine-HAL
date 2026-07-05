"""Serializable multi-output surrogate with multiple uncertainty strategies."""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.dataset.features import engineer_all_features
from src.uncertainty.conformal import ConformalRegressor
from src.uncertainty.quantile import QuantileSurrogate


class SurrogateModel:
    """Column-aware wrapper with configurable uncertainty strategy.

    Supported uncertainty modes:
    - ``conformal``: split conformal prediction (distribution-free, per-target symmetric)
    - ``quantile``: quantile regression (direct lower/upper estimates)
    - ``none``: point estimates only

    If ``target_scalers`` are provided, targets are scaled before fitting and
    inverse-scaled after prediction.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        target_names: list[str],
        pipeline_feature_names: list[str] | None = None,
        calibrator: ConformalRegressor | None = None,
        target_scalers: dict[str, StandardScaler] | None = None,
        quantile_model: QuantileSurrogate | None = None,
        uncertainty_mode: str = "conformal",
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.target_names = target_names
        self.pipeline_feature_names = pipeline_feature_names or feature_names
        self.calibrator = calibrator
        self.target_scalers = target_scalers or {}
        self.quantile_model = quantile_model
        self.uncertainty_mode = uncertainty_mode
        self._feature_importances_: np.ndarray | None = None
        self._calibration_features_: np.ndarray | None = None
        self._calibration_targets_: np.ndarray | None = None

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return engineer_all_features(frame[self.feature_names])

    def _scale_targets(self, values: np.ndarray) -> np.ndarray:
        if not self.target_scalers:
            return values
        out = values.copy()
        for i, name in enumerate(self.target_names):
            scaler = self.target_scalers.get(name)
            if scaler is not None:
                out[:, i] = scaler.transform(out[:, i: i + 1]).ravel()
        return out

    def _unscale_targets(self, values: np.ndarray) -> np.ndarray:
        if not self.target_scalers:
            return values
        out = values.copy()
        for i, name in enumerate(self.target_names):
            scaler = self.target_scalers.get(name)
            if scaler is not None:
                out[:, i] = scaler.inverse_transform(out[:, i: i + 1]).ravel()
        return out

    def _postprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        out = values.copy()
        for column in out.columns:
            if column.endswith("Health"):
                out[column] = out[column].clip(0.0, 1.0)
            elif column in {"Thrust", "TSFC"}:
                out[column] = out[column].clip(lower=0.0)
        return out

    def fit(self, frame: pd.DataFrame) -> "SurrogateModel":
        targets = frame[self.target_names].to_numpy()
        if self.target_scalers:
            for i, name in enumerate(self.target_names):
                scaler = self.target_scalers.get(name)
                if scaler is not None:
                    scaler.fit(targets[:, i: i + 1])
        scaled = self._scale_targets(targets)
        prepared = self._prepare(frame)
        self.pipeline.fit(prepared, scaled)

        # Store feature importances if available
        estimator = self.pipeline.steps[-1][1] if hasattr(self.pipeline, "steps") else self.pipeline
        if hasattr(estimator, "feature_importances_"):
            self._feature_importances_ = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            self._feature_importances_ = np.asarray(estimator.coef_).ravel()

        # Fit quantile model if configured
        if self.uncertainty_mode == "quantile":
            self.quantile_model = QuantileSurrogate(alpha=0.1, seed=42).fit(prepared, frame[self.target_names])

        return self

    def calibrate(self, frame: pd.DataFrame, coverage: float = 0.9) -> "SurrogateModel":
        """Fit split-conformal intervals and store calibration set for adaptive methods."""
        prepared = self._prepare(frame)
        prediction = self.predict(frame)
        self.calibrator = ConformalRegressor(coverage).fit(
            frame[self.target_names].to_numpy(), prediction.to_numpy()
        )
        self._calibration_features_ = prepared.values
        self._calibration_targets_ = frame[self.target_names].values
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare(frame)
        values = np.asarray(self.pipeline.predict(prepared))
        values = self._unscale_targets(values)
        prediction = pd.DataFrame(values, columns=self.target_names, index=frame.index)
        return self._postprocess(prediction)

    def predict_with_uncertainty(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        """Return point, lower, upper, and calibrated coverage.

        Uses the configured uncertainty_mode:
        - ``conformal``: split-conformal symmetric intervals
        - ``quantile``: quantile-regression intervals
        - ``none``: degenerate zero-width intervals
        """
        prepared = self._prepare(frame)
        prediction = self.predict(frame)

        if self.uncertainty_mode == "quantile" and self.quantile_model is not None:
            _, lower_raw, upper_raw = self.quantile_model.predict_interval(prepared)
            lower = self._postprocess(
                pd.DataFrame(lower_raw, columns=self.target_names, index=frame.index)
            )
            upper = self._postprocess(
                pd.DataFrame(upper_raw, columns=self.target_names, index=frame.index)
            )
            return prediction, lower, upper, 1.0 - self.quantile_model.alpha

        if self.calibrator is not None:
            lower_values, upper_values = self.calibrator.predict_interval(prediction.to_numpy())
            lower = self._postprocess(
                pd.DataFrame(lower_values, columns=self.target_names, index=frame.index)
            )
            upper = self._postprocess(
                pd.DataFrame(upper_values, columns=self.target_names, index=frame.index)
            )
            return prediction, lower, upper, self.calibrator.coverage

        return prediction, prediction.copy(), prediction.copy(), 0.0

    def predict_ensemble(
        self, frame: pd.DataFrame, n_members: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (mean, std, lower, upper) via bootstrapped ensemble.

        Creates n_members predictions by adding small noise to the prepared
        features and re-predicting, approximating ensemble uncertainty.
        """
        prepared = self._prepare(frame)
        base_preds = []
        rng = np.random.default_rng(42)
        for _ in range(n_members):
            noise = rng.normal(0, 0.01, size=prepared.values.shape)
            X_noisy = pd.DataFrame(
                prepared.values * (1.0 + noise),
                columns=prepared.columns,
                index=prepared.index,
            )
            noisy_targets = np.asarray(self.pipeline.predict(X_noisy))
            base_preds.append(noisy_targets)
        stacked = np.stack(base_preds, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=1)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        mean_df = pd.DataFrame(self._unscale_targets(mean), columns=self.target_names, index=frame.index)
        std_df = pd.DataFrame(std, columns=[f"{t}_std" for t in self.target_names], index=frame.index)
        lower_df = pd.DataFrame(self._unscale_targets(lower), columns=self.target_names, index=frame.index)
        upper_df = pd.DataFrame(self._unscale_targets(upper), columns=self.target_names, index=frame.index)
        return self._postprocess(mean_df), std_df, self._postprocess(lower_df), self._postprocess(upper_df)

    def explain(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Return feature importance and per-prediction explanation."""
        explained: dict[str, Any] = {"method": "feature_importances"}
        if self._feature_importances_ is not None:
            names = self.pipeline_feature_names
            importance_vals = self._feature_importances_.ravel()
            if len(importance_vals) == len(names):
                pairs = sorted(zip(names, importance_vals), key=lambda x: -abs(x[1]))
                explained["global_importance"] = [
                    {"feature": n, "importance": float(v)} for n, v in pairs
                ]
        if self._calibration_features_ is not None and len(frame) > 0:
            prepared = self._prepare(frame)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
            nn.fit(self._calibration_features_)
            dists, idxs = nn.kneighbors(prepared.values[:1])
            explained["nearest_calibration"] = {
                "mean_distance": float(dists.mean()),
                "n_calibration_points": len(self._calibration_features_),
            }
        return explained

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the model configuration."""
        return {
            "uncertainty_mode": self.uncertainty_mode,
            "targets": self.target_names,
            "n_features": len(self.pipeline_feature_names),
            "feature_names": self.pipeline_feature_names,
            "has_calibrator": self.calibrator is not None,
            "has_quantile": self.quantile_model is not None,
            "has_target_scalers": len(self.target_scalers) > 0,
            "pipeline_steps": [str(s) for s in self.pipeline.steps] if hasattr(self.pipeline, "steps") else [],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        model: Any = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a SurrogateModel")
        for attr in ["pipeline_feature_names", "calibrator", "target_scalers",
                     "quantile_model", "uncertainty_mode", "_feature_importances_",
                     "_calibration_features_", "_calibration_targets_"]:
            if not hasattr(model, attr):
                setattr(model, attr, None if attr != "uncertainty_mode" else "conformal")
                if attr == "target_scalers":
                    setattr(model, attr, {})
                if attr == "uncertainty_mode":
                    setattr(model, attr, "conformal")
        return model
