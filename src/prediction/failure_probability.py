"""Calibratable failure-horizon risk model."""

import json
import math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression


_FALLBACK_HEALTH_COEF = 12.0
_FALLBACK_HORIZON_COEF = 5.0


class FailureProbabilityCalibrator:
    """Logistic risk model calibrated on historical degradation data.

    Fits P(failure) = sigmoid(a * (threshold - health) + b * horizon_term)
    where horizon_term = (horizon - remaining) / horizon.
    """

    def __init__(
        self,
        health_coef: float = _FALLBACK_HEALTH_COEF,
        horizon_coef: float = _FALLBACK_HORIZON_COEF,
    ) -> None:
        self.health_coef = health_coef
        self.horizon_coef = horizon_coef

    def fit(self, health_history: list[float], horizon: int = 25, threshold: float = 0.7) -> "FailureProbabilityCalibrator":
        """Fit coefficients from engine degradation trajectories.

        For each point, labels 1 if health falls below *threshold* within
        *horizon* cycles of that point, else 0. Learns the logistic
        coefficients from the population of all engines.
        """
        if len(health_history) < 10:
            return self
        arr = np.asarray(health_history, dtype=float)
        x_health = np.maximum(threshold - arr, 0.0).reshape(-1, 1)
        features = np.column_stack([x_health, np.zeros_like(x_health)])
        labels = np.zeros(len(arr), dtype=float)
        for t in range(len(arr)):
            future = arr[t + 1 : t + 1 + horizon]
            if len(future) > 0 and future.min() < threshold:
                labels[t] = 1.0
            features[t, 1] = max(0.0, (horizon - (len(arr) - 1 - t)) / horizon)
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count
        if pos_count < 5 or neg_count < 5:
            return self
        model = LogisticRegression(C=1.0, fit_intercept=False, random_state=42)
        model.fit(features, labels)
        self.health_coef = float(model.coef_[0, 0]) if model.coef_[0, 0] > 0 else _FALLBACK_HEALTH_COEF
        self.horizon_coef = float(model.coef_[0, 1]) if model.coef_[0, 1] > 0 else _FALLBACK_HORIZON_COEF
        return self

    def save(self, path: str | Path) -> None:
        """Persist coefficients as JSON."""
        Path(path).write_text(
            json.dumps({"health_coef": self.health_coef, "horizon_coef": self.horizon_coef}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FailureProbabilityCalibrator":
        """Restore coefficients from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(data["health_coef"]), float(data["horizon_coef"]))


def failure_probability(
    health: float,
    remaining_cycles: float,
    horizon_cycles: float = 25,
    threshold: float = 0.3,
    calibrator: FailureProbabilityCalibrator | None = None,
) -> float:
    """Combine health margin and RUL horizon in a bounded logistic risk score."""
    hc = calibrator.health_coef if calibrator is not None else _FALLBACK_HEALTH_COEF
    hzc = calibrator.horizon_coef if calibrator is not None else _FALLBACK_HORIZON_COEF
    health_term = max(threshold - health, 0.0) * hc
    horizon_term = (horizon_cycles - remaining_cycles) / max(horizon_cycles, 1) * hzc
    score = max(min(health_term + horizon_term, 40), -40)
    return 1 / (1 + math.exp(-score))
