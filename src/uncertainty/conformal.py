"""Distribution-free split conformal prediction."""

import numpy as np
from numpy.typing import NDArray


class ConformalRegressor:
    """Per-target symmetric conformal interval calibrator."""

    def __init__(self, coverage: float = 0.9) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        self.coverage = coverage
        self.quantile_: NDArray[np.float64] | None = None

    def fit(
        self, truth: NDArray[np.float64], prediction: NDArray[np.float64]
    ) -> "ConformalRegressor":
        """Calibrate absolute residual quantiles."""
        residual = np.abs(np.asarray(truth) - np.asarray(prediction))
        n = residual.shape[0]
        q = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.quantile_ = np.quantile(residual, q, axis=0, method="higher")
        return self

    def predict_interval(
        self, prediction: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return lower and upper calibrated prediction bounds."""
        if self.quantile_ is None:
            raise RuntimeError("calibrator has not been fitted")
        values = np.asarray(prediction)
        return values - self.quantile_, values + self.quantile_
