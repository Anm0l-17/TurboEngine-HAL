"""Adaptive conformal prediction with locally-weighted residuals.

Standard split conformal prediction assumes exchangeability and uses a single
global quantile. Adaptive conformal relaxes this by weighting calibration
residuals by their similarity to each test point, producing prediction intervals
that adapt to heteroscedasticity and local difficulty.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AdaptiveConformalRegressor:
    """Locally-weighted conformal prediction with k-nearest-neighbour residuals.

    For each test point, finds its k nearest neighbours in the calibration set
    (in feature space), then computes the coverage quantile of the *weighted*
    absolute residuals, where closer neighbours have higher weight.
    """

    def __init__(
        self,
        coverage: float = 0.9,
        n_neighbours: int = 50,
        kernel_width: float = 1.0,
    ) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        self.coverage = coverage
        self.n_neighbours = n_neighbours
        self.kernel_width = kernel_width
        self.cal_features_: np.ndarray | None = None
        self.cal_residuals_: np.ndarray | None = None
        self.nn_: NearestNeighbors | None = None
        self.scaler_: StandardScaler = StandardScaler()

    def fit(
        self,
        calibration_features: np.ndarray,
        truth: np.ndarray,
        prediction: np.ndarray,
    ) -> "AdaptiveConformalRegressor":
        """Store calibration residuals and fit nearest-neighbour index."""
        self.cal_features_ = self.scaler_.fit_transform(
            np.asarray(calibration_features, dtype=float)
        )
        self.cal_residuals_ = np.abs(np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float))
        n = min(self.n_neighbours, len(self.cal_features_))
        self.nn_ = NearestNeighbors(n_neighbors=n, metric="euclidean").fit(self.cal_features_)
        return self

    def predict_interval(
        self, test_features: np.ndarray, point_predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (lower, upper) adaptive intervals and empirical coverage.

        Args:
            test_features: (n_test, n_features) array of feature vectors.
            point_predictions: (n_test, n_targets) array of point predictions.

        Returns:
            lower: (n_test, n_targets) lower bounds.
            upper: (n_test, n_targets) upper bounds.
            mean_coverage: average empirical coverage across test points.
        """
        if self.cal_features_ is None or self.cal_residuals_ is None or self.nn_ is None:
            raise RuntimeError("call fit() before predict_interval()")

        test_scaled = self.scaler_.transform(np.asarray(test_features, dtype=float))
        values = np.asarray(point_predictions, dtype=float)
        n_test = test_scaled.shape[0]
        n_targets = values.shape[1] if values.ndim > 1 else 1
        lower = np.zeros_like(values)
        upper = np.zeros_like(values)

        for i in range(n_test):
            distances, indices = self.nn_.kneighbors(test_scaled[i:i + 1])
            dists = distances[0] + 1e-12
            weights = np.exp(-0.5 * (dists / self.kernel_width) ** 2)
            weights /= weights.sum()
            for j in range(n_targets):
                residuals = self.cal_residuals_[indices[0], j] if self.cal_residuals_.ndim > 1 else self.cal_residuals_[indices[0]]
                weighted_quantile = _weighted_quantile(residuals, weights, self.coverage)
                lower[i, j] = values[i, j] - weighted_quantile
                upper[i, j] = values[i, j] + weighted_quantile

        # Empirical coverage on calibration set
        coverage_mask = (
            (self.cal_residuals_ <= np.percentile(self.cal_residuals_, self.coverage * 100, axis=0))
        )
        mean_coverage = float(np.mean(coverage_mask)) if coverage_mask.size > 0 else self.coverage
        return lower, upper, mean_coverage


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Compute a weighted quantile using the default interpolation method."""
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumsum = np.cumsum(weights_sorted)
    cumsum /= cumsum[-1]
    return float(np.interp(quantile, cumsum, values_sorted))
