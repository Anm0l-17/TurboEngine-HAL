"""Scaled unscented Kalman filter."""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class UnscentedKalmanFilter:
    """Derivative-free nonlinear Gaussian state estimator."""

    def __init__(
        self,
        state: Vector,
        covariance: Vector,
        process_noise: Vector,
        alpha: float = 1e-2,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.state = np.asarray(state, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

    def _points(self) -> tuple[Vector, Vector, Vector]:
        n = self.state.size
        lam = self.alpha**2 * (n + self.kappa) - n
        root = np.linalg.cholesky((n + lam) * (self.covariance + np.eye(n) * 1e-12))
        points = np.vstack([self.state, self.state + root.T, self.state - root.T])
        wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + 1 - self.alpha**2 + self.beta
        return points, wm, wc

    def predict(self, transition: Callable[[Vector], Vector]) -> Vector:
        """Propagate sigma points through transition."""
        points, wm, wc = self._points()
        transformed = np.asarray([transition(point) for point in points])
        self.state = wm @ transformed
        delta = transformed - self.state
        self.covariance = (delta.T * wc) @ delta + self.process_noise
        return self.state.copy()

    def update(
        self,
        measurement: Vector,
        observation: Callable[[Vector], Vector],
        measurement_noise: Vector,
    ) -> Vector:
        """Assimilate a nonlinear observation."""
        points, wm, wc = self._points()
        observed = np.asarray([observation(point) for point in points])
        expected = wm @ observed
        dy = observed - expected
        dx = points - self.state
        innovation_cov = (dy.T * wc) @ dy + measurement_noise
        cross = (dx.T * wc) @ dy
        gain = np.linalg.solve(innovation_cov, cross.T).T
        self.state += gain @ (np.asarray(measurement) - expected)
        self.covariance -= gain @ innovation_cov @ gain.T
        return self.state.copy()
