"""Numerically stable extended Kalman filter."""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class ExtendedKalmanFilter:
    """First-order nonlinear Gaussian state estimator."""

    def __init__(
        self, state: Vector, covariance: Matrix, process_noise: Matrix, measurement_noise: Matrix
    ) -> None:
        self.state = np.asarray(state, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.measurement_noise = np.asarray(measurement_noise, dtype=float)

    def predict(self, transition: Callable[[Vector], Vector], jacobian: Matrix) -> Vector:
        """Advance the distribution through a nonlinear transition."""
        self.state = np.asarray(transition(self.state), dtype=float)
        self.covariance = jacobian @ self.covariance @ jacobian.T + self.process_noise
        return self.state.copy()

    def update(
        self, measurement: Vector, observation: Callable[[Vector], Vector], jacobian: Matrix
    ) -> Vector:
        """Assimilate a measurement using Joseph covariance stabilization."""
        innovation = np.asarray(measurement) - observation(self.state)
        innovation_cov = jacobian @ self.covariance @ jacobian.T + self.measurement_noise
        gain = np.linalg.solve(innovation_cov, jacobian @ self.covariance).T
        self.state = self.state + gain @ innovation
        identity = np.eye(self.state.size)
        residual = identity - gain @ jacobian
        self.covariance = (
            residual @ self.covariance @ residual.T + gain @ self.measurement_noise @ gain.T
        )
        return self.state.copy()
