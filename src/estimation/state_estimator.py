"""Engine-health state estimator facade."""

import numpy as np
from .ekf import ExtendedKalmanFilter


class StateEstimator:
    """EKF for compressor, combustor, turbine, and overall health."""

    def __init__(self) -> None:
        self.filter = ExtendedKalmanFilter(
            np.ones(4), np.eye(4) * 0.02, np.eye(4) * 1e-5, np.eye(4) * 0.01
        )

    def update(self, health_observation: np.ndarray) -> np.ndarray:
        """Assimilate a health observation with monotonic slow degradation."""
        identity = np.eye(4)
        self.filter.predict(lambda x: np.clip(x - 1e-4, 0, 1), identity)
        state = self.filter.update(np.clip(health_observation, 0, 1), lambda x: x, identity)
        return np.clip(state, 0, 1)
