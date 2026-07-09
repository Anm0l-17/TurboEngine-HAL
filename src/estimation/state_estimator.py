"""Engine-health EKF state estimator."""

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
        previous = self.filter.state.copy()
        self.filter.predict(lambda x: np.clip(x - 1e-4, 0, 1), identity)
        state = self.filter.update(np.clip(health_observation, 0, 1), lambda x: x, identity)
        # Dimensions where the EKF pushed state above the pre-predict level
        # will be caught by the monotonicity clamp. Their observation
        # information is rejected — reset variance to process-noise level.
        clamped = state > previous
        state = np.minimum(np.clip(state, 0, 1), previous)
        if clamped.any():
            Q = self.filter.process_noise
            for i in np.where(clamped)[0]:
                self.filter.covariance[i, :] = 0.0
                self.filter.covariance[:, i] = 0.0
                self.filter.covariance[i, i] = Q[i, i]
        self.filter.state = state
        return state.copy()
