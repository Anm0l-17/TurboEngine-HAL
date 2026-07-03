"""Online drift monitoring and recalibration policy."""

from collections import deque
import numpy as np


class DriftMonitor:
    """Windowed normalized residual drift detector."""

    def __init__(self, threshold: float = 0.12, window: int = 50) -> None:
        self.threshold = threshold
        self.residuals: deque[float] = deque(maxlen=window)
        self.calibration_history: list[dict[str, float | bool]] = []

    def update(self, physics_value: float, surrogate_value: float) -> bool:
        """Record disagreement and signal sustained drift."""
        residual = abs(physics_value - surrogate_value) / max(abs(physics_value), 1e-9)
        self.residuals.append(residual)
        drift = (
            len(self.residuals) == self.residuals.maxlen
            and np.mean(self.residuals) > self.threshold
        )
        self.calibration_history.append({"residual": residual, "drift": bool(drift)})
        return bool(drift)
