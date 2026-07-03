"""Bounded component efficiency maps."""

import numpy as np


def compressor_efficiency(corrected_speed_fraction: float, health: float = 1.0) -> float:
    """Approximate compressor efficiency around its design point."""
    eta = 0.88 - 0.22 * (corrected_speed_fraction - 0.9) ** 2
    return float(np.clip(eta * health, 0.45, 0.92))


def turbine_efficiency(speed_fraction: float, health: float = 1.0) -> float:
    """Approximate turbine efficiency around its design point."""
    eta = 0.91 - 0.18 * (speed_fraction - 0.92) ** 2
    return float(np.clip(eta * health, 0.5, 0.94))
