"""Bounded component efficiency maps with off-design and degradation."""

import numpy as np


def compressor_efficiency(corrected_speed_fraction: float, health: float = 1.0) -> float:
    """Compressor isentropic efficiency vs corrected speed, degraded by health."""
    eta_design = 0.87
    speed = np.clip(corrected_speed_fraction, 0.2, 1.2)
    eta = eta_design - 0.30 * (speed - 0.88) ** 2 - 0.10 * (speed - 0.88) ** 4
    return float(np.clip(eta * (0.85 + 0.15 * health), 0.35, 0.92))


def compressor_pressure_ratio(
    corrected_speed_fraction: float, health: float = 1.0, design_pr: float = 10.0
) -> float:
    """Compressor pressure ratio vs corrected speed, degraded by health.

    Returns PR relative to design-point PR at 100% corrected speed.
    """
    speed = np.clip(corrected_speed_fraction, 0.2, 1.15)
    # Normalised off-design PR ratio: ~1.0 at design speed (s=1.0), ~0 at s~0.
    # The raw polynomial (1 + 8.5*s^2.5 - 2.5*s^5) gives 7.0 at s=1.0, so divide by 7.
    pr_ratio = (1.0 + 8.5 * speed ** 2.5 - 2.5 * speed ** 5) / 7.0
    return 1.0 + (design_pr - 1.0) * pr_ratio * health


def turbine_efficiency(speed_fraction: float, health: float = 1.0) -> float:
    """Turbine isentropic efficiency vs speed, degraded by health."""
    eta_design = 0.90
    speed = np.clip(speed_fraction, 0.2, 1.2)
    eta = eta_design - 0.25 * (speed - 0.90) ** 2 - 0.08 * (speed - 0.90) ** 4
    return float(np.clip(eta * (0.85 + 0.15 * health), 0.40, 0.94))


def combustor_efficiency(load_fraction: float, health: float = 1.0) -> float:
    """Combustor efficiency vs load fraction, degraded by health."""
    load = np.clip(load_fraction, 0.1, 1.0)
    eta = 0.995 - 0.04 * (1.0 - load) ** 2
    return float(np.clip(eta * (0.88 + 0.12 * health), 0.70, 0.999))
