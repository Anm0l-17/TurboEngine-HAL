"""Compressor health diagnostics."""

import numpy as np


def estimate_compressor_health(
    measured_pr: float, expected_pr: float, temperature_efficiency: float
) -> float:
    """Fuse pressure-ratio retention and thermal efficiency."""
    retention = measured_pr / max(expected_pr, np.finfo(float).eps)
    return float(np.clip(0.6 * retention + 0.4 * temperature_efficiency, 0, 1))
