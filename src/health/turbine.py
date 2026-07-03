"""Turbine health diagnostics."""

import numpy as np


def estimate_turbine_health(actual_work: float, expected_work: float) -> float:
    """Estimate work extraction retention."""
    return float(np.clip(actual_work / max(expected_work, np.finfo(float).eps), 0, 1))
