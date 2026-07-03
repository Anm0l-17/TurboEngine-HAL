"""Combustor health diagnostics."""

import numpy as np


def estimate_combustor_health(
    actual_delta_t: float, fuel_flow: float, reference_heat_per_fuel: float
) -> float:
    """Estimate combustion effectiveness from normalized heat rise."""
    expected = fuel_flow * reference_heat_per_fuel
    return float(np.clip(actual_delta_t / max(expected, np.finfo(float).eps), 0, 1))
