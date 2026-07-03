"""Fuel efficiency calculations."""

import numpy as np


def calculate_tsfc(fuel_flow_kg_s: float, thrust_n: float) -> float:
    """Calculate thrust-specific fuel consumption in kg/(N s)."""
    if fuel_flow_kg_s < 0 or thrust_n <= 0:
        raise ValueError("fuel flow must be nonnegative and thrust positive")
    return float(fuel_flow_kg_s / max(thrust_n, np.finfo(float).eps))
