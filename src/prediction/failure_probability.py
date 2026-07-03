"""Calibratable failure-horizon risk model."""

import math


def failure_probability(
    health: float, remaining_cycles: float, horizon_cycles: float = 25, threshold: float = 0.3
) -> float:
    """Combine health margin and RUL horizon in a bounded logistic risk score."""
    health_term = (threshold - health) * 12
    horizon_term = (horizon_cycles - remaining_cycles) / max(horizon_cycles, 1) * 5
    score = max(min(health_term + horizon_term, 40), -40)
    return 1 / (1 + math.exp(-score))
