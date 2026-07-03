"""Degradation-trend remaining useful life estimation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RULResult:
    """Remaining life point estimate and empirical uncertainty quantiles."""

    remaining_cycles: float
    remaining_hours: float
    q10: float
    q50: float
    q90: float
    degradation_rate: float


def estimate_rul(
    cycles: np.ndarray, health: np.ndarray, threshold: float = 0.3, hours_per_cycle: float = 1.5
) -> RULResult:
    """Extrapolate robust recent linear degradation to a failure threshold."""
    x, y = np.asarray(cycles, dtype=float), np.asarray(health, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("at least two aligned observations are required")
    window = min(len(x), 50)
    slope, _ = np.polyfit(x[-window:], y[-window:], 1)
    rate = max(-float(slope), 1e-6)
    remaining = max((float(y[-1]) - threshold) / rate, 0.0)
    residual = y[-window:] - np.polyval(np.polyfit(x[-window:], y[-window:], 1), x[-window:])
    uncertainty = 1.645 * float(np.std(residual)) / rate
    return RULResult(
        remaining,
        remaining * hours_per_cycle,
        max(remaining - uncertainty, 0),
        remaining,
        remaining + uncertainty,
        rate,
    )
