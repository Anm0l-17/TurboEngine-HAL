"""Thrust prediction constraints."""

import numpy as np


def constrain_thrust(value: float) -> float:
    """Reject non-finite values and enforce nonnegative thrust."""
    if not np.isfinite(value):
        raise ValueError("thrust must be finite")
    return max(float(value), 0.0)
