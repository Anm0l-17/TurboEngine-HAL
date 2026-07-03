"""Prediction interval diagnostics."""

import numpy as np


def interval_metrics(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Calculate empirical coverage and mean interval width."""
    y, lo, hi = np.asarray(truth), np.asarray(lower), np.asarray(upper)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }
