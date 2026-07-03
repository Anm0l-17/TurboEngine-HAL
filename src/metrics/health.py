"""Health-estimation safety diagnostics."""

import numpy as np


def threshold_accuracy(truth: np.ndarray, prediction: np.ndarray, threshold: float = 0.3) -> float:
    """Calculate agreement on safe/failed classification."""
    return float(np.mean((np.asarray(truth) <= threshold) == (np.asarray(prediction) <= threshold)))
