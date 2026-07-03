"""Ensemble epistemic uncertainty aggregation."""

import numpy as np


def ensemble_statistics(predictions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and sample standard deviation."""
    if len(predictions) < 2:
        raise ValueError("at least two ensemble members are required")
    values = np.asarray(predictions)
    return values.mean(axis=0), values.std(axis=0, ddof=1)
