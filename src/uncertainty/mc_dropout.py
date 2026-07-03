"""Monte Carlo prediction aggregation."""

import numpy as np


def aggregate_mc_dropout(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate stochastic forward passes into mean and standard deviation."""
    values = np.asarray(samples, dtype=float)
    if values.shape[0] < 2:
        raise ValueError("multiple stochastic passes are required")
    return values.mean(axis=0), values.std(axis=0, ddof=1)
