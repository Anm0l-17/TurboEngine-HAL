"""Regression metric suite."""

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate aggregate RMSE, MAE, MAPE, R2, and explained variance."""
    y, p = np.asarray(truth), np.asarray(prediction)
    denominator = np.maximum(np.abs(y), np.finfo(float).eps)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "mape": float(np.mean(np.abs((y - p) / denominator)) * 100),
        "r2": float(r2_score(y, p)),
        "explained_variance": float(explained_variance_score(y, p)),
    }
