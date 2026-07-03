"""Reproducible model selection helper."""

import pandas as pd
from src.surrogate.benchmark import benchmark


def select_model(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[object, list[object]]:
    """Select the lowest-validation-RMSE supported surrogate."""
    return benchmark(train, validation)
