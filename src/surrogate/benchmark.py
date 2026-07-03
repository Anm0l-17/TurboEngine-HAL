"""Deterministic surrogate benchmarking."""

from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.dataset.loader import TARGETS
from .train import create_model


@dataclass(frozen=True)
class BenchmarkResult:
    kind: str
    rmse: float
    mae: float


def benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kinds: tuple[str, ...] = ("extra_trees", "random_forest", "mlp"),
) -> tuple[object, list[BenchmarkResult]]:
    """Fit candidates, rank by aggregate RMSE, and return the best model."""
    results, models = [], []
    for kind in kinds:
        model = create_model(kind).fit(train)
        prediction = model.predict(test)
        results.append(
            BenchmarkResult(
                kind,
                float(root_mean_squared_error(test[TARGETS], prediction)),
                float(mean_absolute_error(test[TARGETS], prediction)),
            )
        )
        models.append(model)
    order = sorted(range(len(results)), key=lambda index: results[index].rmse)
    return models[order[0]], [results[index] for index in order]
