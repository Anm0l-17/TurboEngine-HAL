"""Deterministic surrogate benchmarking with per-target metrics."""

from dataclasses import dataclass, field
import pandas as pd
from src.dataset.loader import TARGETS
from src.metrics.regression import regression_metrics
from .train import create_model


@dataclass(frozen=True)
class BenchmarkResult:
    kind: str
    rmse: float
    mae: float
    per_target: dict[str, dict[str, float]] = field(default_factory=dict)


def benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kinds: tuple[str, ...] = (
        "extra_trees", "random_forest", "gradient_boosting",
        "hist_gradient_boosting", "stacking", "mlp",
    ),
) -> tuple[object, list[BenchmarkResult]]:
    """Fit candidates, rank by aggregate RMSE, and return the best model."""
    results, models = [], []
    for kind in kinds:
        try:
            model = create_model(kind).fit(train)
            prediction = model.predict(test)
            metrics = regression_metrics(test[TARGETS].to_numpy(), prediction.to_numpy())
            results.append(
                BenchmarkResult(kind, metrics["rmse"], metrics["mae"], metrics["per_target"])
            )
            models.append(model)
        except Exception:
            results.append(BenchmarkResult(kind, float("inf"), float("inf")))
            models.append(None)
    valid = [(i, r) for i, r in enumerate(results) if r.rmse < float("inf")]
    order = sorted(valid, key=lambda pair: pair[1].rmse)
    best_idx = order[0][0] if order else 0
    return models[best_idx], [results[i] for i, _ in order]
