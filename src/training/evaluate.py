"""Saved-model evaluation."""

from pathlib import Path
from src.dataset.loader import TARGETS, load_dataset
from src.metrics.regression import regression_metrics
from src.surrogate.model import SurrogateModel


def evaluate_model(model_path: str | Path, data_path: str | Path) -> dict[str, float]:
    """Evaluate a serialized surrogate against a labeled CSV."""
    frame = load_dataset(data_path)
    prediction = SurrogateModel.load(model_path).predict(frame)
    return regression_metrics(frame[TARGETS].to_numpy(), prediction.to_numpy())
