"""End-to-end training workflow."""

from pathlib import Path
import json
from src.dataset.loader import TARGETS, load_dataset
from src.dataset.split import grouped_split
from src.metrics.regression import regression_metrics
from src.surrogate.train import create_model


def train_from_csv(
    data_path: str | Path, output_path: str | Path, kind: str = "extra_trees", seed: int = 42
) -> dict[str, float]:
    """Validate, split, fit, evaluate, and persist a surrogate."""
    frame = load_dataset(data_path)
    train, test = grouped_split(frame, seed=seed)
    model = create_model(kind, seed=seed).fit(train)
    metrics = regression_metrics(test[TARGETS].to_numpy(), model.predict(test).to_numpy())
    model.save(output_path)
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
