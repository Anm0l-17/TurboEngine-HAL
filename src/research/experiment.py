"""Experiment tracking for reproducible research workflows."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import pandas as pd
from src.dataset.loader import TARGETS, load_dataset
from src.dataset.split import grouped_split, official_split
from src.metrics.regression import regression_metrics
from src.surrogate.train import create_model


@dataclass
class ExperimentResult:
    """Record of a single experiment run."""

    experiment_id: str
    timestamp: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    model_path: str | None = None


_EXPERIMENTS: list[ExperimentResult] = []


def run_experiment(
    data_path: str | Path,
    kind: str = "extra_trees",
    n_estimators: int = 300,
    split_strategy: str = "official",
    scale_targets: bool = True,
    seed: int = 42,
    output_dir: str | Path = "results/experiments",
    tag: str = "",
) -> ExperimentResult:
    """Run a single training experiment with full logging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(data_path)
    split_fn = official_split if split_strategy == "official" else grouped_split
    train, test = split_fn(frame, seed=seed)

    model = create_model(kind, seed=seed, n_estimators=n_estimators, scale_targets=scale_targets)
    model.fit(train)
    model.calibrate(test)

    prediction = model.predict(test)
    metrics = regression_metrics(test[TARGETS].to_numpy(), prediction.to_numpy())

    exp_id = f"{kind}_{split_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = output_dir / f"{exp_id}.joblib"
    model.save(model_path)

    config = {
        "kind": kind,
        "n_estimators": n_estimators,
        "split_strategy": split_strategy,
        "scale_targets": scale_targets,
        "seed": seed,
        "data": str(data_path),
        "tag": tag,
    }
    result = ExperimentResult(
        experiment_id=exp_id,
        timestamp=datetime.now().isoformat(),
        config=config,
        metrics=metrics,
        model_path=str(model_path),
    )
    _EXPERIMENTS.append(result)

    report_path = output_dir / f"{exp_id}_report.json"
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def ablation_study(
    data_path: str | Path,
    base_kind: str = "extra_trees",
    seed: int = 42,
    output_dir: str | Path = "results/experiments",
) -> list[ExperimentResult]:
    """Run an ablation study varying model type, split strategy, and scaling."""
    results = []
    variants = [
        ("extra_trees", "official", True),
        ("hist_gradient_boosting", "official", True),
        ("stacking", "official", True),
        ("extra_trees", "grouped", True),
        ("hist_gradient_boosting", "grouped", True),
        ("extra_trees", "official", False),
        ("extra_trees", "official", True),
    ]
    for kind, strategy, scale in variants:
        try:
            r = run_experiment(
                data_path,
                kind=kind,
                split_strategy=strategy,
                scale_targets=scale,
                seed=seed,
                output_dir=output_dir,
                tag="ablation",
            )
            results.append(r)
            print(f"  {kind:>25s} {strategy:>10s} scale={str(scale):>5s}: "
                  f"RMSE={r.metrics['rmse']:.1f}  R2={r.metrics['r2']:.4f}")
        except Exception as e:
            print(f"  {kind:>25s} {strategy:>10s} scale={str(scale):>5s}: FAILED - {e}")
    return results


def summarize_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    """Summarize experiment results in a comparison table."""
    rows = []
    for r in results:
        rows.append({
            "experiment_id": r.experiment_id,
            "kind": r.config.get("kind"),
            "split": r.config.get("split_strategy"),
            "scale": r.config.get("scale_targets"),
            "rmse": r.metrics.get("rmse"),
            "mae": r.metrics.get("mae"),
            "mape": r.metrics.get("mape"),
            "r2": r.metrics.get("r2"),
            "tag": r.config.get("tag", ""),
        })
    return pd.DataFrame(rows)
