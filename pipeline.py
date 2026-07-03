"""Command-line interface for training, evaluation, inference, and demonstration."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.dataset.loader import FEATURES, load_dataset
from src.digital_twin.engine import DigitalTwin
from src.training.evaluate import evaluate_model
from src.training.trainer import train_from_csv
from src.utils.logging import configure_logging
from src.utils.paths import ensure_directories

LOGGER = logging.getLogger(__name__)


def demo_data(engines: int = 5, cycles: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate a physically plausible deterministic dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for engine in range(1, engines + 1):
        rate = rng.uniform(0.0015, 0.004)
        for cycle in range(1, cycles + 1):
            altitude = rng.uniform(0, 10_000)
            mach = rng.uniform(0.1, 0.9)
            tamb = 288.15 - 0.0065 * altitude
            pamb = 101325 * (tamb / 288.15) ** 5.256
            rpm = rng.uniform(70_000, 98_000)
            fuel = rng.uniform(0.5, 1.2)
            health = np.clip(1 - rate * cycle + rng.normal(0, 0.004), 0.2, 1)
            p2 = pamb * (1 + 0.2 * mach**2) ** 3.5
            t2 = tamb * (1 + 0.2 * mach**2)
            p3 = p2 * (1 + 10 * (rpm / 100000) ** 2) * health
            t3 = t2 * (p3 / p2) ** (0.286 / max(0.7 * health, 0.4))
            t4 = min(1750, t3 + fuel * 43000000 / (26 * 1150)) - 250
            p4 = max(pamb * 1.05, p3 * 0.3)
            thrust = max(500, 20000 * health * (rpm / 100000) - altitude * 0.3)
            rows.append(
                [
                    engine,
                    cycle,
                    altitude,
                    mach,
                    tamb,
                    pamb,
                    rpm,
                    fuel,
                    p2,
                    t2,
                    p3,
                    t3,
                    p4,
                    t4,
                    health * 0.99,
                    health * 0.995,
                    health * 0.98,
                    health * 0.987,
                    thrust,
                    fuel / thrust,
                ]
            )
    from src.dataset.loader import TARGETS

    return pd.DataFrame(rows, columns=FEATURES + TARGETS)


def main() -> None:
    """Execute the selected CLI workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="models/best_model.joblib")
    train.add_argument("--kind", default="extra_trees")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", default="models/best_model.joblib")
    predict = commands.add_parser("predict")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", default="models/best_model.joblib")
    predict.add_argument("--output", default="results/predictions.csv")
    commands.add_parser("demo")
    args = parser.parse_args()
    configure_logging()
    ensure_directories()
    if args.command == "train":
        LOGGER.info("metrics=%s", json.dumps(train_from_csv(args.data, args.output, args.kind)))
    elif args.command == "evaluate":
        LOGGER.info("metrics=%s", json.dumps(evaluate_model(args.model, args.data)))
    elif args.command == "predict":
        frame = load_dataset(args.data, require_targets=False)
        result = DigitalTwin().load_model(args.model).batch_predict(frame)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        LOGGER.info("wrote %d predictions to %s", len(result), args.output)
    else:
        path = Path("results/demo_data.csv")
        demo_data().to_csv(path, index=False)
        metrics = train_from_csv(path, "models/best_model.joblib")
        LOGGER.info("demo complete metrics=%s", json.dumps(metrics))


if __name__ == "__main__":
    main()
