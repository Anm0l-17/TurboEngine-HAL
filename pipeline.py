"""Command-line interface for training, evaluation, inference, and demonstration."""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from src.dataset.loader import load_dataset, sample_real_dataset
from src.dataset.split import official_split
from src.digital_twin.engine import DigitalTwin
from src.report.research import generate_research_report
from src.research.experiment import ablation_study, run_experiment
from src.training.evaluate import evaluate_model
from src.training.hyperparameter_search import select_model
from src.training.trainer import train_from_csv
from src.utils.logging import configure_logging
from src.validation.benchmark import run_cmapss_validation, run_validation_suite
from src.performance.benchmark import run_benchmark_suite
from src.utils.paths import ensure_directories

LOGGER = logging.getLogger(__name__)


def demo_data(engines: int | None = 5, cycles: int | None = 30, seed: int = 42) -> pd.DataFrame:
    """Deterministic slice of the official dataset for smoke testing."""
    n_engines = None if engines is None else min(engines, 10)
    n_cycles = None if cycles is None else min(cycles, 30)
    return sample_real_dataset(n_engines=n_engines, n_cycles=n_cycles, seed=seed)


def main() -> None:
    """Execute the selected CLI workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="models/best_model.joblib")
    train.add_argument(
        "--kind",
        default="extra_trees",
        choices=["hist_gradient_boosting", "extra_trees", "random_forest", "stacking", "hybrid"],
    )
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--strategy", default="official", choices=["official", "grouped"])
    tune = commands.add_parser("tune")
    tune.add_argument("--data", required=True)
    tune.add_argument("--output", default="models/best_model.joblib")

    exp = commands.add_parser("experiment")
    exp.add_argument("--data", required=True)
    exp.add_argument("--kind", default="extra_trees")
    exp.add_argument("--n-estimators", type=int, default=300)
    exp.add_argument(
        "--split",
        default="grouped",
        choices=["official", "grouped"],
        help="'grouped' holds out entire engines (tests cross-engine generalization; "
        "15%% of challenge score). 'official' reproduces train.csv/test.csv split "
        "(same engines in both). Report BOTH in your submission.",
    )
    exp.add_argument("--output-dir", default="results/experiments")
    exp.add_argument("--tag", default="")

    ablation = commands.add_parser("ablation")
    ablation.add_argument("--data", required=True)
    ablation.add_argument("--output-dir", default="results/experiments")

    report_cmd = commands.add_parser("report")
    report_cmd.add_argument("--input-dir", default="results/experiments")
    report_cmd.add_argument("--output", default="results/research_report.md")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", default="models/best_model.joblib")
    predict = commands.add_parser("predict")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", default="models/best_model.joblib")
    predict.add_argument("--output", default="results/predictions.csv")
    validate_cmd = commands.add_parser("validation")
    validate_cmd.add_argument("--data", default="data/turbojet_complete_dataset.csv")
    validate_cmd.add_argument("--output-dir", default="results/validation")
    cmapss_cmd = commands.add_parser("cmapss")
    cmapss_cmd.add_argument("--data-dir", default="data/cmapss")
    cmapss_cmd.add_argument("--output-dir", default="results/cmapss")
    benchmark_cmd = commands.add_parser("benchmark")
    benchmark_cmd.add_argument("--data", default="data/turbojet_complete_dataset.csv")
    benchmark_cmd.add_argument("--output-dir", default="results/benchmarks")
    orchestrate_cmd = commands.add_parser("orchestrate")
    orchestrate_cmd.add_argument("--data", default="data/turbojet_complete_dataset.csv")
    orchestrate_cmd.add_argument("--output-dir", default="results")
    commands.add_parser("demo")
    args = parser.parse_args()
    configure_logging()
    ensure_directories()
    if args.command == "train":
        LOGGER.info(
            "metrics=%s",
            json.dumps(
                train_from_csv(
                    args.data,
                    args.output,
                    args.kind,
                    n_estimators=args.n_estimators,
                    strategy=args.strategy,
                )
            ),
        )
    elif args.command == "tune":
        frame = load_dataset(args.data)
        train, test = official_split(frame, seed=42)
        # Hold out half of test for conformal calibration — model selection on the other half
        calibration = test.iloc[: len(test) // 2]
        heldout = test.iloc[len(test) // 2 :]
        model, config = select_model(train, heldout)
        model.calibrate(calibration)
        model.save(args.output)
        config_path = Path(args.output).with_suffix(".config.json")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        LOGGER.info("tune complete best_config=%s", json.dumps(config))
    elif args.command == "evaluate":
        LOGGER.info("metrics=%s", json.dumps(evaluate_model(args.model, args.data)))
    elif args.command == "cmapss":
        results = run_cmapss_validation(data_dir=args.data_dir, output_dir=args.output_dir)
        total = sum(len(v) for v in results.values())
        LOGGER.info("cmapss validation complete %d model×subset runs", total)
    elif args.command == "validation":
        results = run_validation_suite(data_path=args.data, output_dir=args.output_dir)
        LOGGER.info("validation complete %d experiments", len(results))
    elif args.command == "benchmark":
        results = run_benchmark_suite(data_path=args.data, output_dir=args.output_dir)
        LOGGER.info("benchmark complete %d variants", len(results))
    elif args.command == "experiment":
        result = run_experiment(
            args.data,
            kind=args.kind,
            n_estimators=args.n_estimators,
            split_strategy=args.split,
            output_dir=args.output_dir,
            tag=args.tag,
        )
        LOGGER.info(
            "experiment complete id=%s metrics=%s", result.experiment_id, json.dumps(result.metrics)
        )

    elif args.command == "ablation":
        results = ablation_study(args.data, output_dir=args.output_dir)
        report_path = Path(args.output_dir) / "ablation_report.md"
        generate_research_report(
            [
                {"experiment_id": r.experiment_id, "config": r.config, "metrics": r.metrics}
                for r in results
            ],
            report_path,
        )
        LOGGER.info("ablation complete %d experiments, report=%s", len(results), report_path)

    elif args.command == "report":
        reports_dir = Path(args.input_dir)
        experiments = []
        for f in sorted(reports_dir.glob("*_report.json")):
            experiments.append(json.loads(f.read_text(encoding="utf-8")))
        generate_research_report(experiments, args.output)
        LOGGER.info("report generated with %d experiments", len(experiments))

    elif args.command == "orchestrate":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("=== TRAIN ALL VARIANTS ===")
        kinds = ["hist_gradient_boosting", "extra_trees", "stacking", "hybrid"]
        trained = {}
        for kind in kinds:
            local_out = output_dir / "models" / f"{kind}.joblib"
            local_out.parent.mkdir(parents=True, exist_ok=True)
            metrics = train_from_csv(args.data, str(local_out), kind=kind)
            trained[kind] = metrics
            LOGGER.info("%s: rmse=%.2f r2=%.4f", kind, metrics.get("rmse", 0), metrics.get("r2", 0))
        LOGGER.info("=== VALIDATION SUITE ===")
        run_validation_suite(data_path=args.data, output_dir=output_dir / "validation")
        LOGGER.info("=== PERFORMANCE BENCHMARK ===")
        run_benchmark_suite(data_path=args.data, output_dir=output_dir / "benchmarks")
        summary = output_dir / "orchestrate_summary.json"
        summary.write_text(json.dumps(trained, indent=2), encoding="utf-8")
        LOGGER.info("orchestrate complete — all artifacts in %s", output_dir)

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
