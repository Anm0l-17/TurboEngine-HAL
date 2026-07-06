"""Unified stateful digital twin API."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from src.dataset.loader import SENSOR_FEATURES, TARGETS
from src.estimation.state_estimator import StateEstimator
from src.faults.injection import FaultInjector
from src.health.overall import overall_health
from src.maintenance.recommendation import recommend
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.prediction.failure_probability import FailureProbabilityCalibrator, failure_probability
from src.prediction.rul import RULConfig, estimate_rul
from src.surrogate.hybrid import HybridPhysicsMLModel
from src.surrogate.model import SurrogateModel

_HEALTH_TARGETS = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]


class DigitalTwin:
    """Stateful fusion of physics, sensor data, and learned surrogate predictions."""

    def __init__(self, engine_id: str = "engine-1", estimator_method: str = "ekf") -> None:
        self.engine_id = engine_id
        self.estimator_method = estimator_method
        self.physics = BraytonCycle()
        self.estimator = StateEstimator(method=estimator_method)
        self.model: SurrogateModel | HybridPhysicsMLModel | None = None
        self.history: list[dict[str, Any]] = []
        self.fault_injector: FaultInjector = FaultInjector()
        self.failure_calibrator: FailureProbabilityCalibrator | None = None

    def set_failure_calibrator(self, calibrator: FailureProbabilityCalibrator) -> "DigitalTwin":
        """Attach a data-calibrated failure risk model."""
        self.failure_calibrator = calibrator
        return self

    def initialize(self) -> "DigitalTwin":
        """Reset temporal state while retaining a loaded model."""
        self.estimator = StateEstimator(method=self.estimator_method)
        self.history.clear()
        return self

    def load_model(self, path: str | Path) -> "DigitalTwin":
        """Load a trained surrogate artifact (SurrogateModel or HybridPhysicsMLModel)."""
        import joblib
        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "ml_model" in artifact:
            self.model = HybridPhysicsMLModel.load(path)
        elif isinstance(artifact, SurrogateModel):
            self.model = artifact
        else:
            raise TypeError(f"artifact is a {type(artifact).__name__}, expected SurrogateModel or HybridPhysicsMLModel")
        return self

    def _point_bundle(
        self, prediction: dict[str, float], confidence: float, method: str
    ) -> dict[str, Any]:
        """Create a prediction/interval payload from point values."""
        point = {name: float(prediction[name]) for name in TARGETS}
        return {
            "prediction": point,
            "lower": point.copy(),
            "upper": point.copy(),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "method": method,
        }

    def predict_with_uncertainty(
        self, observation: dict[str, float], precomputed: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Predict target values plus calibrated intervals when available."""
        if precomputed is not None:
            if {"prediction", "lower", "upper"}.issubset(precomputed):
                return precomputed
            return self._point_bundle(precomputed, 0.0, "precomputed_point")

        cycle_index = observation.get("Cycle")
        observation = self.fault_injector.apply_to_observation(observation, cycle_index)
        if self.model is not None:
            frame = pd.DataFrame([{name: observation[name] for name in SENSOR_FEATURES}])
            prediction, lower, upper, confidence = self.model.predict_with_uncertainty(frame)
            method = "conformal" if confidence > 0 else "uncalibrated_point"
            return {
                "prediction": {key: float(value) for key, value in prediction.iloc[0].items()},
                "lower": {key: float(value) for key, value in lower.iloc[0].items()},
                "upper": {key: float(value) for key, value in upper.iloc[0].items()},
                "confidence": confidence,
                "method": method,
            }

        base_input = CycleInput(
            observation.get("Altitude", 0),
            observation.get("Mach", 0),
            observation["Tamb"],
            observation["Pamb"],
            observation["RPM"],
            observation["FuelFlow"],
        )
        faulted_input = self.fault_injector.apply_to_cycle_input(base_input, cycle_index)
        cycle = self.physics.evaluate(faulted_input)
        compressor_health = faulted_input.compressor_health
        combustor_health = faulted_input.combustor_health
        turbine_health = faulted_input.turbine_health
        return self._point_bundle(
            {
                "CompressorHealth": compressor_health,
                "CombustorHealth": combustor_health,
                "TurbineHealth": turbine_health,
                "OverallHealth": overall_health(
                    compressor_health, combustor_health, turbine_health
                ),
                "Thrust": cycle.thrust_n,
                "TSFC": cycle.tsfc_kg_n_s,
            },
            1.0,
            "physics_deterministic",
        )

    def predict_performance(
        self, observation: dict[str, float], precomputed: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Predict health and performance using surrogate or physics fallback."""
        return self.predict_with_uncertainty(observation, precomputed)["prediction"]

    def estimate_health(
        self, observation: dict[str, float], precomputed: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Filter surrogate subsystem-health observations."""
        prediction = self.predict_performance(observation, precomputed)
        raw = np.array(
            [
                prediction["CompressorHealth"],
                prediction["CombustorHealth"],
                prediction["TurbineHealth"],
                prediction["OverallHealth"],
            ]
        )
        state = self.estimator.update(raw)
        state[3] = overall_health(*state[:3])
        self.estimator.filter.state = state.copy()
        return dict(
            zip(
                _HEALTH_TARGETS,
                map(float, state),
                strict=True,
            )
        )

    def update(
        self, observation: dict[str, float], precomputed: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Assimilate one cycle and return complete health, performance, risk, and action state."""
        bundle = self.predict_with_uncertainty(observation, precomputed)
        performance = bundle["prediction"]
        health = self.estimate_health(observation, performance)
        lower = dict(bundle["lower"])
        upper = dict(bundle["upper"])
        for name in _HEALTH_TARGETS:
            lower[name] = min(float(lower[name]), health[name])
            upper[name] = max(float(upper[name]), health[name])
        cycle = float(observation.get("Cycle", len(self.history) + 1))
        self.history.append({"Cycle": cycle, **health})
        if len(self.history) >= 2:
            rul = estimate_rul(
                np.array([x["Cycle"] for x in self.history]),
                np.array([x["OverallHealth"] for x in self.history]),
                RULConfig(failure_threshold=0.3, warning_threshold=0.7),
            )
            remaining = rul.remaining_cycles
            rul_lower, rul_upper = rul.q10, rul.q90
            degradation_rate = rul.degradation_rate
        else:
            remaining = 1_000.0
            rul_lower = rul_upper = remaining
            degradation_rate = 0.0
        probability = failure_probability(health["OverallHealth"], remaining, calibrator=self.failure_calibrator)
        probability_lower = failure_probability(float(upper["OverallHealth"]), rul_upper, calibrator=self.failure_calibrator)
        probability_upper = failure_probability(float(lower["OverallHealth"]), rul_lower, calibrator=self.failure_calibrator)
        probability_lower, probability_upper = sorted((probability_lower, probability_upper))
        decision = recommend(
            health["OverallHealth"],
            remaining,
            probability,
            min(health, key=lambda key: health[key] if key != "OverallHealth" else 2),
        )
        interval_fields = {}
        for name in TARGETS:
            interval_fields[f"{name}Lower"] = float(lower[name])
            interval_fields[f"{name}Upper"] = float(upper[name])
        return {
            "engine_id": self.engine_id,
            "Cycle": cycle,
            **health,
            "Thrust": performance["Thrust"],
            "TSFC": performance["TSFC"],
            "RULCycles": remaining,
            "RULCyclesLower": rul_lower,
            "RULCyclesUpper": rul_upper,
            "DegradationRate": degradation_rate,
            "FailureProbability": probability,
            "FailureProbabilityLower": probability_lower,
            "FailureProbabilityUpper": probability_upper,
            "Confidence": float(bundle["confidence"]),
            "UncertaintyMethod": bundle["method"],
            **interval_fields,
            "Maintenance": decision.action,
            "RiskLevel": decision.risk_level,
        }

    def calibrate_failure_model(self, horizon: int = 25, threshold: float = 0.7) -> "DigitalTwin":
        """Fit the failure probability model from accumulated health history."""
        if len(self.history) >= 10:
            health_trace = [x["OverallHealth"] for x in self.history]
            self.failure_calibrator = FailureProbabilityCalibrator().fit(health_trace, horizon, threshold)
        return self

    def batch_predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run ordered stateful inference over a frame, batching model calls."""
        precomputed_rows = None
        if self.model is not None and not self.fault_injector.faults:
            prediction, lower, upper, confidence = self.model.predict_with_uncertainty(
                frame[SENSOR_FEATURES]
            )
            method = "conformal" if confidence > 0 else "uncalibrated_point"
            precomputed_rows = [
                {
                    "prediction": prediction.iloc[i].to_dict(),
                    "lower": lower.iloc[i].to_dict(),
                    "upper": upper.iloc[i].to_dict(),
                    "confidence": confidence,
                    "method": method,
                }
                for i in range(len(frame))
            ]
        results = []
        for i, (_, row) in enumerate(frame.iterrows()):
            pre = precomputed_rows[i] if precomputed_rows is not None else None
            results.append(self.update(row.to_dict(), pre))
        self.calibrate_failure_model()
        return pd.DataFrame(results)

    def stream_predict(self, observations: Iterable[dict[str, float]]) -> Iterator[dict[str, Any]]:
        """Yield stateful predictions from an observation stream."""
        for observation in observations:
            yield self.update(observation)

    def save_state(self, path: str | Path) -> None:
        """Persist JSON-safe runtime history."""
        Path(path).write_text(
            json.dumps({"engine_id": self.engine_id, "history": self.history}), encoding="utf-8"
        )

    def load_state(self, path: str | Path) -> "DigitalTwin":
        """Restore runtime history and rebuild estimator state."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.engine_id, self.history = payload["engine_id"], payload["history"]
        if self.history:
            last = self.history[-1]
            self.estimator.filter.state = np.array(
                [
                    last[k]
                    for k in (
                        "CompressorHealth",
                        "CombustorHealth",
                        "TurbineHealth",
                        "OverallHealth",
                    )
                ]
            )
        return self
