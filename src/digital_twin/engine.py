"""Unified stateful digital twin API."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from src.dataset.loader import FEATURES
from src.estimation.state_estimator import StateEstimator
from src.faults.injection import FaultInjector
from src.health.overall import overall_health
from src.maintenance.recommendation import recommend
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.prediction.failure_probability import failure_probability
from src.prediction.rul import estimate_rul
from src.surrogate.model import SurrogateModel


class DigitalTwin:
    """Stateful fusion of physics, sensor data, and learned surrogate predictions."""

    def __init__(self, engine_id: str = "engine-1") -> None:
        self.engine_id = engine_id
        self.physics = BraytonCycle()
        self.estimator = StateEstimator()
        self.model: SurrogateModel | None = None
        self.history: list[dict[str, Any]] = []
        self.fault_injector: FaultInjector = FaultInjector()

    def initialize(self) -> "DigitalTwin":
        """Reset temporal state while retaining a loaded model."""
        self.estimator = StateEstimator()
        self.history.clear()
        return self

    def load_model(self, path: str | Path) -> "DigitalTwin":
        """Load a trained surrogate artifact."""
        self.model = SurrogateModel.load(path)
        return self

    def predict_performance(
        self, observation: dict[str, float], precomputed: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Predict health and performance using surrogate or physics fallback."""
        if precomputed is not None:
            return precomputed
        cycle_index = observation.get("Cycle")
        observation = self.fault_injector.apply_to_observation(observation, cycle_index)
        if self.model is not None:
            frame = pd.DataFrame([{name: observation[name] for name in FEATURES}])
            return {key: float(value) for key, value in self.model.predict(frame).iloc[0].items()}
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
        return {
            "CompressorHealth": compressor_health,
            "CombustorHealth": combustor_health,
            "TurbineHealth": turbine_health,
            "OverallHealth": overall_health(compressor_health, combustor_health, turbine_health),
            "Thrust": cycle.thrust_n,
            "TSFC": cycle.tsfc_kg_n_s,
        }

    def estimate_health(
        self, observation: dict[str, float], precomputed: dict[str, float] | None = None
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
        return dict(
            zip(
                ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"],
                map(float, state),
                strict=True,
            )
        )

    def update(
        self, observation: dict[str, float], precomputed: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Assimilate one cycle and return complete health, performance, risk, and action state."""
        performance = self.predict_performance(observation, precomputed)
        health = self.estimate_health(observation, precomputed)
        self.history.append(
            {"Cycle": float(observation.get("Cycle", len(self.history) + 1)), **health}
        )
        if len(self.history) >= 2:
            rul = estimate_rul(
                np.array([x["Cycle"] for x in self.history]),
                np.array([x["OverallHealth"] for x in self.history]),
            )
            remaining = rul.remaining_cycles
        else:
            remaining = 1_000.0
        probability = failure_probability(health["OverallHealth"], remaining)
        decision = recommend(
            health["OverallHealth"],
            remaining,
            probability,
            min(health, key=lambda key: health[key] if key != "OverallHealth" else 2),
        )
        return {
            "engine_id": self.engine_id,
            **health,
            "Thrust": performance["Thrust"],
            "TSFC": performance["TSFC"],
            "RULCycles": remaining,
            "FailureProbability": probability,
            "Maintenance": decision.action,
            "RiskLevel": decision.risk_level,
        }

    def batch_predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run ordered stateful inference over a frame, batching model calls."""
        precomputed_rows = None
        if self.model is not None:
            preds = self.model.predict(frame[FEATURES])
            precomputed_rows = preds.to_dict("records")
        results = []
        for i, (_, row) in enumerate(frame.iterrows()):
            pre = precomputed_rows[i] if precomputed_rows is not None else None
            results.append(self.update(row.to_dict(), pre))
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
