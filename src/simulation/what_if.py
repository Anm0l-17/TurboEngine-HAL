"""What-if scenario simulator. Stateless per call."""

from dataclasses import dataclass
import numpy as np
from src.health.overall import overall_health
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.prediction.failure_probability import failure_probability
from src.prediction.rul import RULConfig, estimate_rul


@dataclass(frozen=True)
class ScenarioAdjustment:
    fuel_flow_kg_s: float | None = None
    rpm: float | None = None
    ambient_temperature_k: float | None = None
    ambient_pressure_pa: float | None = None
    compressor_efficiency: float | None = None
    turbine_efficiency: float | None = None
    sensor_noise_std: float = 0.0


def _snapshot(cycle_input: CycleInput, sensor_noise_std: float = 0.0) -> dict:
    state = cycle_input
    physics = BraytonCycle()
    cycle = physics.evaluate(state)
    health = overall_health(state.compressor_health, state.combustor_health, state.turbine_health)
    rul = estimate_rul(
        np.array([0.0, 1.0]),
        np.array([1.0, health]),
        RULConfig(failure_threshold=0.3),
    )
    remaining = min(rul.remaining_cycles, 5000.0)
    return {
        "compressor_health": state.compressor_health,
        "combustor_health": state.combustor_health,
        "turbine_health": state.turbine_health,
        "overall_health": health,
        "remaining_useful_life_cycles": remaining,
        "failure_probability": failure_probability(health, remaining),
        "thrust_n": cycle.thrust_n,
        "tsfc_kg_n_s": cycle.tsfc_kg_n_s,
        "confidence": float(np.clip(1.0 - sensor_noise_std, 0.0, 1.0)),
    }


def simulate_scenario(baseline_observation: dict, adjustment: ScenarioAdjustment) -> dict:
    """Compare baseline vs adjusted operating conditions. Returns dict with baseline, adjusted, delta."""

    def _make_input(overrides: dict) -> CycleInput:
        return CycleInput(
            altitude_m=overrides.get("altitude_m", baseline_observation.get("Altitude", 0.0)),
            mach=overrides.get("mach", baseline_observation.get("Mach", 0.0)),
            ambient_temperature_k=overrides.get("tamb", baseline_observation["Tamb"]),
            ambient_pressure_pa=overrides.get("pamb", baseline_observation["Pamb"]),
            rpm=overrides.get("rpm", baseline_observation["RPM"]),
            fuel_flow_kg_s=overrides.get("fuel_flow", baseline_observation["FuelFlow"]),
            compressor_health=overrides.get("comp_health", 1.0),
            turbine_health=overrides.get("turb_health", 1.0),
        )

    base = _make_input({})
    overrides = {
        "fuel_flow": adjustment.fuel_flow_kg_s,
        "rpm": adjustment.rpm,
        "tamb": adjustment.ambient_temperature_k,
        "pamb": adjustment.ambient_pressure_pa,
        "comp_health": adjustment.compressor_efficiency,
        "turb_health": adjustment.turbine_efficiency,
    }
    adjusted = _make_input({k: v for k, v in overrides.items() if v is not None})

    base_snap = _snapshot(base)
    adj_snap = _snapshot(adjusted, adjustment.sensor_noise_std)

    delta = {k: adj_snap[k] - base_snap[k] for k in base_snap}
    return {"baseline": base_snap, "adjusted": adj_snap, "delta": delta}
