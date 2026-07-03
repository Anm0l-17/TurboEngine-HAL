"""Unit tests for src.faults.injection."""

import pytest

from src.faults.injection import FaultInjector, FaultSpec, FaultType
from src.physics.cycle_model import CycleInput


def _baseline_input() -> CycleInput:
    return CycleInput(
        altitude_m=0.0,
        mach=0.0,
        ambient_temperature_k=288.15,
        ambient_pressure_pa=101_325.0,
        rpm=80_000.0,
        fuel_flow_kg_s=1.0,
    )


def test_severity_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        FaultSpec(FaultType.COMPRESSOR_FOULING, severity=1.5)


def test_sensor_fault_requires_target() -> None:
    with pytest.raises(ValueError):
        FaultSpec(FaultType.SENSOR_BIAS, severity=0.5)


def test_compressor_fouling_degrades_health() -> None:
    injector = FaultInjector([FaultSpec(FaultType.COMPRESSOR_FOULING, severity=0.6)])
    faulted = injector.apply_to_cycle_input(_baseline_input())
    assert faulted.compressor_health < 1.0
    assert faulted.turbine_health == 1.0


def test_fuel_nozzle_blockage_reduces_fuel_flow() -> None:
    injector = FaultInjector([FaultSpec(FaultType.FUEL_NOZZLE_BLOCKAGE, severity=0.5)])
    faulted = injector.apply_to_cycle_input(_baseline_input())
    assert faulted.fuel_flow_kg_s < 1.0


def test_multiple_faults_compose() -> None:
    injector = FaultInjector(
        [
            FaultSpec(FaultType.COMPRESSOR_FOULING, severity=0.3),
            FaultSpec(FaultType.TURBINE_EROSION, severity=0.3),
            FaultSpec(FaultType.BEARING_WEAR, severity=0.2),
        ]
    )
    faulted = injector.apply_to_cycle_input(_baseline_input())
    assert faulted.compressor_health < 1.0
    assert faulted.turbine_health < 1.0


def test_sensor_bias_offsets_observation() -> None:
    injector = FaultInjector([FaultSpec(FaultType.SENSOR_BIAS, severity=0.5, target_sensor="T3")])
    observation = {"T3": 800.0}
    corrupted = injector.apply_to_observation(observation)
    assert corrupted["T3"] != observation["T3"]


def test_sensor_drift_grows_with_elapsed_cycles() -> None:
    injector = FaultInjector(
        [FaultSpec(FaultType.SENSOR_DRIFT, severity=0.5, target_sensor="T3", onset_cycle=0)]
    )
    early = injector.apply_to_observation({"T3": 800.0}, cycle=1)
    late = injector.apply_to_observation({"T3": 800.0}, cycle=50)
    assert abs(late["T3"] - 800.0) > abs(early["T3"] - 800.0)


def test_onset_cycle_gates_activation() -> None:
    spec = FaultSpec(FaultType.COMPRESSOR_FOULING, severity=0.5, onset_cycle=10)
    injector = FaultInjector([spec])
    assert injector.apply_to_cycle_input(_baseline_input(), cycle=5).compressor_health == 1.0
    assert injector.apply_to_cycle_input(_baseline_input(), cycle=15).compressor_health < 1.0


def test_summary_round_trip() -> None:
    injector = FaultInjector([FaultSpec(FaultType.BEARING_WEAR, severity=0.4)])
    restored = FaultInjector.from_summary(injector.to_summary())
    assert restored.faults[0].fault_type == FaultType.BEARING_WEAR
    assert restored.faults[0].severity == pytest.approx(0.4)


def test_unknown_sensor_target_logs_and_skips() -> None:
    injector = FaultInjector([FaultSpec(FaultType.SENSOR_BIAS, severity=0.5, target_sensor="Missing")])
    observation = {"T3": 800.0}
    result = injector.apply_to_observation(observation)
    assert result == observation
