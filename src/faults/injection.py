"""Realistic aircraft-engine fault injection.

Faults are modelled as deterministic perturbations to either:

* physical health multipliers consumed by :class:`~src.physics.cycle_model.BraytonCycle`
  (compressor fouling, turbine erosion, fuel nozzle blockage, bearing wear), or
* raw sensor observations before they reach the state estimator
  (sensor drift, sensor bias).

A :class:`FaultInjector` holds zero or more active :class:`FaultSpec` entries and
exposes two composition points used by the rest of the platform:

``apply_to_cycle_input`` -- mutates compressor/combustor/turbine health and fuel
flow so faults propagate through the physics model.

``apply_to_observation`` -- mutates a raw sensor observation dict so faults
propagate through the state estimator exactly as a real sensor fault would.

Both are pure functions of (spec, value) so faults compose predictably when several
are active simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

from src.physics.cycle_model import CycleInput

logger = logging.getLogger(__name__)


class FaultType(str, Enum):
    """Supported fault modes.

    Component faults degrade physical health multipliers. Sensor faults corrupt
    measurements without changing the true underlying physics.
    """

    COMPRESSOR_FOULING = "compressor_fouling"
    TURBINE_EROSION = "turbine_erosion"
    FUEL_NOZZLE_BLOCKAGE = "fuel_nozzle_blockage"
    BEARING_WEAR = "bearing_wear"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_BIAS = "sensor_bias"


_COMPONENT_FAULTS = {
    FaultType.COMPRESSOR_FOULING,
    FaultType.TURBINE_EROSION,
    FaultType.FUEL_NOZZLE_BLOCKAGE,
    FaultType.BEARING_WEAR,
}
_SENSOR_FAULTS = {FaultType.SENSOR_DRIFT, FaultType.SENSOR_BIAS}


@dataclass(frozen=True)
class FaultSpec:
    """A single active fault and its severity/target.

    Attributes:
        fault_type: Which fault mode is active.
        severity: Normalized severity in ``[0, 1]``. ``0`` is no effect, ``1`` is
            worst-case (e.g. full nozzle blockage, +100% sensor bias band).
        target_sensor: For sensor faults, the observation key to corrupt
            (e.g. ``"T3"``). Ignored for component faults.
        onset_cycle: Optional cycle index after which the fault becomes active,
            enabling progressive/onset scenarios. ``None`` means always active.
    """

    fault_type: FaultType
    severity: float = 0.5
    target_sensor: str | None = None
    onset_cycle: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if self.fault_type in _SENSOR_FAULTS and not self.target_sensor:
            raise ValueError(f"{self.fault_type} requires target_sensor")

    def is_active(self, cycle: float | None) -> bool:
        """Return whether this fault is active at the given cycle index."""
        if self.onset_cycle is None or cycle is None:
            return True
        return cycle >= self.onset_cycle


@dataclass
class FaultInjector:
    """Composable collection of active faults.

    Example:
        >>> injector = FaultInjector([FaultSpec(FaultType.COMPRESSOR_FOULING, 0.4)])
        >>> injector.active_faults()
        [FaultSpec(fault_type=<FaultType.COMPRESSOR_FOULING: 'compressor_fouling'>, ...)]
    """

    faults: list[FaultSpec] = field(default_factory=list)

    def add(self, spec: FaultSpec) -> "FaultInjector":
        """Activate an additional fault, returning self for chaining."""
        self.faults.append(spec)
        logger.info("fault activated: %s severity=%.2f", spec.fault_type.value, spec.severity)
        return self

    def clear(self) -> "FaultInjector":
        """Deactivate all faults."""
        self.faults.clear()
        return self

    def active_faults(self, cycle: float | None = None) -> list[FaultSpec]:
        """Return faults active at the given cycle (or all, if cycle is None)."""
        return [f for f in self.faults if f.is_active(cycle)]

    def apply_to_cycle_input(self, base: CycleInput, cycle: float | None = None) -> CycleInput:
        """Return a new :class:`CycleInput` with component faults applied.

        Compressor fouling and bearing wear reduce ``compressor_health``.
        Turbine erosion reduces ``turbine_health``. Fuel nozzle blockage reduces
        effective ``fuel_flow_kg_s`` (partial blockage starves the combustor).
        """
        compressor_health = base.compressor_health
        turbine_health = base.turbine_health
        fuel_flow = base.fuel_flow_kg_s
        for spec in self.active_faults(cycle):
            if spec.fault_type == FaultType.COMPRESSOR_FOULING:
                compressor_health *= 1.0 - 0.5 * spec.severity
            elif spec.fault_type == FaultType.BEARING_WEAR:
                # Bearing wear increases friction losses on both spools.
                compressor_health *= 1.0 - 0.25 * spec.severity
                turbine_health *= 1.0 - 0.25 * spec.severity
            elif spec.fault_type == FaultType.TURBINE_EROSION:
                turbine_health *= 1.0 - 0.5 * spec.severity
            elif spec.fault_type == FaultType.FUEL_NOZZLE_BLOCKAGE:
                fuel_flow *= 1.0 - 0.6 * spec.severity
        return CycleInput(
            altitude_m=base.altitude_m,
            mach=base.mach,
            ambient_temperature_k=base.ambient_temperature_k,
            ambient_pressure_pa=base.ambient_pressure_pa,
            rpm=base.rpm,
            fuel_flow_kg_s=max(fuel_flow, 0.0),
            mass_flow_kg_s=base.mass_flow_kg_s,
            compressor_health=max(compressor_health, 0.05),
            combustor_health=base.combustor_health,
            turbine_health=max(turbine_health, 0.05),
        )

    def apply_to_observation(
        self, observation: dict[str, float], cycle: float | None = None
    ) -> dict[str, float]:
        """Return a corrupted copy of a raw sensor observation.

        Sensor drift adds a cycle-growing offset (proportional to elapsed cycles
        past onset). Sensor bias adds a fixed offset proportional to severity and
        the sensor's own magnitude.
        """
        result = dict(observation)
        for spec in self.active_faults(cycle):
            if spec.fault_type not in _SENSOR_FAULTS:
                continue
            sensor = spec.target_sensor
            if sensor not in result:
                logger.warning("target_sensor %r not present in observation; skipping", sensor)
                continue
            magnitude = abs(result[sensor]) or 1.0
            if spec.fault_type == FaultType.SENSOR_BIAS:
                result[sensor] = result[sensor] + spec.severity * 0.1 * magnitude
            elif spec.fault_type == FaultType.SENSOR_DRIFT:
                elapsed = 0.0
                if spec.onset_cycle is not None and cycle is not None:
                    elapsed = max(cycle - spec.onset_cycle, 0.0)
                result[sensor] = result[sensor] + spec.severity * 0.01 * magnitude * elapsed
        return result

    def to_summary(self) -> list[dict[str, Any]]:
        """Return a JSON-safe summary of active faults, for API/UI display."""
        return [
            {
                "fault_type": spec.fault_type.value,
                "severity": spec.severity,
                "target_sensor": spec.target_sensor,
                "onset_cycle": spec.onset_cycle,
            }
            for spec in self.faults
        ]

    @classmethod
    def from_summary(cls, payload: list[dict[str, Any]]) -> "FaultInjector":
        """Reconstruct a :class:`FaultInjector` from ``to_summary`` output."""
        return cls(
            [
                FaultSpec(
                    fault_type=FaultType(item["fault_type"]),
                    severity=float(item.get("severity", 0.5)),
                    target_sensor=item.get("target_sensor"),
                    onset_cycle=item.get("onset_cycle"),
                )
                for item in payload
            ]
        )
