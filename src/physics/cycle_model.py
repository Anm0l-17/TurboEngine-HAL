"""Physically consistent single-spool turbojet Brayton-cycle model."""

from dataclasses import dataclass
import math
from .constants import CP_AIR, CP_GAS, GAMMA_AIR, GAMMA_GAS, LOWER_HEATING_VALUE
from .thermodynamics import speed_of_sound, total_pressure, total_temperature


@dataclass(frozen=True)
class CycleInput:
    """Cycle boundary conditions in SI units."""

    altitude_m: float
    mach: float
    ambient_temperature_k: float
    ambient_pressure_pa: float
    rpm: float
    fuel_flow_kg_s: float
    mass_flow_kg_s: float = 25.0
    compressor_health: float = 1.0
    combustor_health: float = 1.0
    turbine_health: float = 1.0


@dataclass(frozen=True)
class CycleState:
    """Reconstructed station state and performance."""

    p2: float
    t2: float
    p3: float
    t3: float
    p4: float
    t4: float
    thrust_n: float
    tsfc_kg_n_s: float
    thermal_efficiency: float
    compressor_work_w: float
    turbine_work_w: float
    energy_residual_w: float


class BraytonCycle:
    """Zero-dimensional turbojet cycle with spool energy balance."""

    def __init__(self, max_temperature_k: float = 1900.0) -> None:
        self.max_temperature_k = max_temperature_k

    def evaluate(self, value: CycleInput) -> CycleState:
        """Reconstruct cycle stations and reject nonphysical boundary conditions."""
        if not (
            0 <= value.mach <= 3
            and value.ambient_temperature_k > 0
            and value.ambient_pressure_pa > 0
            and value.mass_flow_kg_s > 0
            and value.fuel_flow_kg_s >= 0
        ):
            raise ValueError("nonphysical cycle input")
        t2 = total_temperature(value.ambient_temperature_k, value.mach)
        p2 = 0.98 * total_pressure(value.ambient_pressure_pa, value.mach)
        speed_fraction = max(0.2, min(1.15, value.rpm / 100_000.0))
        pressure_ratio = 1 + 11 * speed_fraction**2 * value.compressor_health
        eta_c = max(0.5, min(0.9, 0.86 * value.compressor_health))
        p3 = p2 * pressure_ratio
        t3s = t2 * pressure_ratio ** ((GAMMA_AIR - 1) / GAMMA_AIR)
        t3 = t2 + (t3s - t2) / eta_c
        air_flow = value.mass_flow_kg_s
        fuel_energy = value.fuel_flow_kg_s * LOWER_HEATING_VALUE * 0.99 * value.combustor_health
        t_turbine_in = t3 + fuel_energy / ((air_flow + value.fuel_flow_kg_s) * CP_GAS)
        if t_turbine_in > self.max_temperature_k:
            t_turbine_in = self.max_temperature_k
        compressor_work = air_flow * CP_AIR * (t3 - t2)
        turbine_flow = air_flow + value.fuel_flow_kg_s
        t4 = t_turbine_in - compressor_work / (turbine_flow * CP_GAS)
        eta_t = max(0.5, min(0.93, 0.9 * value.turbine_health))
        ideal_drop = max((t_turbine_in - t4) / eta_t, 0.0)
        p4 = 0.95 * p3 * (max(0.05, 1 - ideal_drop / t_turbine_in) ** (GAMMA_GAS / (GAMMA_GAS - 1)))
        exit_velocity = math.sqrt(
            max(
                0.0,
                2
                * CP_GAS
                * t4
                * (
                    1
                    - (value.ambient_pressure_pa / max(p4, value.ambient_pressure_pa))
                    ** ((GAMMA_GAS - 1) / GAMMA_GAS)
                ),
            )
        )
        flight_velocity = value.mach * speed_of_sound(value.ambient_temperature_k)
        thrust = max(0.0, turbine_flow * exit_velocity - air_flow * flight_velocity)
        tsfc = value.fuel_flow_kg_s / max(thrust, 1e-9)
        turbine_work = turbine_flow * CP_GAS * (t_turbine_in - t4)
        jet_power = 0.5 * turbine_flow * max(exit_velocity**2 - flight_velocity**2, 0)
        efficiency = jet_power / max(fuel_energy, 1e-9)
        return CycleState(
            p2,
            t2,
            p3,
            t3,
            p4,
            t4,
            thrust,
            tsfc,
            min(max(efficiency, 0.0), 1.0),
            compressor_work,
            turbine_work,
            turbine_work - compressor_work,
        )
