"""Reusable ideal-gas thermodynamic relations."""

import math
from .constants import GAS_CONSTANT_AIR, isa_pressure, isa_temperature


def total_temperature(static_temperature: float, mach: float, gamma: float = 1.4) -> float:
    """Return stagnation temperature for an adiabatic inlet."""
    return static_temperature * (1 + 0.5 * (gamma - 1) * mach**2)


def total_pressure(static_pressure: float, mach: float, gamma: float = 1.4) -> float:
    """Return isentropic stagnation pressure."""
    return static_pressure * (1 + 0.5 * (gamma - 1) * mach**2) ** (gamma / (gamma - 1))


def speed_of_sound(temperature: float, gamma: float = 1.4, gas_constant: float = GAS_CONSTANT_AIR) -> float:
    """Return ideal-gas speed of sound in m/s."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return math.sqrt(gamma * gas_constant * temperature)


def standard_atmosphere(altitude_m: float) -> tuple[float, float]:
    """Return (temperature_K, pressure_Pa) for the ISA at a given altitude."""
    return isa_temperature(altitude_m), isa_pressure(altitude_m)
