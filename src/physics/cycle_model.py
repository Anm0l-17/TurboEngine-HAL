"""Physically consistent single-spool turbojet Brayton-cycle model with variable gas properties.

Station convention. This module's public station labels (p2/t2, p3/t3, p4/t4)
match the dataset schema exactly as specified in the challenge problem
statement's measurement table, NOT the classical Brayton-cycle numbering:

    Symbol   Challenge label            Physical station
    ------   -------------------------  --------------------------------
    P2, T2   Compressor Exit            downstream of the compressor
    P3, T3   Combustor Exit             = turbine inlet (challenge PDF
                                         explicitly calls T3 "Turbine
                                         Inlet Temperature")
    P4, T4   Turbine Exit               downstream of the turbine

An unlabeled freestream/ambient inlet station (call it station "1", ram
compression from ambient conditions) exists internally between the ambient
boundary condition and the compressor, but is not part of the reported
schema and has no output field. Earlier revisions of this module
incorrectly treated p2/t2 as the *compressor inlet* (i.e. used the
classical numbering rather than the challenge's), which silently
misaligned every downstream equation with the actual dataset columns and
produced systematically biased Thrust/TSFC predictions (physics-only
R² as low as -115 on TSFC). This revision fixes that station mapping.
"""

from dataclasses import dataclass
import math
from .constants import (
    LOWER_HEATING_VALUE,
    cp_air,
    cp_gas,
    gamma_from_cp,
)
from .component_maps import (
    combustor_efficiency,
    compressor_efficiency,
    compressor_pressure_ratio,
    turbine_efficiency,
)
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
    """Reconstructed station state and performance.

    p2/t2 = compressor exit, p3/t3 = combustor exit (turbine inlet),
    p4/t4 = turbine exit — see module docstring for the station convention.
    """

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
    """Zero-dimensional turbojet cycle with variable specific heats and realistic component maps.

    Parameters
    ----------
    max_temperature_k : float
        Turbine inlet temperature limit (TIT), i.e. an upper bound on the
        combustor-exit temperature (challenge's T3).
    design_pr : float
        Compressor pressure ratio at 100 % design RPM.
    design_mass_flow : float
        Corrected air mass flow (kg/s) at sea-level static, 100 % RPM.
    design_rpm : float
        Design-point rotational speed (rev/min).
    """

    def __init__(
        self,
        max_temperature_k: float = 1900.0,
        design_pr: float = 10.0,
        design_mass_flow: float = 55.0,
        design_rpm: float = 100_000.0,
    ) -> None:
        self.max_temperature_k = max_temperature_k
        self.design_pr = design_pr
        self.design_mass_flow = design_mass_flow
        self.design_rpm = design_rpm

    def _compute_mass_flow(self, rpm: float, pamb: float, tamb: float) -> float:
        """Compressor inlet mass flow scaled by corrected-speed / corrected-flow relationships.

        For a compressor operating on a constant working line the corrected mass
        flow is approximately proportional to corrected speed:

            m_dot * sqrt(theta) / delta  ~  N / sqrt(theta)

        Solving for actual mass flow:

            m_dot = design_mass_flow * (N / N_design) * (delta / theta)

        where delta = Pamb / P0, theta = Tamb / T0, P0 = 101325 Pa, T0 = 288.15 K.
        """
        p0 = 101325.0
        t0 = 288.15
        delta = pamb / p0
        theta = tamb / t0
        speed_frac = max(0.1, rpm / self.design_rpm)
        return self.design_mass_flow * speed_frac * delta / max(theta, 0.1)

    def evaluate(self, value: CycleInput) -> CycleState:
        """Reconstruct cycle stations and reject nonphysical boundary conditions.

        Station sequence (see module docstring for the label convention):

            ambient --[ram compression]--> station 1 (internal, unreported)
                    --[compressor]-------> p2, t2   (Compressor Exit)
                    --[combustor]--------> p3, t3   (Combustor Exit / Turbine Inlet)
                    --[turbine]----------> p4, t4   (Turbine Exit)
                    --[nozzle]-----------> thrust
        """
        if not (
            0 <= value.mach <= 3
            and value.ambient_temperature_k > 0
            and value.ambient_pressure_pa > 0
            and value.fuel_flow_kg_s >= 0
        ):
            raise ValueError("nonphysical cycle input")
        tamb = value.ambient_temperature_k
        pamb = value.ambient_pressure_pa
        mach = value.mach

        # --- Station 1 (internal): ram compression at the inlet, ambient -> t1/p1 ---
        gamma_air = gamma_from_cp(cp_air(tamb))
        t1 = total_temperature(tamb, mach, gamma_air)
        p1 = 0.98 * total_pressure(pamb, mach, gamma_air)

        # --- Compressor: station 1 -> station 2 (p2/t2 = Compressor Exit) ---
        speed_fraction = max(0.2, min(1.15, value.rpm / self.design_rpm))
        pr = compressor_pressure_ratio(speed_fraction, value.compressor_health, self.design_pr)
        eta_c = compressor_efficiency(speed_fraction, value.compressor_health)
        p2 = p1 * pr

        gamma_c = gamma_from_cp(cp_air(0.5 * (t1 + t1 * pr ** ((gamma_air - 1) / gamma_air))))
        t2s = t1 * pr ** ((gamma_c - 1) / gamma_c)
        t2 = t1 + (t2s - t1) / max(eta_c, 0.01)

        air_flow = self._compute_mass_flow(value.rpm, pamb, tamb)
        cp_comp = cp_air(0.5 * (t1 + t2))
        compressor_work_w = air_flow * cp_comp * (t2 - t1)

        # --- Combustor: station 2 -> station 3 (p3/t3 = Combustor Exit / Turbine Inlet) ---
        far = value.fuel_flow_kg_s / max(air_flow, 1e-9)
        eta_burn = combustor_efficiency(speed_fraction, value.combustor_health)
        fuel_energy = value.fuel_flow_kg_s * LOWER_HEATING_VALUE * eta_burn
        turbine_flow = air_flow + value.fuel_flow_kg_s
        cp_burn = cp_gas(t2, far)
        t3 = t2 + fuel_energy / (turbine_flow * cp_burn)
        if t3 > self.max_temperature_k:
            t3 = self.max_temperature_k
        # Small combustor total-pressure loss (typically 2-6% for a turbojet).
        p3 = p2 * (0.96 - 0.03 * (1.0 - value.combustor_health))

        # --- Turbine: station 3 -> station 4 (p4/t4 = Turbine Exit) ---
        # Spool power balance: turbine work must supply the compressor work
        # (single-spool, no gearbox/accessory losses modeled).
        eta_t = turbine_efficiency(speed_fraction, value.turbine_health)
        cp_turb_avg = cp_gas(0.5 * (t3 + max(t3 - 400.0, 400.0)), far)
        gamma_t = gamma_from_cp(cp_turb_avg)
        t4 = t3 - compressor_work_w / (turbine_flow * max(cp_turb_avg, 1.0))
        # Iterate once on gas Cp evaluated at the actual T3/T4 midpoint.
        cp_turb_avg = cp_gas(0.5 * (t3 + t4), far)
        t4 = t3 - compressor_work_w / (turbine_flow * max(cp_turb_avg, 1.0))
        cp_turb_avg = cp_gas(0.5 * (t3 + t4), far)
        gamma_t = gamma_from_cp(cp_turb_avg)

        # Isentropic turbine exit temperature (from actual T4 and efficiency),
        # used only to derive the expansion pressure ratio.
        t4s = t3 - (t3 - t4) / max(eta_t, 0.01)
        p4_factor = max(0.05, t4s / max(t3, 1.0))
        p4 = p3 * p4_factor ** (gamma_t / (gamma_t - 1))

        # --- Nozzle: station 4 -> ambient, produces thrust ---
        exit_gamma = gamma_from_cp(cp_gas(t4, far))
        pressure_ratio_nozzle = max(pamb, 1.0) / max(p4, pamb, 1.0)
        exit_velocity = math.sqrt(
            max(
                0.0,
                2 * cp_turb_avg * t4 * (1.0 - pressure_ratio_nozzle ** ((exit_gamma - 1) / exit_gamma)),
            )
        )
        flight_velocity = mach * speed_of_sound(tamb)
        thrust = max(0.0, turbine_flow * exit_velocity - air_flow * flight_velocity)
        tsfc = value.fuel_flow_kg_s / max(thrust, 1e-9)
        turbine_work = turbine_flow * cp_turb_avg * (t3 - t4)
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
            compressor_work_w,
            turbine_work,
            turbine_work - compressor_work_w,
        )
