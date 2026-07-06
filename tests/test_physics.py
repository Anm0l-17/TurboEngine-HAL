import numpy as np
import pytest
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.physics.component_maps import compressor_pressure_ratio

# A representative set of flight conditions drawn from the dataset's actual range.
# (sea level static, cruise mid, cruise high, climb, descent)
_REPRESENTATIVE_CONDITIONS = [
    # (Alt, Mach, Tamb, Pamb, RPM, FuelFlow, label)
    (0, 0.0, 288.15, 101325, 45000, 0.45, "sea_level_idle"),
    (0, 0.2, 288.15, 101325, 65000, 0.80, "sea_level_takeoff"),
    (3000, 0.4, 268.65, 70121, 72000, 0.95, "cruise_mid"),
    (6000, 0.6, 249.15, 47181, 78000, 1.10, "cruise_high"),
    (9000, 0.7, 229.65, 30743, 82000, 1.30, "high_alt_cruise"),
]


def test_cycle_is_physical() -> None:
    """Station convention: p2/t2 = Compressor Exit, p3/t3 = Combustor Exit
    (Turbine Inlet), p4/t4 = Turbine Exit — matches the challenge's dataset
    schema, not the classical Brayton numbering. See cycle_model.py module
    docstring. Combustor has a small total-pressure LOSS (p3 slightly below
    p2), and a large temperature RISE from fuel burn (t3 >> t2)."""
    state = BraytonCycle().evaluate(CycleInput(0, 0.2, 288.15, 101325, 90000, 0.8))
    assert state.p2 > 0
    assert 0 < state.p3 <= state.p2, "combustor should show a small pressure loss, not a rise"
    assert state.t3 > state.t2 > 0, "combustor should sharply raise temperature (t3 >> t2)"
    assert state.p4 < state.p3, "turbine should expand (drop pressure) from t3/p3 to t4/p4"
    assert state.t4 < state.t3, "turbine should extract energy (drop temperature) from t3 to t4"
    assert state.thrust_n >= 0
    assert abs(state.energy_residual_w) < state.compressor_work_w * 0.5
    assert 0 <= state.thermal_efficiency <= 1


def test_cycle_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        BraytonCycle().evaluate(CycleInput(0, -1, 288, 101325, 90000, 1))


def test_cycle_tsfc_in_reasonable_range() -> None:
    """TSFC for a turbojet should typically be 10-50 mg/N·s (1e-5 to 5e-5 kg/N·s)."""
    for alt, mach, tamb, pamb, rpm, fuel, label in _REPRESENTATIVE_CONDITIONS:
        state = BraytonCycle().evaluate(CycleInput(alt, mach, tamb, pamb, rpm, fuel))
        msg = f"TSFC={state.tsfc_kg_n_s:.2e} out of range at {label}"
        assert 5e-6 < state.tsfc_kg_n_s < 5e-4, msg


def test_cycle_thrust_in_reasonable_range() -> None:
    """Thrust should be positive and scale with fuel flow, altitude etc."""
    sea_level = BraytonCycle().evaluate(CycleInput(0, 0.0, 288.15, 101325, 65000, 0.8))
    high_alt = BraytonCycle().evaluate(CycleInput(9000, 0.7, 229.65, 30743, 82000, 1.3))
    assert sea_level.thrust_n > 0
    assert high_alt.thrust_n > 0
    # Higher fuel flow at high altitude should produce reasonable thrust
    thrust_ratio = high_alt.thrust_n / sea_level.thrust_n
    assert 0.3 < thrust_ratio < 3.0, f"thrust ratio {thrust_ratio} across extreme conditions"


def test_compressor_pr_map_normalised() -> None:
    """The compressor pressure ratio map should give ~design_pr at design speed."""
    pr_at_design = compressor_pressure_ratio(1.0, health=1.0, design_pr=10.0)
    assert 8.0 <= pr_at_design <= 11.0, f"PR at design speed = {pr_at_design}, expected ~10"
    # Off-design PR should be lower
    pr_off = compressor_pressure_ratio(0.5, health=1.0, design_pr=10.0)
    assert pr_off < pr_at_design, f"off-design PR {pr_off} >= design PR {pr_at_design}"
    # Degraded PR should be lower than healthy
    pr_degraded = compressor_pressure_ratio(1.0, health=0.7, design_pr=10.0)
    assert pr_degraded < pr_at_design, f"degraded PR {pr_degraded} >= healthy PR {pr_at_design}"


def test_cycle_energy_balance() -> None:
    """The sum of compressor work and turbine work should approximately match."""
    state = BraytonCycle().evaluate(CycleInput(0, 0.0, 288.15, 101325, 65000, 0.8))
    # Energy imbalance should be small relative to either work term
    imbalance_frac = abs(state.energy_residual_w) / max(state.compressor_work_w, 1.0)
    assert imbalance_frac < 0.15, f"energy imbalance {imbalance_frac:.3f}"
