import pytest
from src.physics.cycle_model import BraytonCycle, CycleInput


def test_cycle_is_physical() -> None:
    state = BraytonCycle().evaluate(CycleInput(0, 0.2, 288.15, 101325, 90000, 0.8))
    assert state.p3 > state.p2 > 0
    assert state.t3 > state.t2 > 0
    assert state.thrust_n >= 0
    assert abs(state.energy_residual_w) < 1e-6
    assert 0 <= state.thermal_efficiency <= 1


def test_cycle_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        BraytonCycle().evaluate(CycleInput(0, -1, 288, 101325, 90000, 1))
