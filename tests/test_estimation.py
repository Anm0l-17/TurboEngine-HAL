import numpy as np
import pytest

from src.estimation.ekf import ExtendedKalmanFilter
from src.estimation.particle_filter import ParticleFilter


def test_ekf_converges() -> None:
    ekf = ExtendedKalmanFilter(np.array([0.0]), np.eye(1), np.eye(1) * 0.01, np.eye(1) * 0.1)
    for _ in range(20):
        ekf.predict(lambda x: x, np.eye(1))
        ekf.update(np.array([1.0]), lambda x: x, np.eye(1))
    assert ekf.state[0] == pytest.approx(1, abs=0.05)


def test_particle_filter_tracks_measurement() -> None:
    particles = np.linspace(-2, 2, 1000).reshape(-1, 1)
    pf = ParticleFilter(particles)
    state = pf.update(np.array([0.5]), lambda x: x, np.array([0.1]))
    assert abs(state[0] - 0.5) < 0.05
