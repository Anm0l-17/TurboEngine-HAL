"""Bootstrap sequential Monte Carlo filter."""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray


class ParticleFilter:
    """Particle filter with systematic resampling and Gaussian likelihood."""

    def __init__(self, particles: NDArray[np.float64], seed: int = 42) -> None:
        self.particles = np.asarray(particles, dtype=float)
        self.weights = np.full(len(self.particles), 1 / len(self.particles))
        self.rng = np.random.default_rng(seed)

    @property
    def state(self) -> NDArray[np.float64]:
        """Return weighted posterior mean."""
        return np.average(self.particles, axis=0, weights=self.weights)

    def predict(
        self,
        transition: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        noise_std: NDArray[np.float64],
    ) -> None:
        """Propagate particles and add process noise."""
        self.particles = np.asarray([transition(p) for p in self.particles])
        self.particles += self.rng.normal(0, noise_std, self.particles.shape)

    def update(
        self,
        measurement: NDArray[np.float64],
        observation: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        noise_std: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Weight and resample particles from a Gaussian sensor model."""
        residual = np.asarray([measurement - observation(p) for p in self.particles]) / noise_std
        log_weights = -0.5 * np.sum(residual**2, axis=1)
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum() + np.finfo(float).eps
        if 1 / np.sum(self.weights**2) < len(self.weights) / 2:
            positions = (self.rng.random() + np.arange(len(self.weights))) / len(self.weights)
            cumulative = np.cumsum(self.weights)
            indices = np.searchsorted(cumulative, positions)
            self.particles = self.particles[indices]
            self.weights.fill(1 / len(self.weights))
        return self.state
