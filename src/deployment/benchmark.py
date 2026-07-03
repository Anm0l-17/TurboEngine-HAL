"""CPU inference latency benchmark."""

from collections.abc import Callable
from time import perf_counter
from typing import Any
import numpy as np


def benchmark_latency(
    function: Callable[[Any], Any], payload: Any, runs: int = 100
) -> dict[str, float]:
    """Measure warm CPU latency distribution in milliseconds."""
    function(payload)
    samples = []
    for _ in range(runs):
        start = perf_counter()
        function(payload)
        samples.append((perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "runs": float(runs),
    }
