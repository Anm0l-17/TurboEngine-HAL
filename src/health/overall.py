"""Safety-conservative overall health fusion."""

import numpy as np


def overall_health(compressor: float, combustor: float, turbine: float) -> float:
    """Return weighted geometric mean, penalizing a weak subsystem."""
    values = np.clip([compressor, combustor, turbine], 1e-8, 1)
    return float(np.exp(np.dot([0.35, 0.25, 0.4], np.log(values))))
