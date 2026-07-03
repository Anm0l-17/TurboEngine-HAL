"""Physics-informed feature generation.

``engineer_features`` keeps the original stable ratios/deltas. This module
also adds ``healthy_reference_residuals``: for each row, the Brayton-cycle
physics model (``src.physics.cycle_model.BraytonCycle``) is evaluated at
that row's flight condition (altitude, Mach, ambient state, RPM, fuel flow)
*assuming a fully healthy engine* (component health = 1.0), and the actual
measured P2/T2/P3/T3/P4/T4 are expressed as fractional residuals against
that healthy prediction.

Rationale: instantaneous sensor readings are dominated by operating
condition (altitude/Mach/RPM can swing P3/P4 by an order of magnitude),
while health degradation in this dataset only perturbs those same readings
by a few percent. A feature-based regressor trained on raw sensors has to
learn to separate a small health-driven signal from a much larger
condition-driven one with only a handful of engines to learn from, and in
practice fails to (near-zero correlation, near-zero test R² on unseen
engines even for a single-digit percent effect). Normalizing by the
healthy-engine expectation at the *same* flight condition removes almost
all of the condition-driven variance and leaves mostly the degradation
signal, which is far easier for a small feature-based model to pick up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.physics.cycle_model import BraytonCycle, CycleInput

RESIDUAL_COLUMNS = ["ResP2", "ResT2", "ResP3", "ResT3", "ResP4", "ResT4"]

_CYCLE = BraytonCycle()


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable ratios and thermodynamic deltas without mutating input."""
    out = frame.copy()
    eps = np.finfo(float).eps
    out["CompressorPR"] = out["P3"] / out["P2"].clip(lower=eps)
    out["TurbinePR"] = out["P4"] / out["P3"].clip(lower=eps)
    out["CompressorDeltaT"] = out["T3"] - out["T2"]
    out["TurbineDeltaT"] = out["T4"] - out["T3"]
    out["FuelPerRPM"] = out["FuelFlow"] / out["RPM"].clip(lower=eps)
    out["CorrectedRPM"] = out["RPM"] / np.sqrt((out["T2"] / 288.15).clip(lower=eps))
    return out


def _healthy_station_state(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    """Evaluate the Brayton cycle at this row's flight condition, health = 1.0."""
    cycle_input = CycleInput(
        altitude_m=float(row["Altitude"]),
        mach=float(row["Mach"]),
        ambient_temperature_k=float(row["Tamb"]),
        ambient_pressure_pa=float(row["Pamb"]),
        rpm=float(row["RPM"]),
        fuel_flow_kg_s=float(row["FuelFlow"]),
    )
    try:
        state = _CYCLE.evaluate(cycle_input)
    except ValueError:
        return (np.nan,) * 6
    return state.p2, state.t2, state.p3, state.t3, state.p4, state.t4


def healthy_reference_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Fractional deviation of measured stations from a healthy-engine prediction.

    Requires ``Altitude``, ``Mach``, ``Tamb``, ``Pamb``, ``RPM``, ``FuelFlow``,
    and measured ``P2``, ``T2``, ``P3``, ``T3``, ``P4``, ``T4`` columns.
    Rows with a nonphysical flight condition (rejected by ``BraytonCycle``)
    get ``NaN`` residuals, which the downstream imputer fills with the
    training median like any other missing value.
    """
    eps = np.finfo(float).eps
    healthy = frame.apply(_healthy_station_state, axis=1, result_type="expand")
    healthy.columns = ["hP2", "hT2", "hP3", "hT3", "hP4", "hT4"]
    out = pd.DataFrame(index=frame.index)
    out["ResP2"] = (frame["P2"] - healthy["hP2"]) / healthy["hP2"].abs().clip(lower=eps)
    out["ResT2"] = (frame["T2"] - healthy["hT2"]) / healthy["hT2"].abs().clip(lower=eps)
    out["ResP3"] = (frame["P3"] - healthy["hP3"]) / healthy["hP3"].abs().clip(lower=eps)
    out["ResT3"] = (frame["T3"] - healthy["hT3"]) / healthy["hT3"].abs().clip(lower=eps)
    out["ResP4"] = (frame["P4"] - healthy["hP4"]) / healthy["hP4"].abs().clip(lower=eps)
    out["ResT4"] = (frame["T4"] - healthy["hT4"]) / healthy["hT4"].abs().clip(lower=eps)
    return out[RESIDUAL_COLUMNS]


def engineer_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Full feature set used by the surrogate: ratios/deltas + physics residuals."""
    out = engineer_features(frame)
    residuals = healthy_reference_residuals(frame)
    for column in RESIDUAL_COLUMNS:
        out[column] = residuals[column]
    return out
