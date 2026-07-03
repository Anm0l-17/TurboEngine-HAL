"""Physics-informed feature generation."""

import numpy as np
import pandas as pd


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
