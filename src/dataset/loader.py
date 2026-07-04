"""Canonical turbojet dataset loader."""

from pathlib import Path
import pandas as pd

FEATURES = [
    "EngineID",
    "Cycle",
    "Altitude",
    "Mach",
    "Tamb",
    "Pamb",
    "RPM",
    "FuelFlow",
    "P2",
    "T2",
    "P3",
    "T3",
    "P4",
    "T4",
]
TARGETS = [
    "CompressorHealth",
    "CombustorHealth",
    "TurbineHealth",
    "OverallHealth",
    "Thrust",
    "TSFC",
]

# Raw exports (e.g. Kaggle-style dumps) ship unit-suffixed column names.
# Map every known variant back to the canonical schema above.
_COLUMN_ALIASES = {
    "Altitude_m": "Altitude",
    "Tamb_K": "Tamb",
    "Pamb_Pa": "Pamb",
    "RPM_rev_min": "RPM",
    "FuelFlow_kg_s": "FuelFlow",
    "P2_Pa": "P2",
    "T2_K": "T2",
    "P3_Pa": "P3",
    "T3_K": "T3",
    "P4_Pa": "P4",
    "T4_K": "T4",
    "Thrust_N": "Thrust",
    "TSFC_g_N_s": "TSFC",
}


def load_dataset(path: str | Path, require_targets: bool = True) -> pd.DataFrame:
    """Read a CSV and enforce the official numeric schema."""
    frame = pd.read_csv(path)
    tsfc_in_grams = "TSFC" not in frame.columns and "TSFC_g_N_s" in frame.columns
    frame = frame.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in frame.columns})
    required = FEATURES + (TARGETS if require_targets else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame[required] = frame[required].apply(pd.to_numeric, errors="raise")
    if tsfc_in_grams and "TSFC" in frame.columns:
        frame["TSFC"] = frame["TSFC"] / 1000.0
    if frame[required].isna().any().any():
        raise ValueError("Dataset contains missing required values")
    return frame
