"""NASA C-MAPSS turbofan dataset loader and adapter.

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) simulates
run-to-failure degradation of a 90 000 lbf turbofan engine. Four subsets
(FD001–FD004) vary in operating conditions and fault modes.

Sensor mapping to the turbojet schema is partial (C-MAPSS is a turbofan;
our physics model assumes a single-spool turbojet).  ML models use the
full 24 C-MAPSS features (3 ops + 21 sensors) directly; no physics-based
feature engineering is applied.

Download from:
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve
import numpy as np
import pandas as pd

SUBSETS: list[Literal["FD001", "FD002", "FD003", "FD004"]] = ["FD001", "FD002", "FD003", "FD004"]

# C-MAPSS column layout (space-delimited, no header)
CMAPSS_COLUMNS = (
    ["unit", "cycle", "altitude", "mach", "tra"]
    + [f"s{i}" for i in range(1, 22)]
)

# Operational-setting labels matching our sensor schema
OP_LABELS = {"altitude": "Altitude", "mach": "Mach"}

# C-MAPSS → turbojet sensor aliases (closest available)
SENSOR_ALIASES: dict[str, str] = {
    "s1": "T2",    # fan inlet temperature  → our T2
    "s2": "T24",   # LPC outlet temperature → no direct match, keep raw
    "s3": "T3",    # HPC outlet temperature → our T3
    "s4": "T50",   # LPT outlet temperature → partial match to our T4
    "s5": "P2",    # fan inlet pressure     → our P2
    "s6": "P15",   # bypass-duct pressure   → no direct match
    "s7": "P3",    # HPC outlet pressure    → our P3
    "s8": "Nf",    # fan speed              → no direct match
    "s9": "RPM",   # core speed             → our RPM
}

CMAPSS_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

# Expected filenames inside the single NASA zip
_CMAPSS_FILES: dict[str, dict[str, str]] = {
    "FD001": {
        "train": "train_FD001.txt",
        "test": "test_FD001.txt",
        "rul": "RUL_FD001.txt",
    },
    "FD002": {
        "train": "train_FD002.txt",
        "test": "test_FD002.txt",
        "rul": "RUL_FD002.txt",
    },
    "FD003": {
        "train": "train_FD003.txt",
        "test": "test_FD003.txt",
        "rul": "RUL_FD003.txt",
    },
    "FD004": {
        "train": "train_FD004.txt",
        "test": "test_FD004.txt",
        "rul": "RUL_FD004.txt",
    },
}


def download(data_dir: str | Path = "data/cmapss") -> dict[str, Path]:
    """Download all C-MAPSS subsets from NASA Open Data Portal.

    The dataset ships as a single zip containing train_FD*.txt,
    test_FD*.txt, and RUL_FD*.txt.  This function extracts each
    subset and caches as per-subset Parquet files.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Check if any subset is already cached
    have_all = all((data_dir / f"{s}_train.parquet").exists() for s in SUBSETS)
    if have_all:
        return {s: data_dir / f"{s}_train.parquet" for s in SUBSETS}

    import io
    import zipfile
    from urllib.request import urlopen

    print(f"  Downloading {CMAPSS_ZIP_URL} ...")
    try:
        resp = urlopen(CMAPSS_ZIP_URL, timeout=180)
        raw = resp.read()
    except Exception as exc:
        print(f"  WARN: download failed: {exc}")
        print(f"  Place the C-MAPSS text files manually in {data_dir}/")
        return {}

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        for subset, files in _CMAPSS_FILES.items():
            for key, fname in files.items():
                if fname in names:
                    text = zf.read(fname).decode()
                    if key == "rul":
                        df = pd.read_csv(io.StringIO(text), header=None, names=["RUL"])
                        dest = data_dir / f"{subset}_RUL.parquet"
                    else:
                        df = _read_cmapss_txt(io.StringIO(text))
                        dest = data_dir / f"{subset}_{key}.parquet"
                    df.to_parquet(dest)
                    print(f"    {fname} -> {dest.name}")
            paths[subset] = data_dir / f"{subset}_train.parquet"
            print(f"  {subset} ready")

    return paths


def _read_cmapss_txt(source) -> pd.DataFrame:
    """Parse a C-MAPSS space-delimited text file into a DataFrame."""
    df = pd.read_csv(source, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="c")
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert altitude from kft to m (matches our schema)
    if "altitude" in df.columns:
        df["altitude_m"] = df["altitude"] * 304.8
    # Add T2 sensor alias for convenience
    if "s1" in df.columns:
        df["T2"] = df["s1"]
    if "s3" in df.columns:
        df["T3"] = df["s3"]
    if "s5" in df.columns:
        df["P2"] = df["s5"]
    if "s7" in df.columns:
        df["P3"] = df["s7"]
    if "s9" in df.columns:
        df["RPM"] = df["s9"]
    return df


def load_subset(
    subset: Literal["FD001"] | Literal["FD002"] | Literal["FD003"] | Literal["FD004"],
    data_dir: str | Path = "data/cmapss",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load a C-MAPSS subset as (train, test, rul).

    Returns:
        train: training DataFrame with all features
        test: test DataFrame (full run-to-failure for each unit)
        rul: RUL ground truth for test units (None if unavailable)
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"{subset}_train.parquet"
    test_path = data_dir / f"{subset}_test.parquet"
    rul_path = data_dir / f"{subset}_RUL.parquet"

    train = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    test = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()
    rul = pd.read_parquet(rul_path) if rul_path.exists() else None

    return train, test, rul


def add_rul_column(test: pd.DataFrame, rul: pd.DataFrame) -> pd.DataFrame:
    """Add RUL (remaining useful life in cycles) as a per-engine column.

    C-MAPSS test files contain the full run-to-failure trajectory for each
    test unit.  The RUL file gives remaining cycles at *the last row* of
    each unit.  This function backfills the monotonically decreasing RUL
    for every row.
    """
    out = test.copy()
    last_rul = rul["RUL"].values
    engine_ids = sorted(out["unit"].unique())
    if len(last_rul) != len(engine_ids):
        raise ValueError(
            f"RUL file has {len(last_rul)} entries but test data has {len(engine_ids)} engines"
        )
    for eid, rul_val in zip(engine_ids, last_rul):
        mask = out["unit"] == eid
        n_cycles = mask.sum()
        # RUL decreases monotonically: earliest row has most remaining life
        out.loc[mask, "RUL"] = np.arange(rul_val + n_cycles - 1, rul_val - 1, -1, dtype=float)
    return out


def cmapss_feature_columns() -> list[str]:
    """Return all C-MAPSS columns suitable as ML features (excludes unit, cycle)."""
    return [c for c in CMAPSS_COLUMNS if c not in ("unit", "cycle")] + ["altitude_m"]


def _add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1 and delta features per engine. Returns copy."""
    out = frame.copy()
    feature_cols = [c for c in frame.columns if c.startswith("s")] + ["altitude", "mach", "tra"]
    gb = frame[feature_cols + ["unit"]].groupby("unit")
    for c in feature_cols:
        shifted = gb[c].shift(1).fillna(frame[c])
        out[f"{c}_lag1"] = shifted
        out[f"{c}_delta"] = frame[c] - shifted
    return out


def prepare_ml_data(
    train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare feature matrix and RUL target for ML training/evaluation.

    Adds simple temporal features (lag-1, delta) per engine to capture
    degradation trends.  Returns:
        X_train, y_train, X_test, y_test
    """
    base_cols = cmapss_feature_columns()

    train_feat = _add_temporal_features(train)
    test_feat = _add_temporal_features(test)
    feature_cols = [c for c in train_feat.columns if c in test_feat.columns and c not in ("unit", "cycle")]

    X_train = train_feat[feature_cols].copy()
    y_train = train.groupby("unit").cumcount(ascending=False).values.astype(float)

    if rul is not None:
        test_with_rul = add_rul_column(test, rul)
        y_test = test_with_rul["RUL"].values
    else:
        y_test = test.groupby("unit").cumcount(ascending=False).values.astype(float)

    X_test = test_feat[feature_cols].copy()
    return X_train, y_train, X_test, y_test


def benchmark_subset(
    subset: str,
    data_dir: str | Path = "data/cmapss",
) -> dict[str, float]:
    """Run a quick benchmark of all tree-based models on a C-MAPSS subset.

    Returns {model_kind: rmse}.
    """
    from src.surrogate.train import create_model

    train, test, rul = load_subset(subset, data_dir)
    if train.empty:
        return {}

    X_train, y_train, X_test, y_test = prepare_ml_data(train, test, rul)

    results: dict[str, float] = {}
    for kind in ["extra_trees", "hist_gradient_boosting", "random_forest"]:
        model = create_model(kind, n_estimators=200, scale_targets=False)
        # Fit univariate regressor (RUL is single-target)
        model.pipeline.fit(X_train, y_train)
        preds = model.pipeline.predict(X_test)
        from sklearn.metrics import mean_squared_error

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        results[kind] = rmse
    return results
