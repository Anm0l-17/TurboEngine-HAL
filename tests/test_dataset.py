from pipeline import demo_data
from src.dataset.features import engineer_features
from src.dataset.loader import IDENTIFIER_COLUMNS, SENSOR_FEATURES, load_dataset
from src.dataset.split import grouped_split, official_split
from src.dataset.validation import validate_frame
import pytest


def test_demo_schema_and_features() -> None:
    frame = demo_data(2, 3)
    assert validate_frame(frame).valid
    assert "CompressorPR" in engineer_features(frame)


def test_validation_detects_invalid_mach() -> None:
    frame = demo_data(1, 2)
    frame.loc[0, "Mach"] = 5
    assert not validate_frame(frame).valid


def test_loader_converts_unit_suffixed_tsfc_alias(tmp_path) -> None:
    frame = demo_data(1, 1)
    expected = frame.loc[0, "TSFC"]
    aliased = frame.copy()
    aliased["TSFC_g_N_s"] = aliased.pop("TSFC") * 1000
    aliased.to_csv(tmp_path / "aliased.csv", index=False)
    loaded = load_dataset(tmp_path / "aliased.csv")
    assert loaded.loc[0, "TSFC"] == pytest.approx(expected)


def test_sensor_features_exclude_identifiers() -> None:
    """SENSOR_FEATURES must not contain EngineID or Cycle (feature leakage guard)."""
    assert "EngineID" not in SENSOR_FEATURES, "EngineID leaked into sensor features"
    assert "Cycle" not in SENSOR_FEATURES, "Cycle leaked into sensor features"
    assert "EngineID" in IDENTIFIER_COLUMNS
    assert "Cycle" in IDENTIFIER_COLUMNS


def test_grouped_split_holds_out_entire_engines() -> None:
    """grouped_split must ensure no EngineID appears in both train and test."""
    frame = demo_data(10, 30)
    train, test = grouped_split(frame, test_size=0.2, seed=42)
    train_engines = set(train["EngineID"].unique())
    test_engines = set(test["EngineID"].unique())
    overlap = train_engines & test_engines
    assert len(overlap) == 0, f"grouped_split leaked engines: {overlap}"


def test_official_split_shares_engines_across_splits() -> None:
    """official_split must include every EngineID in both train and test."""
    frame = demo_data(10, 30)
    train, test = official_split(frame, test_size=0.2, seed=42)
    train_engines = set(train["EngineID"].unique())
    test_engines = set(test["EngineID"].unique())
    assert train_engines == test_engines, "official_split must share all engines between splits"
    assert len(train_engines) == 10, f"expected 10 engines, got {len(train_engines)}"
