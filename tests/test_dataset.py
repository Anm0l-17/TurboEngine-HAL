from pipeline import demo_data
from src.dataset.features import engineer_features
from src.dataset.loader import load_dataset
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
