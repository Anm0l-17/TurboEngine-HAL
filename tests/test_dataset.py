from pipeline import demo_data
from src.dataset.features import engineer_features
from src.dataset.validation import validate_frame


def test_demo_schema_and_features() -> None:
    frame = demo_data(2, 3)
    assert validate_frame(frame).valid
    assert "CompressorPR" in engineer_features(frame)


def test_validation_detects_invalid_mach() -> None:
    frame = demo_data(1, 2)
    frame.loc[0, "Mach"] = 5
    assert not validate_frame(frame).valid
