from pipeline import demo_data
from src.digital_twin.engine import DigitalTwin


def test_physics_fallback_and_persistence(tmp_path) -> None:
    frame = demo_data(1, 3)
    twin = DigitalTwin("E1")
    result = twin.batch_predict(frame)
    assert len(result) == 3
    assert result["OverallHealth"].between(0, 1).all()
    path = tmp_path / "state.json"
    twin.save_state(path)
    restored = DigitalTwin().load_state(path)
    assert len(restored.history) == 3
