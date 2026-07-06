from pipeline import demo_data
from src.training.trainer import train_from_csv
from src.surrogate.model import SurrogateModel
from src.surrogate.train import create_model
from src.dataset.loader import SENSOR_FEATURES, IDENTIFIER_COLUMNS


def test_train_and_load(tmp_path) -> None:
    data = tmp_path / "data.csv"
    model = tmp_path / "model.joblib"
    frame = demo_data(5, 12)
    frame.to_csv(data, index=False)
    metrics = train_from_csv(data, model)
    assert model.exists()
    assert metrics["rmse"] >= 0
    loaded = SurrogateModel.load(model)
    assert len(loaded.predict(frame.head())) == 5
    prediction, lower, upper, confidence = loaded.predict_with_uncertainty(frame.head())
    assert confidence == 0.9
    assert ((lower <= prediction) & (prediction <= upper)).all().all()


def test_model_feature_names_are_sensors_only() -> None:
    """SurrogateModel.feature_names must not include EngineID or Cycle."""
    model = create_model("extra_trees")
    assert "EngineID" not in model.feature_names, "EngineID leaked into model features"
    assert "Cycle" not in model.feature_names, "Cycle leaked into model features"
    assert set(model.feature_names) == set(SENSOR_FEATURES), (
        f"Model features {model.feature_names} don't match SENSOR_FEATURES"
    )


def test_model_ignores_identifiers_in_pipeline() -> None:
    """SurrogateModel._prepare uses SENSOR_FEATURES (not EngineID/Cycle) for the pipeline."""
    from src.surrogate.train import create_model
    from src.dataset.features import engineer_all_features
    frame = demo_data(3, 10)
    model = create_model("extra_trees")
    prepared = model._prepare(frame)
    # EngineID and Cycle should NOT appear in prepared features
    assert "EngineID" not in prepared.columns, "EngineID leaked into pipeline features"
    assert "Cycle" not in prepared.columns, "Cycle leaked into pipeline features"
    # Sensor features should be present (after feature engineering)
    for col in SENSOR_FEATURES:
        assert col in prepared.columns, f"SENSOR_FEATURE {col} missing from pipeline input"
