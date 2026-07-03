"""Surrogate construction and training."""

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from src.dataset.loader import FEATURES, TARGETS
from src.dataset.preprocess import build_preprocessor
from .model import SurrogateModel


def create_model(
    kind: str = "extra_trees", seed: int = 42, n_estimators: int = 200
) -> SurrogateModel:
    """Construct a reproducible multi-output surrogate."""
    estimators = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed, early_stopping=True
        ),
    }
    if kind not in estimators:
        raise ValueError(f"Unsupported model kind: {kind}")
    pipeline = Pipeline([("preprocess", build_preprocessor(FEATURES)), ("model", estimators[kind])])
    return SurrogateModel(pipeline, FEATURES, TARGETS)
