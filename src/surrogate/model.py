"""Serializable multi-output surrogate model."""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SurrogateModel:
    """Column-aware wrapper around a fitted scikit-learn pipeline."""

    def __init__(
        self, pipeline: Pipeline, feature_names: list[str], target_names: list[str]
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.target_names = target_names

    def fit(self, frame: pd.DataFrame) -> "SurrogateModel":
        """Fit inputs to all configured targets."""
        self.pipeline.fit(frame[self.feature_names], frame[self.target_names])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return named predictions."""
        values = np.asarray(self.pipeline.predict(frame[self.feature_names]))
        return pd.DataFrame(values, columns=self.target_names, index=frame.index)

    def save(self, path: str | Path) -> None:
        """Atomically serialize the model payload."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        """Load and type-check a serialized model."""
        model: Any = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a SurrogateModel")
        return model
