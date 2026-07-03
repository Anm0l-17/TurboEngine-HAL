"""Engine-grouped cross-validation."""

import pandas as pd
from sklearn.model_selection import GroupKFold
from src.dataset.loader import TARGETS
from src.metrics.regression import regression_metrics
from src.surrogate.train import create_model


def cross_validate(
    frame: pd.DataFrame, folds: int = 5, kind: str = "extra_trees"
) -> list[dict[str, float]]:
    """Evaluate a model without leaking an engine across folds."""
    unique = frame["EngineID"].nunique()
    splitter = GroupKFold(n_splits=min(folds, unique))
    scores = []
    for train_idx, test_idx in splitter.split(frame, groups=frame["EngineID"]):
        model = create_model(kind).fit(frame.iloc[train_idx])
        pred = model.predict(frame.iloc[test_idx])
        scores.append(regression_metrics(frame.iloc[test_idx][TARGETS].to_numpy(), pred.to_numpy()))
    return scores
