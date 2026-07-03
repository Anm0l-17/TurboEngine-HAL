"""Engine-grouped dataset splitting."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by EngineID to prevent temporal and engine identity leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["EngineID"]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
