"""Dataset splitting strategies.

Two strategies are provided:

- ``grouped_split`` — holds out entire engines. Harder: the model must
  generalize degradation behaviour to engines it has never seen at all.
  Useful as a stress test, but not what the official evaluation does.
- ``official_split`` — holds out a fraction of each engine's own cycles,
  matching the actual ``train.csv``/``test.csv`` files distributed with the
  challenge (same ``EngineID`` values appear in both train and test, just
  at different ``Cycle`` values, at roughly an 80/20 per-engine ratio).
  Use this one to get metrics that match how submissions are actually
  graded.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by EngineID to prevent temporal and engine identity leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["EngineID"]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()


def official_split(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by holding out a fraction of each engine's cycles.

    Every engine appears in both the returned train and test frames; only
    the specific cycles differ. This reproduces the row counts and
    per-engine cycle distribution of the officially distributed
    ``train.csv``/``test.csv`` (each engine loses roughly ``test_size`` of
    its cycles to test, independently of the other engines).
    """
    train_parts, test_parts = [], []
    for _, group in frame.groupby("EngineID", sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        cut = max(1, int(round(len(shuffled) * test_size)))
        test_parts.append(shuffled.iloc[:cut])
        train_parts.append(shuffled.iloc[cut:])
    train = pd.concat(train_parts).sort_index()
    test = pd.concat(test_parts).sort_index()
    return train.copy(), test.copy()
