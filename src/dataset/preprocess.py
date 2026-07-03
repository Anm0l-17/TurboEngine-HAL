"""Leakage-safe numerical preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Build a robust numerical transformer fitted on training data only."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    return ColumnTransformer([("numeric", numeric, columns)], remainder="drop")
