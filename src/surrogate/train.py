"""Surrogate construction with per-target models, target scaling, and stacking."""

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.dataset.features import RESIDUAL_COLUMNS
from src.dataset.loader import SENSOR_FEATURES, TARGETS
from src.dataset.preprocess import build_preprocessor
from .model import SurrogateModel

_ENGINEERED_COLUMNS = [
    "CompressorPR",
    "TurbinePR",
    "CompressorDeltaT",
    "TurbineDeltaT",
    "FuelPerRPM",
    "CorrectedRPM",
    "TempRatioComp",
    "TempRatioTurb",
    "OverallPR",
    "BurnerTempRise",
    "FlowSquared",
    "RPMSquared",
    "FuelFlowRPM",
    "CorrectedFuelFlow",
    *RESIDUAL_COLUMNS,
]
PIPELINE_FEATURES = SENSOR_FEATURES + _ENGINEERED_COLUMNS

_HEALTH_TARGETS = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
_PERF_TARGETS = ["Thrust", "TSFC"]


def _base_estimator(kind: str, seed: int, n_estimators: int):
    """Return a single-output estimator for the given kind."""
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2
        )
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, random_state=seed)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, max_depth=6, random_state=seed, early_stopping=False
        )
    if kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("Install xgboost to use model kind 'xgboost'") from error
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=6, random_state=seed, objective="reg:squarederror"
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def _base_estimators(seed: int, n_estimators: int) -> list[tuple[str, object]]:
    """Standard set of base estimators for stacking."""
    return [
        ("et", ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed)),
    ]


def create_model(
    kind: str = "extra_trees",
    seed: int = 42,
    n_estimators: int = 300,
    scale_targets: bool = True,
) -> SurrogateModel:
    """Construct a reproducible multi-output surrogate.

    Parameters
    ----------
    kind : str
        One of ``extra_trees``, ``random_forest``, ``gradient_boosting``,
        ``hist_gradient_boosting``, ``stacking``, ``xgboost``, ``mlp``.
    seed : int
        Random seed for reproducibility.
    n_estimators : int
        Number of trees / iterations.
    scale_targets : bool
        If True, targets are standardized before fitting.
    """
    target_scalers = {}
    if kind == "stacking":
        # Wrap a single-output StackingRegressor in MultiOutputRegressor so
        # it fits one stack per target column.
        base_estimators_list = _base_estimators(seed, n_estimators)
        estimator = MultiOutputRegressor(
            StackingRegressor(
                estimators=base_estimators_list,
                final_estimator=Ridge(alpha=1.0, random_state=seed),
                cv=5,
            ),
            n_jobs=1,
        )
    elif kind == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed, early_stopping=True
        )
    elif kind == "ensemble":
        estimator = MultiOutputRegressor(
            VotingRegressor(_base_estimators(seed, n_estimators))
        )
    elif kind == "xgboost":
        estimator = MultiOutputRegressor(_base_estimator(kind, seed, n_estimators))
    else:
        estimator = MultiOutputRegressor(_base_estimator(kind, seed, n_estimators))

    if scale_targets:
        for name in TARGETS:
            target_scalers[name] = StandardScaler()

    pipeline = Pipeline(
        [("preprocess", build_preprocessor(PIPELINE_FEATURES)), ("model", estimator)]
    )
    return SurrogateModel(
        pipeline, SENSOR_FEATURES, TARGETS, PIPELINE_FEATURES, target_scalers=target_scalers
    )
