"""Portable model export helpers."""

from pathlib import Path
from typing import Any


def export_sklearn_onnx(model: Any, feature_count: int, destination: str | Path) -> Path:
    """Export a compatible scikit-learn pipeline using skl2onnx when installed."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as error:
        raise RuntimeError("Install skl2onnx to enable ONNX export") from error
    artifact = convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, feature_count]))]
    )
    path = Path(destination)
    path.write_bytes(artifact.SerializeToString())
    return path
