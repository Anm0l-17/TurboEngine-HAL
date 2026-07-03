"""Repository path management."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def ensure_directories() -> None:
    """Create mutable artifact directories."""
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
