"""Physical and relational dataset validation."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Validation status and human-readable violations."""

    valid: bool
    errors: tuple[str, ...]


def validate_frame(frame: pd.DataFrame) -> ValidationResult:
    """Check physical ranges, identity, and cycle ordering."""
    errors: list[str] = []
    ranges = {
        "Mach": (0, 3),
        "RPM": (0, 200_000),
        "FuelFlow": (0, 20),
        "Tamb": (150, 400),
        "Pamb": (1_000, 120_000),
    }
    for name, (low, high) in ranges.items():
        if name in frame and not frame[name].between(low, high).all():
            errors.append(f"{name} outside [{low}, {high}]")
    for name in ("CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"):
        if name in frame and not frame[name].between(0, 1).all():
            errors.append(f"{name} outside [0, 1]")
    if {"EngineID", "Cycle"}.issubset(frame) and frame.duplicated(["EngineID", "Cycle"]).any():
        errors.append("duplicate EngineID/Cycle rows")
    return ValidationResult(not errors, tuple(errors))
