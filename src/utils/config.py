"""Typed application configuration."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str = "data/turbojet.csv"
    test_size: float = Field(0.2, gt=0, lt=1)


class ModelConfig(BaseModel):
    kind: str = "extra_trees"
    n_estimators: int = Field(200, ge=10)


class PhysicsConfig(BaseModel):
    max_temperature_k: float = Field(1900.0, gt=1000)
    compressor_pressure_ratio: float = Field(10.0, gt=1)


class RuntimeConfig(BaseModel):
    drift_threshold: float = Field(0.12, gt=0)
    failure_health_threshold: float = Field(0.3, gt=0, lt=1)


class Settings(BaseModel):
    seed: int = 42
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    physics: PhysicsConfig = PhysicsConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def load_config(path: str | Path = "config.yaml") -> Settings:
    """Load and validate YAML settings."""
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return Settings.model_validate(raw)
