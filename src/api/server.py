"""FastAPI service for real-time and batch engine inference."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.digital_twin.engine import DigitalTwin
from src.explainability.root_cause import analyze_scenario
from src.explainability.shap_explainer import explain_prediction
from src.faults.injection import FaultInjector, FaultSpec, FaultType
from src.maintenance.recommendation import recommend
from src.simulation.what_if import ScenarioAdjustment, simulate_scenario

logger = logging.getLogger(__name__)


class ExplainRequest(BaseModel):
    """Request for model explanation of sensor observations."""

    observations: list["Observation"]
    feature_names: list[str] | None = None


class Observation(BaseModel):
    """Validated official online sensor payload."""

    EngineID: float = 1
    Cycle: float = Field(ge=0)
    Altitude: float
    Mach: float = Field(ge=0, le=3)
    Tamb: float = Field(gt=0)
    Pamb: float = Field(gt=0)
    RPM: float = Field(gt=0)
    FuelFlow: float = Field(ge=0)
    P2: float = Field(gt=0)
    T2: float = Field(gt=0)
    P3: float = Field(gt=0)
    T3: float = Field(gt=0)
    P4: float = Field(gt=0)
    T4: float = Field(gt=0)


class ScenarioRequest(BaseModel):
    """What-if simulation request with baseline and adjustments."""

    baseline: Observation
    fuel_flow_kg_s: float | None = None
    rpm: float | None = None
    ambient_temperature_k: float | None = None
    ambient_pressure_pa: float | None = None
    compressor_efficiency: float | None = Field(default=None, ge=0, le=1)
    turbine_efficiency: float | None = Field(default=None, ge=0, le=1)
    sensor_noise_std: float = Field(default=0.0, ge=0)


class FaultSpecRequest(BaseModel):
    """One fault to activate on an engine's injector."""

    fault_type: FaultType
    severity: float = Field(0.5, ge=0, le=1)
    target_sensor: str | None = None
    onset_cycle: float | None = None


twins: dict[str, DigitalTwin] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Optionally load the default model at application startup."""
    model_path = Path("models/best_model.joblib")
    twin = DigitalTwin()
    if model_path.exists():
        twin.load_model(model_path)
    twins["engine-1"] = twin
    yield
    twins.clear()


app = FastAPI(title="Turbojet Digital Twin API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def service_health() -> dict[str, str]:
    """Return service liveness."""
    return {"status": "ok"}


@app.post("/v1/engines/{engine_id}/update")
def update_engine(engine_id: str, observation: Observation) -> dict[str, Any]:
    """Assimilate one sensor observation for an engine."""
    try:
        twin = twins.get(engine_id)
        if twin is None:
            twin = DigitalTwin(engine_id)
            twins[engine_id] = twin
        default = twins.get("engine-1")
        if twin.model is None and default is not None:
            twin.model = default.model
        return twin.update(observation.model_dump())
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/engines/{engine_id}/batch")
def batch_engine(engine_id: str, observations: list["Observation"]) -> list[dict[str, Any]]:
    """Run ordered batch inference."""
    return [update_engine(engine_id, item) for item in observations]


@app.post("/v1/scenarios/simulate")
def run_scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Run a what-if scenario and return a before/after comparison with root cause."""
    try:
        adjustment = ScenarioAdjustment(
            fuel_flow_kg_s=request.fuel_flow_kg_s,
            rpm=request.rpm,
            ambient_temperature_k=request.ambient_temperature_k,
            ambient_pressure_pa=request.ambient_pressure_pa,
            compressor_efficiency=request.compressor_efficiency,
            turbine_efficiency=request.turbine_efficiency,
            sensor_noise_std=request.sensor_noise_std,
        )
        comparison = simulate_scenario(request.baseline.model_dump(), adjustment)
        baseline_inputs = {
            "FuelFlow": request.baseline.FuelFlow,
            "RPM": request.baseline.RPM,
            "Tamb": request.baseline.Tamb,
            "Pamb": request.baseline.Pamb,
            "compressor_efficiency": 1.0,
            "turbine_efficiency": 1.0,
        }
        adjusted_inputs = {
            "FuelFlow": adjustment.fuel_flow_kg_s or request.baseline.FuelFlow,
            "RPM": adjustment.rpm or request.baseline.RPM,
            "Tamb": adjustment.ambient_temperature_k or request.baseline.Tamb,
            "Pamb": adjustment.ambient_pressure_pa or request.baseline.Pamb,
            "compressor_efficiency": adjustment.compressor_efficiency or 1.0,
            "turbine_efficiency": adjustment.turbine_efficiency or 1.0,
        }
        report = analyze_scenario(
            baseline_inputs, adjusted_inputs, comparison["delta"]["overall_health"]
        )
        return {
            "baseline": comparison["baseline"],
            "adjusted": comparison["adjusted"],
            "delta": comparison["delta"],
            "root_cause": {
                "summary": report.summary,
                "factors": [vars(f) for f in report.factors],
                "causal_chain": report.causal_chain,
            },
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/engines/{engine_id}/faults")
def set_faults(engine_id: str, faults: list[FaultSpecRequest]) -> dict[str, Any]:
    """Replace the active fault set for an engine's digital twin."""
    twin = twins.setdefault(engine_id, DigitalTwin(engine_id))
    try:
        twin.fault_injector = FaultInjector(
            [FaultSpec(f.fault_type, f.severity, f.target_sensor, f.onset_cycle) for f in faults]
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    logger.info("engine %s faults updated: %d active", engine_id, len(faults))
    return {"engine_id": engine_id, "active_faults": twin.fault_injector.to_summary()}


@app.get("/v1/engines/{engine_id}/faults")
def get_faults(engine_id: str) -> dict[str, Any]:
    """Return the active fault set for an engine's digital twin."""
    twin = twins.setdefault(engine_id, DigitalTwin(engine_id))
    return {"engine_id": engine_id, "active_faults": twin.fault_injector.to_summary()}


@app.post("/v1/engines/{engine_id}/maintenance/recommend")
def maintenance_recommend(
    engine_id: str, health: float, rul_cycles: float, failure_probability: float
) -> dict[str, Any]:
    """Return a maintenance recommendation for the given risk indicators."""
    try:
        decision = recommend(health, rul_cycles, failure_probability)
        return {"engine_id": engine_id, "decision": vars(decision)}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/explain")
def explain_predictions(request: ExplainRequest) -> dict[str, Any]:
    """Return SHAP-based or permutation-based explanations for model predictions."""
    try:
        twin = twins.get("engine-1")
        if twin is None or twin.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        obs_dicts = [obs.model_dump() for obs in request.observations]
        frame = pd.DataFrame(obs_dicts)
        raw_features = twin.model._prepare(frame)
        pipeline = twin.model.pipeline

        def predict_fn(x: pd.DataFrame) -> np.ndarray:
            return np.asarray(pipeline.predict(x))

        explanation = explain_prediction(
            predict_fn,
            raw_features,
            feature_names=request.feature_names or list(raw_features.columns),
        )
        return explanation
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
