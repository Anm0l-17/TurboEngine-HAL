"""FastAPI service for real-time and batch inference."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.digital_twin.engine import DigitalTwin


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
        twin = twins.setdefault(engine_id, DigitalTwin(engine_id))
        default = twins.get("engine-1")
        if twin.model is None and default is not None:
            twin.model = default.model
        return twin.update(observation.model_dump())
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/engines/{engine_id}/batch")
def batch_engine(engine_id: str, observations: list[Observation]) -> list[dict[str, Any]]:
    """Run ordered batch inference."""
    return [update_engine(engine_id, item) for item in observations]
