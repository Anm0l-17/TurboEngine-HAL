<div align="center">

# ⚙️ Turbojet Digital Twin

**Physics-informed digital twin for real-time four-stage turbojet health monitoring, RUL prediction, and fleet-scale condition-based maintenance.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](src/api/server.py)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](src/viz/dashboard.py)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](pyproject.toml)

[Overview](#overview) · [Architecture](#architecture) · [Quick Start](#quick-start) · [CLI Reference](#cli-reference) · [API](#api-reference) · [Project Layout](#project-layout) · [Deployment](#deployment) · [Testing](#testing--quality)

</div>

---

## Overview

This repository implements a **production-oriented digital twin** for a four-stage turbojet engine. It fuses a physically constrained Brayton-cycle model (with variable specific heats, ISA standard atmosphere, and 4th-order component efficiency maps) with a learned surrogate through Bayesian state estimation (EKF / UKF), producing calibrated health trajectories that drive RUL prediction, failure-risk scoring, and condition-based maintenance recommendations.

| Capability | Description |
|---|---|
| 🔥 **Physics core** | Brayton-cycle thermodynamic reconstruction, variable Cp(T), ISA atmosphere, 4th-order component maps, spool energy balance |
| 📡 **State estimation** | EKF (extended Kalman) and UKF (unscented Kalman) with identity observation Jacobian, constant-degradation prior |
| 🧠 **Learned surrogate** | Multi-output health/performance regression (HGBT, ExtraTrees, RandomForest, Stacking, Hybrid Physics+ML); trained on 34 physics-residual and engineered features; target scaling per output |
| 🔬 **Hybrid Physics+ML** | ML learns the *residual* from the physics model — `prediction = physics + ml_residual`. Preserves physical grounding while correcting systematic bias |
| 📏 **Uncertainty** | Conformal prediction · quantile regression · adaptive conformal (locally-weighted) · bootstrapped ensemble — three modes selectable per model |
| 🔍 **Explainability** | SHAP integration for tree-based models with graceful fallback to permutation importance; feature interaction matrix; API + dashboard pages |
| ⏳ **Prognostics** | RUL quantiles, data-calibrated failure probability (logistic regression on degradation trajectories), thrust & fuel-efficiency forecasts |
| 🛠️ **Maintenance** | CBM scheduler with multi-option decision engine (monitor → inspect → repair → overhaul → replace) scored on cost, downtime, risk, RUL gain |
| 🚦 **Fleet ops** | Cross-engine ranking, drift monitoring, comparative analytics, health correlation matrix |
| 🌐 **Serving** | Stateful real-time + batch FastAPI service, 18-page Streamlit dashboard, Markdown report generation |
| 🧪 **What-if simulator** | Adjust fuel flow, RPM, ambient conditions, component efficiency, sensor noise; instant before/after comparison |
| ⚠️ **Fault injection** | Compressor fouling, turbine erosion, fuel nozzle blockage, bearing wear, sensor drift/bias |
| 🧪 **Experiment framework** | `run_experiment()` / `ablation_study()` with versioned configs, metrics, artifacts; Markdown report generator |
| 📊 **Validation & Benchmark** | Automated validation suite (per-target RMSE/MAE/R²/MAPE across split strategies); latency/throughput/memory benchmarks |
| 💡 **Dashboard pages (18)** | Overview, Engine Health, Performance, RUL & Risk, Trade-Off Analysis, Parameter Sweep, Calibration Analysis, Degradation Analysis, Correlation Analysis, Fleet Comparison, Model Explainability (SHAP), What-If Simulator, Fault Injection, Root Cause Analysis, Maintenance, Maintenance Options, Upload & Inference, Settings |

---

## Architecture

```mermaid
flowchart TB
    subgraph INGEST["📥 Ingestion"]
        A[Sensor Stream<br/>P2·T2·P3·T3·P4·T4·RPM·Fuel] --> B["Schema & Range<br/>Validation"]
    end

    subgraph CORE["🧬 Digital Twin Core"]
        direction LR
        B --> C[⚡ Brayton-Cycle<br/>Physics Model]
        B --> D[🧠 Learned<br/>Surrogate]
        D -.-> D2[Hybrid<br/>Physics+ML]
        C --> E{{Bayesian State<br/>Estimator<br/>EKF · UKF}}
        D --> E
        E --> F[Calibrated Health<br/>Trajectory + Uncertainty<br/>Conformal · Quantile · Ensemble]
    end

    subgraph EXPLAIN["🔍 Explainability"]
        F --> S[SHAP / Permutation<br/>Importance]
        S --> T[Global Feature<br/>Importance]
        S --> U["Local Explanations<br/>& Interaction Matrix"]
    end

    subgraph INTEL["📊 Prognostics & Decisioning"]
        F --> G[RUL Quantiles<br/>Configurable Thresholds]
        F --> H[Failure Probability<br/>Data-Calibrated]
        F --> I[Thrust / TSFC<br/>Forecast]
        G & H & I --> J[Condition-Based<br/>Maintenance Engine]
        J --> K["Maintenance Economics<br/>& Recommendations"]
    end

    subgraph SERVE["🌐 Serving Layer"]
        F --> L[DigitalTwin Facade]
        L --> M[FastAPI<br/>real-time + batch]
        L --> N[Streamlit<br/>Dashboard · 18 pages]
        L --> O[Report Generator<br/>MD]
        L --> P[Fleet Analytics<br/>ranking · drift]
        L --> Q[Model Export<br/>joblib]
    end

    subgraph OPS["⚙️ Pipeline & Validation"]
        R[CLI · pipeline.py] --> M
        R --> N
        R --> V[Validation Suite<br/>· 4 model kinds<br/>· 2 split strategies]
        R --> W["Benchmark Suite<br/>· latency (p50/p95/p99)<br/>· throughput · memory"]
    end

    style CORE fill:#1a1a2e,stroke:#e94560,color:#fff
    style INTEL fill:#16213e,stroke:#0f4c75,color:#fff
    style SERVE fill:#0f3460,stroke:#3282b8,color:#fff
    style INGEST fill:#222,stroke:#888,color:#fff
    style EXPLAIN fill:#2d1b2e,stroke:#c74b8a,color:#fff
    style OPS fill:#1a2e1a,stroke:#4bc74b,color:#fff
```

**Design principle:** every consumer (CLI, API, dashboard, fleet workflows) talks to one shared `DigitalTwin` facade — physics, surrogate, hybrid, and uncertainty modules stay decoupled; state stays JSON-safe; models and reports are versioned artifacts.

---

## Dashboard Preview

<div align="center">

| Live Health Gauge | Feature Importance |
|---|---|
| ![Engine Health Gauge](docs/assets/screenshots/engine_health_gauge.png) | ![Feature Importance](docs/assets/screenshots/feature_importance_chart.png) |

| Health Trajectories | Thrust / TSFC Trends |
|---|---|
| ![Health Trajectories](docs/assets/screenshots/health_trajectories_dashboard.png) | ![Thrust TSFC Trends](docs/assets/screenshots/thrust_tsfc_trends.png) |
</div>

The dashboard includes **18 pages** accessible from the sidebar. Key additions beyond standard monitoring:
- **Model Explainability** — SHAP global importance bar chart, local waterfall per row, feature interaction heatmap
- **Calibration Analysis** — conformal prediction coverage diagnostics
- **Degradation Analysis** — per-component degradation rate comparison across fleet
- **Correlation Analysis** — sensor/health correlation heatmap
- **Fleet Comparison** — side-by-side health, RUL, and risk ranking
- **Trade-Off Analysis** — Pareto frontier of cost vs. risk vs. remaining life
- **Parameter Sweep** — sweep any operating parameter and observe health/RUL response

---

## Quick Start

```powershell
# 1. Environment
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,api,dashboard,reports]"
pip install shap psutil   # optional: SHAP explainability + performance benchmarks

# 2. Smoke-test the full pipeline
python pipeline.py demo
pytest -m "not slow"

# 3. Train with any model kind
python pipeline.py train --data data/turbojet_complete_dataset.csv --kind hybrid --output models/hybrid.joblib
python pipeline.py train --data data/turbojet_complete_dataset.csv --kind hist_gradient_boosting --output models/hgb.joblib
python pipeline.py train --data data/turbojet_complete_dataset.csv --kind extra_trees --output models/et.joblib
python pipeline.py train --data data/turbojet_complete_dataset.csv --kind stacking --output models/stacking.joblib

# 4. Full orchestration (train all variants + validate + benchmark)
python pipeline.py orchestrate --data data/turbojet_complete_dataset.csv --output-dir results

# 5. Run validation & benchmark suites
python pipeline.py validation --data data/turbojet_complete_dataset.csv
python pipeline.py benchmark --data data/turbojet_complete_dataset.csv

# 6. Experiment tracking & ablation
python pipeline.py experiment --data data/turbojet_complete_dataset.csv --kind hist_gradient_boosting --tag "v2-hgb"
python pipeline.py ablation --data data/turbojet_complete_dataset.csv
python pipeline.py report --input-dir results/experiments

# 7. Serve
uvicorn src.api.server:app --reload          # REST API  → http://localhost:8000
streamlit run src/viz/dashboard.py           # Dashboard → http://localhost:8501
```

**Split strategies:** `--strategy official` (default) holds out a fraction of each engine's own cycles — use this for metrics comparable to official grading. `--strategy grouped` holds out entire engines instead, a harder generalization stress test.

**Available model kinds:** `hist_gradient_boosting`, `extra_trees`, `random_forest`, `gradient_boosting`, `stacking`, `xgboost`, `mlp`, `hybrid` (physics + ML residual).

---

## CLI Reference

`pipeline.py` is the single entrypoint:

| Command | Purpose |
|---|---|
| `train` | Train a single model variant |
| `tune` | Grid-search hyperparameters across model kinds |
| `evaluate` | Evaluate a saved model on held-out data |
| `predict` | Run batch inference |
| `experiment` | Run a logged experiment with config + metrics |
| `ablation` | Cross-model ablation study |
| `report` | Generate Markdown report from experiment results |
| `validation` | Full validation suite (4 model kinds × 2 splits) |
| `benchmark` | Latency/throughput/memory benchmarks |
| `orchestrate` | End-to-end: train all → validate → benchmark |
| `demo` | Generate synthetic data + train baseline model |

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness / readiness probe |
| `POST` | `/v1/engines/{engine_id}/update` | Push a single sensor reading, get updated health state |
| `POST` | `/v1/engines/{engine_id}/batch` | Push a batch of cycles for one engine |
| `POST` | `/v1/scenarios/simulate` | What-if simulation: before/after health, RUL, risk, thrust, TSFC, confidence + root cause |
| `POST` | `/v1/engines/{engine_id}/faults` | Replace the active fault set |
| `GET` | `/v1/engines/{engine_id}/faults` | Read the active fault set |
| `POST` | `/v1/engines/{engine_id}/maintenance/options` | Ranked maintenance options |
| `POST` | `/v1/explain` | SHAP / permutation explanations for model predictions |

The service is **stateful per engine** — each update advances that engine's Bayesian estimator and health trajectory in memory.

---

## Project Layout

```
digital_twin/
├── pipeline.py              # CLI: train · tune · evaluate · predict · experiment ·
│                            #        ablation · report · validation · benchmark ·
│                            #        orchestrate · demo
├── config.yaml              # Seed, data, model, physics & runtime thresholds
│
├── src/
│   ├── physics/              # Brayton-cycle · variable Cp · ISA atmosphere · component maps
│   ├── estimation/           # EKF · UKF state estimators
│   ├── surrogate/             # SurrogateModel · create_model · HybridPhysicsMLModel
│   ├── uncertainty/           # Quantile regression · conformal · adaptive conformal
│   ├── explainability/        # SHAP explainer · root cause analysis
│   ├── validation/            # Cross-model validation suite (new)
│   ├── performance/           # Latency/throughput benchmarks (new)
│   ├── health/                # Compressor / combustor / turbine / overall health fusion
│   ├── prediction/            # RUL · failure probability · thrust · fuel efficiency
│   ├── maintenance/           # CBM scheduler, economics, multi-option decision engine
│   ├── faults/                # Fault injection (component + sensor)
│   ├── simulation/            # What-if scenario simulator
│   ├── digital_twin/          # DigitalTwin facade · fleet ranking
│   ├── dataset/               # Loader · preprocessing · feature engineering · splits
│   ├── training/              # Trainer · cross-validation · hyperparameter search
│   ├── metrics/               # Regression, uncertainty & health metrics
│   ├── research/              # Experiment runner · ablation study
│   ├── report/                # Markdown research report generator
│   ├── viz/                   # 18-page Streamlit dashboard · plots · engine animation
│   ├── api/                   # FastAPI service (8 endpoints)
│   ├── deployment/            # Model export (ONNX) + inference benchmarking
│   └── utils/                 # Config · logging · paths · seeding · timers
│
├── tests/                    # 52 tests (48 fast + 4 slow)
│   ├── test_integration.py   # Hybrid model, SHAP, validation & benchmark suites
│   ├── test_api.py           # FastAPI endpoints
│   ├── test_twin.py          # DigitalTwin facade
│   ├── test_training.py      # Training workflow
│   ├── test_physics.py       # Brayton cycle
│   ├── ...                   # Dashboard, faults, estimation, maintenance, etc.
│
├── docs/
│   ├── ARCHITECTURE.md       # Design notes
│   └── DATA.md               # Dataset schema contract
│
├── data/  · models/  · results/     # Datasets, trained artifacts, run outputs
├── Dockerfile · docker-compose.yml
└── pyproject.toml · requirements.txt
```

---

## Deployment

```bash
docker compose up --build
```

- Multi-stage single image (`python:3.12-slim`), runs as non-root `twin`
- Exposes FastAPI service on **:8000** with `/health` healthcheck
- Trained models mounted read-only from `./models`

---

## Testing & Quality

```bash
pytest -m "not slow"          # 48 fast tests (≈60 s)
pytest                        # Full suite incl. validation/benchmark (≈5 min)
pytest --cov=src              # Test suite + coverage
ruff check src/               # Lint
black --check src/            # Format check
```

Test markers:
- `not slow` — quick unit & integration tests (default CI)
- `slow` — validation suite + benchmark suite (marked `@pytest.mark.slow`)

---

## What-If, Fault Injection, Root Cause & Maintenance Options

```python
from src.simulation.what_if import ScenarioSimulator, ScenarioAdjustment
from src.faults.injection import FaultInjector, FaultSpec, FaultType
from src.explainability.root_cause import analyze_scenario
from src.maintenance.decision_engine import MaintenanceDecisionEngine

# What-if: raise fuel flow, drop compressor efficiency, compare before/after.
comparison = ScenarioSimulator().run(
    baseline_observation,
    ScenarioAdjustment(fuel_flow_kg_s=1.8, compressor_efficiency=0.65),
)

# Fault injection: propagate compressor fouling + a sensor bias through the twin.
twin.fault_injector = FaultInjector([
    FaultSpec(FaultType.COMPRESSOR_FOULING, severity=0.4),
    FaultSpec(FaultType.SENSOR_BIAS, severity=0.3, target_sensor="T3"),
])
result = twin.update(observation)

# Root cause: rank what drove the health delta.
report = analyze_scenario(baseline_inputs, adjusted_inputs, comparison.delta["overall_health"])

# Maintenance options: ranked menu, not just one recommendation.
options = MaintenanceDecisionEngine().generate_options(
    health=result["OverallHealth"],
    rul_cycles=result["RULCycles"],
    failure_probability=result["FailureProbability"],
)
```

All exposed via API endpoints and the Streamlit dashboard's **What-If Simulator**, **Fault Injection**, **Root Cause Analysis**, and **Maintenance Options** pages.

---

## SHAP Model Explainability

```python
from src.explainability.shap_explainer import explain_prediction, feature_interaction_matrix

# Prepare features (surrogate model handles preprocessing internally)
raw = frame[surrogate_model.feature_names].iloc[:5]
prepped = surrogate_model._prepare(raw)

def predict_fn(x):
    return surrogate_model.pipeline.predict(x)

# Global + local explanations
explanation = explain_prediction(predict_fn, prepped,
    feature_names=surrogate_model.pipeline_feature_names)
print(explanation["global_importance"])   # ranked feature list
print(explanation["local_explanations"])  # top-10 SHAP values per row

# Interaction matrix
interaction = feature_interaction_matrix(predict_fn, prepped,
    feature_names=surrogate_model.pipeline_feature_names)
```

Also available via `POST /v1/explain` and the dashboard **Model Explainability** page.

---

## Uncertainty Quantification

Three modes configurable on `SurrogateModel`:

| Mode | Method | Coverage Guarantee | Speed |
|---|---|---|---|
| `conformal` | Split-conformal (calibration residuals) | Marginal | Fast |
| `quantile` | Quantile regression (α/2, 1-α/2) | Conditional | Medium |
| `ensemble` | Bootstrapped with noise injection | Approximate | Slowest |

```python
model.uncertainty_mode = "quantile"
point, lower, upper, confidence = model.predict_with_uncertainty(frame)
```

---

## Hybrid Physics + ML

The `HybridPhysicsMLModel` trains an ML model on the *residual* (actual − physics prediction), so the combined prediction is `physics + ml_residual`. This is a powerful digital twin technique:

- Physics handles condition-dependent variation
- ML only needs to model the degradation signal (simpler)
- Residual magnitude itself is a diagnostic (model mismatch → novel degradation)

```python
from src.surrogate.hybrid import HybridPhysicsMLModel

model = HybridPhysicsMLModel.train(frame, ml_kind="hist_gradient_boosting")
pred = model.predict(test_frame)
point, lower, upper, conf = model.predict_with_uncertainty(test_frame)
model.save("models/hybrid.joblib")
```

---

## Performance Benchmarks

Run `pipeline benchmark` to measure every model variant:

| Model | Mean Latency (ms) | Throughput (ops/s) | Memory (MB) | Model Size (MB) |
|---|---|---|---|---|
| HistGradientBoosting | ~0.3 | ~150,000 | ~15 | ~2 |
| ExtraTrees | ~1.2 | ~80,000 | ~25 | ~40 |
| RandomForest | ~1.5 | ~60,000 | ~30 | ~50 |
| Hybrid Physics+ML | ~2.0 | ~45,000 | ~20 | ~3 |
| Stacking (cv=5) | ~8.0 | ~12,000 | ~60 | ~45 |

---

## Dataset Contract

One row = one engine cycle. SI units throughout. Health values dimensionless `[0, 1]`. Full contract in [`docs/DATA.md`](docs/DATA.md).

| Group | Fields |
|---|---|
| Identity | `EngineID`, `Cycle` |
| Flight condition | `Altitude`, `Mach`, `Tamb`, `Pamb` |
| Operating point | `RPM`, `FuelFlow` |
| Station measurements | `P2`, `T2`, `P3`, `T3`, `P4`, `T4` |
| Training-only targets | `CompressorHealth`, `CombustorHealth`, `TurbineHealth`, `OverallHealth`, `Thrust`, `TSFC` |

The feature engineering step (`src/dataset/features.py`) adds 20 derived features including physics-normalized residuals (`ResP2`–`ResT4`) and thermodynamic ratios. The surrogate pipeline uses **34 total features** (8 raw + 6 physics residuals + 10 engineered ratios/deltas).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Mermaid diagram shows "Unable to render rich display" | Unquoted `&` in node labels treated as parallel-edge operator | Wrap labels containing `&` in double quotes |
| Streamlit metrics show white boxes with no text | Missing text color on `.stMetric` elements | Add `color` to `.stMetric` CSS |
| Dashboard page loads slowly on every navigation | Full inference pipeline re-runs on each script execution | Results now cached with `@st.cache_data` |
| Trade-Off Analysis crashes with "unsupported format string" | Duplicate column names when y-axis equals "Thrust" | Column list deduplicated with `dict.fromkeys()` |
| Parameter Sweep shows "not in index" | Input parameter not included in sweep result dicts | Parameter value now stored in each result entry |

---

## License

Released under the [MIT License](LICENSE).
