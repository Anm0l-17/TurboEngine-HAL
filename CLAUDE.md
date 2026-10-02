# TurboEngine-HAL: Assistant Instructions & Repository Guide

## Project Summary
TurboEngine-HAL is a physics-informed digital twin for single-spool, four-stage turbojet engines. It couples Brayton thermodynamic cycle simulation with tree-based machine learning surrogates (ExtraTrees, HGBT, Stacking) in a hybrid residual architecture. Bayesian Kalman Filters (EKF/UKF) track health degradation states, conformal prediction quantifies uncertainty with 90% validity, and PyVista/VTK renders interactive 3D CAD engine meshes inside Streamlit and web dashboards.

## Key Environment Commands
- **Python Version**: Python 3.10+
- **Environment Setup**:
  ```bash
  pip install -e ".[all]"
  # Or with explicit groups including 3D visualization:
  pip install -e ".[dev,api,dashboard,reports,ml,viz3d]"
  ```
- **Run Full Pipeline**: `python pipeline.py run-all`
- **Run Tests**: `pytest tests/ -v`
- **Run FastAPI**: `python pipeline.py serve-api --host 0.0.0.0 --port 8000`
- **Run Streamlit Dashboard**: `python pipeline.py serve-dashboard --port 8501`

## Critical Architectural Guidelines
1. **Thermodynamic Consistency**: Always ensure thermodynamic stations follow standard SAE ARP 755A notation: Station 0 (ambient), Station 1 (intake), Station 2 (compressor face), Station 3 (compressor discharge), Station 4 (combustor exit / turbine inlet), Station 5 (turbine exit), Station 9 (exhaust).
2. **Residual Modeling**: ML models must learn the *residual* between sensor measurements and ideal thermodynamic Brayton cycle predictions: $\hat{y} = y_{physics} + f_{ML}(x)$. Never replace physics baselines with unconstrained black boxes.
3. **State Estimation**: Health state $h \in [0, 1]$ monotonically decreases with cycle count; degradation velocity $\dot{h}$ must remain non-negative.
4. **Streamlit & PyVista**: The Streamlit dashboard imports `pyvista` at startup. Always ensure `pyvista` is installed when launching the dashboard or testing viz3d modules.
5. **Testing**: Write unit tests in `tests/` matching module hierarchy. Preserve test coverage above 80%.

## Documentation Reference
- `llms.txt`: Standard LLM navigation file.
- `llms-full.txt`: Consolidated context of mathematical formulation and architecture.
- `docs/PROJECT_GUIDE.md`: Deep physical explanation of all equations in plain English.
- `docs/ARCHITECTURE.md`: High-level component interactions and design decisions.
