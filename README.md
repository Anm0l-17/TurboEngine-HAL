# Turbojet Digital Twin

Production-oriented physics-informed digital twin for real-time four-stage turbojet health
monitoring. It combines Brayton-cycle constraints, Bayesian state estimation, supervised
surrogates, uncertainty calibration, RUL prediction, fleet analytics, and condition-based
maintenance.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,api,dashboard,reports]"
python pipeline.py demo
pytest
uvicorn src.api.server:app --reload
streamlit run src/viz/dashboard.py
```

Train with `python pipeline.py train --data path/to/data.csv`, then infer with
`python pipeline.py predict --data path/to/data.csv`. Artifacts are written below `models/`
and `results/`. The expected CSV schema is documented in `docs/DATA.md`.

## Engineering scope

- Physically constrained Brayton-cycle reconstruction with conservation residuals
- EKF, UKF, and sequential Monte Carlo state estimators
- Multi-output health/performance surrogate selection and conformal intervals
- Data-driven degradation trajectories, RUL quantiles, and failure probabilities
- Stateful real-time and batch API, fleet ranking, drift monitoring, and maintenance economics
- Streamlit dashboard, Markdown/HTML reports, model export, CLI, tests, and containers

All stochastic workflows use configured seeds. Optional accelerators degrade cleanly when their
libraries are not installed.
