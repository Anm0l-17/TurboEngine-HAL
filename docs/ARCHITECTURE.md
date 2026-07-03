# Architecture

Sensor records pass through schema/range validation into two paths: a constrained Brayton-cycle
model and a learned multi-output surrogate. A Bayesian estimator fuses health observations over
time. The resulting trajectory drives RUL, failure-risk, and maintenance decisions. The
`DigitalTwin` facade is shared by CLI, FastAPI, dashboard, and fleet workflows. Artifacts are
versionable joblib models and JSON/CSV reports; online state is JSON-safe.
