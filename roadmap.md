# AeroPulse Twin — Project Roadmap

## Challenge

**Physics-Informed Digital Twin for Real-Time Four-Stage Turbojet Health Monitoring**

Build a virtual representation of a single-spool turbojet which uses limited measurements to estimate:

- Compressor health
- Combustor health
- Turbine health
- Overall engine health
- Thrust, fuel-efficiency metrics, and degradation trajectory
- Confidence/uncertainty in predictions

The supplied dataset is synthetic but physics-based. The immediate goal is a convincing Round-1 presentation; the final goal is an interactive virtual digital-twin dashboard.

---

## Plain-language engine model

```text
Ambient air → compressor → combustor → turbine → nozzle/exhaust
                 ↑              │          │
                 └──── single rotating shaft ────┘
```

1. **Compressor:** squeezes air; pressure and temperature rise.
2. **Combustor:** burns fuel in compressed air; temperature rises sharply, with a small pressure loss.
3. **Turbine:** hot gas expands through turbine blades; it powers the compressor through the shared shaft.
4. **Nozzle:** remaining energy accelerates exhaust backward, creating forward thrust.

The dataset stations are approximately:

| Station | Measurements | Meaning |
|---|---|---|
| Ambient/inlet | `Tamb`, `Pamb`, altitude, Mach | Flight environment |
| Compressor exit | `P2`, `T2` | Compression response |
| Combustor exit / turbine inlet | `P3`, `T3` | Combustion response |
| Turbine exit | `P4`, `T4` | Turbine/exhaust response |
| Engine control | RPM, fuel flow | Operating demand |

---

## Central idea

At the same altitude, Mach, RPM, and fuel flow, a healthy engine behaves predictably. Persistent difference between the expected healthy response and measured response indicates degradation.

```text
Flight + engine inputs
          ↓
Healthy-engine surrogate predicts expected P2–T4
          ↓
Compare expected and actual sensor values (residuals)
          ↓
Infer compressor / combustor / turbine health
          ↓
Overall health, thrust estimate, confidence, alert
```

This distinguishes normal changes caused by altitude/weather from true engine deterioration.

---

## Core physics features

Use these as interpretable ML features and physics-consistency checks.

```text
Compressor pressure ratio:       CPR = P2 / Pamb
Combustor pressure retention:    BPR = P3 / P2
Turbine pressure ratio:          TPR = P4 / P3

Compressor temperature rise:     ΔTc = T2 − Tamb
Combustor temperature rise:      ΔTb = T3 − T2
Turbine temperature drop:        ΔTt = T3 − T4

Flight velocity:                 V0 = Mach × √(γ R Tamb)
```

Expected physical behaviour:

- Compressor: pressure and temperature increase.
- Combustor: temperature increases; pressure reduces slightly.
- Turbine: pressure and temperature reduce.
- In a single-spool engine, turbine shaft power approximately meets compressor shaft-power demand.

### Thrust

General jet thrust equation:

```text
F = ṁe Ve − ṁa V0 + (pe − p0) Ae
```

For an ideally expanded turbojet nozzle, `pe ≈ p0`, so the pressure-area term is small.

Specific thrust:

```text
Fs = F / ṁa = (1 + f) Ve − V0
f = fuel mass flow / air mass flow
```

The dataset does not list nozzle area or air mass flow. Therefore thrust must be presented as a **physics-constrained surrogate estimate**, unless the supplied dataset includes a true thrust label.

---

## Solution architecture

### 1. Data and atmospheric normalization

Inputs:

```text
Altitude, Mach, Tamb, Pamb, RPM, Fuel Flow
```

Outputs/measurements:

```text
P2, T2, P3, T3, P4, T4
```

First account for different altitude and Mach conditions. A lower pressure at high altitude is normal and must not be classified as degradation.

### 2. Healthy-engine surrogate

Train a fast tabular model to predict healthy expected sensor response:

```text
[altitude, Mach, Tamb, Pamb, RPM, fuel flow]
                         ↓
[expected P2, T2, P3, T3, P4, T4, thrust]
```

Initial model: Gradient-Boosted Trees (XGBoost, LightGBM, or CatBoost).

Why:

- Excellent baseline for tabular sensor data
- Fast inference
- Easier to train and explain than a deep neural network
- Supports feature importance and ensembles

### 3. Residual-based health estimation

Calculate deviations from expected healthy behaviour:

```text
eP2 = P2_actual − P2_expected
eT4 = T4_actual − T4_expected
```

Infer component scores from residuals, ratios, and trend history.

| Subsystem | Key evidence |
|---|---|
| Compressor | `P2/Pamb`, `T2−Tamb`, RPM, P2/T2 residual trend |
| Combustor | `P3/P2`, `T3−T2`, fuel-flow-to-temperature relation |
| Turbine | `P4/P3`, `T3−T4`, turbine residuals, shaft-power mismatch |

Output normalized health scores from 0 to 100. If no true health labels exist, call them **relative health indicators**, not literal physical percentages.

### 4. Physics constraints

Use soft penalties or validation checks for impossible predictions:

- `P2 > Pamb` during normal compression
- `T3 > T2` during combustion
- `P3 < P2` because of combustor pressure loss
- `P4 < P3` and normally `T4 < T3` through turbine expansion
- Non-negative fuel flow and thrust
- Smooth health change over consecutive cycles

### 5. Overall health and alerts

Example:

```text
Overall HI = 0.60 × minimum(component scores)
           + 0.40 × weighted average(component scores)
```

This gives extra importance to the weakest critical component.

Suggested alerts:

```text
80–100: Healthy
60–79:  Monitor
40–59:  Inspect soon
0–39:   Critical
```

### 6. Uncertainty

Train five models with different random seeds. Use their prediction spread as a confidence band.

Example output:

```text
Turbine health: 74 ± 6
Predicted thrust: 12.3 ± 0.8 kN
```

---

## Data checklist — do this first

- Inspect all files, columns, units, and documentation.
- Count engine IDs and cycles per engine.
- Confirm whether health labels, thrust, TSFC, or failure labels exist.
- Confirm whether pressures and temperatures are static or total/stagnation values.
- Identify missing values and unrealistic readings.
- Plot trends over cycles for several individual engines.
- Split data by **Engine ID**, not random rows, to test generalization to unseen engines.

---

## Round-1 presentation plan

The template cannot be changed and extra slides cannot be added. Fit the content into the provided slide structure.

### Engineering rationale

- Direct sensor values alone cannot diagnose health because flight condition changes them.
- Hidden health states are inferred from pressure/temperature response after atmospheric normalization.
- Explain degradation mechanisms: compressor fouling, combustor efficiency loss, turbine erosion.

### Surrogate modelling strategy

- Healthy baseline: gradient-boosted, multi-output sensor surrogate.
- Inputs: environment + RPM + fuel flow.
- Outputs: expected station pressures/temperatures plus estimated performance.
- Physics filters reject/penalize impossible outputs.
- Fast enough for real-time inference.

### Health-estimation methodology

- Calculate derived pressure ratios and temperature changes.
- Compare observed readings against healthy-baseline prediction.
- Use residual history to estimate component health and degradation trend.
- Fuse subsystem scores into an overall health index.
- Show model confidence using an ensemble.

### Key results and insights

Include at least:

- Dataset overview: engines, cycles, operating range.
- One chart of sensor trends across cycles.
- One chart comparing expected healthy vs observed degraded response.
- One example health/degradation trajectory.
- Inference-time target and validation plan.
- **Synchronized 2D/3D Dashboard Showcase**:
  - 3D engine model visual representation with dynamic, RPM-scaled airflow streamlines.
  - Tightly-linked 2D diagnostics: any EKF health drop or injected fault instantly highlights in pulsing red on the 3D model.
  - Three-state color scheme: Green (healthy working $\ge$ 85%), Yellow (probable issues 60%-85%), Red (failed/faulty $<$ 60% or active fault).

### Strong one-line pitch

> Our physics-informed twin separates normal flight-condition variation from true degradation, explains the affected engine subsystem, and provides confidence-aware real-time health guidance.

---

## Final-round virtual demonstration

Build the dashboard in Streamlit after Round 1.

### Modes

1. **Dataset replay:** Select an engine ID and replay its sensor history cycle by cycle.
2. **Manual scenario (What-If):** Enter altitude, Mach, ambient pressure/temperature, RPM, and fuel flow. The 3D model reflects health/efficiency adjustments in real time.
3. **Route simulation:** Select Delhi → Bengaluru and run a representative mission profile.
4. **Fault Injection:** Inject sensor bias, drifts, or component efficiency loss. Active faults immediately illuminate the target component in pulsing bright red on the 3D model.

### Delhi → Bengaluru demo profile

| Flight phase | Representative condition |
|---|---|
| Take-off from Delhi | Low altitude; high thrust/fuel-flow demand |
| Climb | Increasing altitude and Mach; high RPM |
| Cruise | Stable high altitude and Mach |
| Descent | Reducing altitude and engine demand |
| Bengaluru approach/landing | Low altitude; changing ambient conditions |

Weather data should adjust ambient temperature and pressure. For high altitude, use the International Standard Atmosphere as the default and apply weather-derived corrections where supported.

Dashboard disclosure:

> This scenario combines open meteorological conditions with a representative flight profile. It is a virtual demonstration, not live aircraft telemetry.

### Dashboard panels

- **Integrated 2D-3D Panel**: Linked 2D control and telemetry indicators side-by-side with an interactive 3D WebGL engine model.
- **Airflow Streamline Visualization**: Animated flow lines representing air movement through the engine, with velocity dynamically proportional to RPM.
- **Three-State Color-Coded Mesh**: Components rendered in Green (operating normally), Yellow (potential degradation/warning), or Red (critical failure or active fault).
- Flight conditions: route, altitude, Mach, ambient pressure and temperature
- Engine conditions: RPM, fuel flow, P2–T4
- Health gauges: compressor, combustor, turbine, overall
- Trends: health vs time/cycle; sensor values vs time/cycle
- Performance: predicted thrust and TSFC
- Alert + explanation: why the twin raised the warning (instant pulsing red on the 3D model)
- Confidence band on each prediction

---

## Timeline

### Before Round-1 deadline

1. Obtain and inspect the official dataset.
2. Create derived physics features and exploratory charts.
3. Train a basic surrogate baseline.
4. Produce one meaningful health/degradation example.
5. Create a template-compliant PPT with a dashboard mock-up.

### If selected

1. Improve surrogate and test on unseen engines.
2. Add residual-based health estimator and uncertainty ensemble.
3. Build Streamlit dashboard and dataset replay.
4. Add manual scenario simulator.
5. Add Delhi → Bengaluru route/weather demonstration.
6. Test the full demonstration offline and prepare a short presentation script.

---

## Team roles (2–4 members)

| Role | Responsibilities |
|---|---|
| Data/ML | Dataset inspection, surrogate model, validation |
| Physics | Derived features, constraints, health-index logic |
| UI/PPT | Dashboard mock-up, Streamlit interface, presentation story |
| Integration/QA | Uncertainty, scenario testing, final demo |

---

## Public technical references

- NASA Glenn, [Engine Thrust Equations](https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/thsum508.html)
- NASA Glenn, [General Thrust Equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-force/)
- NASA Glenn, [Compressor Thermodynamics](https://www.grc.nasa.gov/www/k-12/airplane/compth.html)
- NASA Glenn, [Burner Thermodynamics](https://www.grc.nasa.gov/www/k-12/airplane/burnth.html)
- NASA Glenn, [Turbine Thermodynamics](https://www.grc.nasa.gov/WWW/K-12/BGP/powtrbth.html)
- NASA Glenn, [Mass Flow and Choking](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mass-flow-rate-equations/)
- NASA, [C-MAPSS degradation-data description](https://c3.ndc.nasa.gov/dashlink/resources/14/)
