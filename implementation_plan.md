# Implementation Plan: Physics-Informed Digital Twin (AeroPulse Twin)

This implementation plan details the coding tasks required to build the Physics-Informed Digital Twin for Real-Time Four-Stage Turbojet Health Monitoring.

## Goal Description
Build a virtual representation of a single-spool four-stage turbojet engine. The twin uses flight conditions and engine controls to estimate expected healthy sensor outputs, compares them to actual readings to extract residuals, and feeds these residuals into an Unscented Kalman Filter (UKF) to track hidden component health states (Compressor, Combustor, Turbine). It then predicts performance parameters (Thrust, TSFC) and displays them in an interactive Streamlit dashboard with a live physics-vs-hybrid comparison graph and automated aviation maintenance action items.

---

## Proposed Tasks & Milestones

### Phase 1: Data Exploration & Preprocessing
- [x] Inspect the official CSV datasets to identify columns and units (datasets placed under `data/`).
- [x] Implement environmental normalization factors ($\theta$, $\delta$) to correct readings (implemented in `src/physics/thermodynamics.py`).
- [x] Calculate derived physical features like CPR, BPR, TPR, and temperature deltas (implemented in `src/dataset/features.py`).

### Phase 2: Healthy-Baseline Surrogate Modeling
- [x] Filter dataset to isolate healthy engine cycles (implemented in `src/dataset/split.py`).
- [x] Train multi-output regression models (implemented in `src/surrogate/model.py` and `src/surrogate/train.py`).
- [x] Evaluate surrogate performance on validation sets (implemented in `src/training/evaluate.py`).

### Phase 3: State Tracking & Health Estimation
- [x] Define hidden health states and transition equations (defined in `src/estimation/ukf.py` and `src/health/overall.py`).
- [x] Implement Unscented Kalman Filter (UKF) / state estimator (implemented in `src/estimation/ukf.py`).
- [x] Map residuals to health indices and track overall health over cycles.
- [x] Perform Remaining Useful Life (RUL) estimation (implemented in `src/prediction/rul.py`).

### Phase 4: Streamlit Dashboard Construction
- [x] Build an interactive Streamlit operations dashboard (implemented in `src/viz/dashboard.py` with 2D charts and a 3D WebGL engine viewer).
- [x] Add dynamic, RPM-scaled airflow streamlines through the engine components in the 3D model.
- [x] Implement three-state health color rendering: Green (Healthy $\ge$ 85%), Yellow (Warning 60%–85%), and Red (Failed/Critical $<$ 60% or Active Fault).
- [x] Integrate 2D diagnostics with the 3D reflection, ensuring UKF warnings, What-If efficiency adjustments, and injected faults immediately illuminate the target component in pulsing bright red.

### Phase 5: Physics vs. Hybrid Live Comparison Graph
- [ ] Add a dual-curve overlay chart to the dashboard that plots the **pure thermodynamic Brayton-cycle estimation** alongside the **final hybrid (physics + ML residual) prediction** for each sensor target ($P_2, T_2, P_3, T_3, P_4, T_4$) over cycles.
- [ ] Shade the gap between the two curves to visually quantify the **ML-isolated degradation residual** — the exact mechanical wear the machine learning model is capturing beyond what the physics baseline can explain.
- [ ] Include a dynamic annotation or tooltip showing the residual magnitude ($\Delta = \hat{Y}_{hybrid} - \hat{Y}_{physics}$) at the current cycle, proving to judges that the AI is not a black box but a transparent physics-augmented system.
- [ ] Ensure the chart updates in real time during Flight Replay, What-If, and Fault Injection modes.

### Phase 6: Automated Aviation Maintenance Action Items
- [ ] When any component health state crosses into **Yellow** ($60\% \le HI < 85\%$), automatically generate and display a **Warning-Level Maintenance Action Item** panel on the dashboard. Example actions:
    - *Compressor Yellow*: "Schedule borescope inspection within 50 cycles. Monitor CPR trend for further decay. Consider compressor wash if fouling pattern detected."
    - *Combustor Yellow*: "Inspect fuel nozzle spray pattern and combustor liner at next scheduled downtime. Verify HRR trend stability."
    - *Turbine Yellow*: "Schedule thermal paint / borescope of turbine blading within 30 cycles. Monitor TIT exceedances."
- [ ] When any component crosses into **Red** ($HI < 60\%$ or active fault), escalate to a **Critical-Level Maintenance Action Item** with urgency flags:
    - *Compressor Red*: "GROUND ENGINE. Compressor section failed safety threshold. Mandatory borescope and possible blade replacement before return to service."
    - *Combustor Red*: "GROUND ENGINE. Combustor efficiency critically degraded. Full hot-section inspection required. Check for liner cracking or fuel nozzle failure."
    - *Turbine Red*: "GROUND ENGINE. Turbine erosion/blade damage exceeds safe limits. Remove from service for turbine overhaul. Do NOT operate."
- [ ] Log each generated action item with a timestamp, affected component, health value, triggering residual pattern, and recommended action into a **Maintenance Decision Log** table in the dashboard.
- [ ] Provide a one-click "Export Maintenance Report" button that generates a downloadable PDF/Markdown summary of all active and historical action items for the selected engine.

---

## Verification Plan

### Automated Verification
- Run unit tests to verify standard-day correction factor functions.
- Run tests on the surrogate model to ensure it achieves $< 1\%$ validation error on healthy test cases.
- Run simulation loops to check that Kalman filter state estimates stay within $[0.0, 1.0]$ limits.
- Run dashboard unit tests to ensure that the 3D engine mesh loader and viewer template render HTML payloads correctly.
- Verify the physics-vs-hybrid comparison chart renders both curves and the shaded residual region for all 6 sensor targets.
- Verify that Yellow and Red health thresholds trigger the correct maintenance action item text and severity level.

### Manual Verification
- Launch the Streamlit dashboard locally and select the **3D Engine** view mode in **Overview**, **Engine Health**, **Degradation Analysis**, **What-If Simulator**, and **Fault Injection** pages.
- Verify that airflow streamline speeds dynamically increase as throttle RPM increases.
- Change the Compressor efficiency slider under **What-If Simulator** to 0.70 (Yellow) and 0.40 (Red) and check that the 3D compressor section changes color correspondingly.
- Inject a turbine erosion fault under **Fault Injection** and confirm that the turbine section on the 3D model immediately flashes in pulsing red.
- Confirm the **Physics vs. Hybrid** chart displays two distinct curves with the shaded ML-residual gap, and that the residual annotation updates as the cycle advances.
- Confirm that lowering compressor health to Yellow triggers a Warning-level action item ("Schedule borescope inspection…") and lowering it further to Red triggers a Critical-level action item ("GROUND ENGINE…").
- Verify the Maintenance Decision Log table accumulates entries with correct timestamps, health values, and recommended actions.
- Test the "Export Maintenance Report" button and verify the downloaded file contains all logged action items.
