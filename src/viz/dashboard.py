"""Professional Streamlit operations dashboard with advanced analytics."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from src.dataset.loader import _COLUMN_ALIASES
from src.digital_twin.engine import DigitalTwin
from src.digital_twin.fleet import rank_fleet
from src.explainability.root_cause import analyze_scenario
from src.faults.injection import FaultInjector, FaultSpec, FaultType
from src.maintenance.decision_engine import MaintenanceDecisionEngine
from src.simulation.what_if import ScenarioAdjustment, ScenarioSimulator
from src.viz.engine_animation import engine_schematic
from src.viz.plots import (
    calibration_plot, correlation_heatmap, health_gauge,
    health_trajectory_plot, pareto_frontier, trend,
)

st.set_page_config(page_title="Turbojet Digital Twin", page_icon="✈", layout="wide")
st.markdown("""
<style>
    .main .block-container { padding: 1rem 1.5rem; }
    .stMetric { background: #f8fafc; padding: 0.75rem 0.5rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; min-height: 5rem; }
    .stMetric * { color: #1e293b !important; }
    .stMetric label { color: #475569 !important; font-weight: 600; font-size: 0.75rem !important; white-space: normal !important; word-wrap: break-word !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.1rem !important; white-space: normal !important; word-wrap: break-word !important; }
    .st-bb { border-bottom: 2px solid #6366f1; }
    h1 { color: #1e293b; }
    h2, h3 { color: #334155; }
    .stButton button { background: #6366f1; color: white; border-radius: 0.375rem; }
    .stSelectbox label, .stSlider label { font-weight: 600; color: #475569; }
    div[data-testid="column"] { overflow: visible; }
</style>
""", unsafe_allow_html=True)

st.title("✈ Four-Stage Turbojet Digital Twin")

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Engine Health",
        "Performance",
        "RUL & Risk",
        "Trade-Off Analysis",
        "Parameter Sweep",
        "Calibration Analysis",
        "Degradation Analysis",
        "Correlation Analysis",
        "Fleet Comparison",
        "Model Explainability",
        "What-If Simulator",
        "Fault Injection",
        "Root Cause Analysis",
        "Maintenance",
        "Maintenance Options",
        "Upload & Inference",
        "Settings",
    ],
)

uploaded = st.sidebar.file_uploader("Sensor dataset (CSV)", type="csv")
model_path = st.sidebar.text_input("Model artifact", "models/best_model.joblib")
estimator_method = st.sidebar.selectbox("State Estimator", ["ekf", "ukf"], index=0)

if uploaded is None:
    st.info("Upload an official-schema CSV to begin inference.")
    st.stop()

@st.cache_data(show_spinner="Processing sensor data and running inference...")
def run_inference(file_bytes: bytes, model_path: str, estimator_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import io
    data = pd.read_csv(io.BytesIO(file_bytes))
    data = data.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in data.columns})
    outputs = []
    for engine_id, group in data.groupby("EngineID", sort=False):
        group = group.sort_values("Cycle").reset_index(drop=True)
        twin = DigitalTwin(str(engine_id), estimator_method=estimator_method)
        if Path(model_path).exists():
            twin.load_model(model_path)
        result = twin.batch_predict(group)
        result["EngineID"] = engine_id
        outputs.append(result)
    output = pd.concat(outputs, ignore_index=True)
    output = output.loc[:, ~output.columns.duplicated()]
    return data, output

try:
    file_bytes = uploaded.getvalue()
    data, output = run_inference(file_bytes, model_path, estimator_method)
    show_debug = st.sidebar.checkbox("Show raw data", False)
    if show_debug:
        st.subheader("Raw Input")
        st.dataframe(data.head(10), width="stretch")

    latest = output.iloc[-1]
    latest_per_engine = output.sort_values("Cycle").groupby("EngineID", as_index=False).tail(1)
    latest_engine_id = int(latest["EngineID"])
    latest_input = data[data["EngineID"] == latest_engine_id].sort_values("Cycle").iloc[-1]
    latest_engine_output = output[output["EngineID"] == latest_engine_id].reset_index(drop=True)

    if page == "Overview":
        kpi_row = st.columns(6)
        kpi_row[0].metric("Overall Health", f"{latest['OverallHealth']:.1%}",
                          help="Fused health indicator (0=failed, 1=pristine)")
        kpi_row[1].metric("Thrust", f"{latest['Thrust']:.0f} N",
                          help="Predicted engine thrust")
        kpi_row[2].metric("RUL", f"{latest['RULCycles']:.0f} cycles",
                          help="Remaining useful life estimate")
        kpi_row[3].metric("Confidence", f"{latest['Confidence']:.0%}",
                          help="Conformal prediction coverage")
        kpi_row[4].metric("Risk", str(latest.get("RiskLevel", "N/A")).upper(),
                          help="Maintenance risk level")
        kpi_row[5].metric("Failure Prob.", f"{latest['FailureProbability']:.1%}",
                          help="Probability of failure within horizon")

        op_row = st.columns(4)
        op_row[0].metric("Altitude", f"{latest_input.get('Altitude', 0):.0f} m")
        op_row[1].metric("Mach", f"{latest_input.get('Mach', 0):.2f}")
        op_row[2].metric("RPM", f"{latest_input.get('RPM', 0):.0f}")
        op_row[3].metric("Fuel flow", f"{latest_input.get('FuelFlow', 0):.3f} kg/s")

        h_row = st.columns(4)
        h_row[0].metric("Compressor", f"{latest['CompressorHealth']:.1%}")
        h_row[1].metric("Combustor", f"{latest['CombustorHealth']:.1%}")
        h_row[2].metric("Turbine", f"{latest['TurbineHealth']:.1%}")
        h_row[3].metric("Degradation rate", f"{latest['DegradationRate']:.5f}/cycle")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(health_gauge(float(latest["OverallHealth"])), width='stretch')
        with right:
            schematic_health = {
                "CompressorHealth": float(latest["CompressorHealth"]),
                "CombustorHealth": float(latest["CombustorHealth"]),
                "TurbineHealth": float(latest["TurbineHealth"]),
            }
            st.plotly_chart(engine_schematic(schematic_health), width='stretch')

        cols = st.multiselect(
            "Trend lines", ["OverallHealth", "Confidence", "RULCycles", "FailureProbability"],
            default=["OverallHealth", "Confidence"]
        )
        if cols:
            st.plotly_chart(trend(latest_engine_output, cols, "Health & Confidence Trend"),
                            width='stretch')

    elif page in {"Engine Health", "Performance", "RUL & Risk"}:
        col_map = {
            "Engine Health": ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"],
            "Performance": ["Thrust", "TSFC"],
            "RUL & Risk": ["RULCycles", "FailureProbability", "DegradationRate", "Confidence"],
        }
        columns = col_map[page]
        st.plotly_chart(health_trajectory_plot(latest_engine_output) if page == "Engine Health"
                        else trend(latest_engine_output, columns, page),
                        width='stretch')
        st.dataframe(latest_engine_output[["Cycle"] + columns].tail(10), width="stretch")

    elif page == "Trade-Off Analysis":
        st.subheader("Design Trade-Off Analysis")
        health_cols = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
        perf_cols = ["Thrust", "TSFC", "DegradationRate", "FailureProbability"]
        x_axis = st.selectbox("X-axis", health_cols + perf_cols, index=3)
        y_axis = st.selectbox("Y-axis", health_cols + perf_cols, index=4)
        cols = list(dict.fromkeys([x_axis, y_axis, "Thrust"]))
        valid = output[cols].dropna()
        if len(valid) > 1:
            st.plotly_chart(
                pareto_frontier(
                    valid[x_axis].values, valid[y_axis].values, valid["Thrust"].values
                ), width='stretch'
            )
        st.subheader("Per-Engine Trade Space")
        for eid in sorted(output["EngineID"].unique()):
            eng = output[output["EngineID"] == eid]
            st.plotly_chart(
                trend(eng, [x_axis, y_axis], f"Engine {eid}"),
                width='stretch',
            )

    elif page == "Parameter Sweep":
        st.subheader("Parameter Sweep")
        st.markdown("Vary one input parameter and observe the effect on key outputs.")
        col1, col2 = st.columns(2)
        param = col1.selectbox("Input parameter", ["FuelFlow", "RPM", "Altitude", "Mach"])
        n_points = col2.slider("Number of steps", 3, 15, 6)
        twin_sweep = DigitalTwin(str(latest_engine_id), estimator_method=estimator_method)
        if Path(model_path).exists():
            twin_sweep.load_model(model_path)
        base_row = latest_input.to_dict()
        vals = {
            "FuelFlow": [max(0.1, base_row.get("FuelFlow", 1.0) * (0.5 + 0.2 * i)) for i in range(n_points)],
            "RPM": [min(120_000, max(20_000, base_row.get("RPM", 80_000) * (0.6 + 0.15 * i))) for i in range(n_points)],
            "Altitude": [max(0, min(15_000, base_row.get("Altitude", 5_000) * (0.0 + 0.4 * i))) for i in range(n_points)],
            "Mach": [max(0.05, min(2.0, base_row.get("Mach", 0.5) * (0.4 + 0.2 * i))) for i in range(n_points)],
        }
        sweep_results = []
        for v in vals[param]:
            obs = base_row.copy()
            obs[param] = v
            try:
                result = twin_sweep.update(obs)
                result[param] = v
                sweep_results.append(result)
            except (ValueError, KeyError):
                continue
        if sweep_results:
            df_sweep = pd.DataFrame(sweep_results)
            out_cols = st.multiselect(
                "Outputs", ["OverallHealth", "Thrust", "TSFC", "RULCycles", "FailureProbability",
                            "CompressorHealth", "TurbineHealth"],
                default=["OverallHealth", "Thrust"]
            )
            if out_cols:
                plot_df = df_sweep[[param] + out_cols].set_index(param)
                st.line_chart(plot_df)

    elif page == "Calibration Analysis":
        st.subheader("Conformal Prediction Calibration")
        cal_twin = DigitalTwin(str(latest_engine_id), estimator_method=estimator_method)
        if Path(model_path).exists() and len(data) > 0:
            cal_twin.load_model(model_path)
        if cal_twin.model is not None:
            from src.dataset.loader import TARGETS as TGT
            cal_engine_data = data[data["EngineID"] == latest_engine_id].sort_values("Cycle")
            pred, lower, upper, coverage = cal_twin.model.predict_with_uncertainty(cal_engine_data)
            actual = output[output["EngineID"] == latest_engine_id].sort_values("Cycle").reset_index(drop=True)
            target = st.selectbox("Target", TGT, index=3)
            if target in pred.columns and target in actual.columns:
                st.plotly_chart(
                    calibration_plot(
                        pred[target].values, lower[target].values, upper[target].values,
                        actual[target].values, target,
                    ), width='stretch'
                )
                st.info(f"Nominal coverage: {coverage:.1%}")
                in_interval = ((actual[target].values >= lower[target].values) &
                               (actual[target].values <= upper[target].values))
                st.metric("Empirical Coverage", f"{in_interval.mean():.1%}")
        else:
            st.warning("No surrogate model loaded.")

    elif page == "Degradation Analysis":
        st.subheader("Degradation Trajectory Analysis")
        engine_choice = st.selectbox("Engine", sorted(output["EngineID"].unique()))
        eng_out = output[output["EngineID"] == engine_choice].sort_values("Cycle").reset_index(drop=True)
        st.plotly_chart(health_trajectory_plot(eng_out), width='stretch')
        st.metric("Initial Health", f"{eng_out['OverallHealth'].iloc[0]:.1%}")
        st.metric("Final Health", f"{eng_out['OverallHealth'].iloc[-1]:.1%}")
        total_drop = eng_out['OverallHealth'].iloc[0] - eng_out['OverallHealth'].iloc[-1]
        st.metric("Total Degradation", f"{total_drop:.1%}")
        st.metric("Degradation Rate", f"{eng_out['DegradationRate'].iloc[-1]:.5f}/cycle")

    elif page == "Correlation Analysis":
        st.subheader("Feature Correlation Analysis")
        n_features = st.slider("Features to include", 5, 20, 12)
        corr_cols = data.select_dtypes(include=[np.number]).columns[:n_features].tolist()
        st.plotly_chart(correlation_heatmap(data, corr_cols), width='stretch')

    elif page == "Fleet Comparison":
        st.subheader("Fleet-Wide Comparison")
        fleet_input = latest_per_engine.drop(columns=["engine_id"], errors="ignore").rename(
            columns={"EngineID": "engine_id"}
        )
        ranked = rank_fleet(fleet_input)
        st.dataframe(ranked.style
                     .highlight_min(["OverallHealth"], color="#fecaca")
                     .highlight_max(["RiskScore"], color="#fecaca"),
                     width="stretch")

    elif page == "Model Explainability":
        st.subheader("Model Explainability (SHAP / Permutation Importance)")
        twin_exp = DigitalTwin(estimator_method=estimator_method)
        if Path(model_path).exists():
            twin_exp.load_model(model_path)
        model = twin_exp.model
        if model is None:
            st.warning("No model loaded.")
        else:
            from src.explainability.shap_explainer import explain_prediction, feature_interaction_matrix
            try:
                feature_names = getattr(model, "pipeline_feature_names", model.feature_names)
                raw_background = output[model.feature_names].dropna().iloc[:200] if len(output) > 200 else output[model.feature_names].dropna()
                raw_sample = output[model.feature_names].dropna().iloc[:5]
                background = model._prepare(raw_background)
                sample = model._prepare(raw_sample)

                def predict_fn(x: pd.DataFrame) -> np.ndarray:
                    return np.asarray(model.pipeline.predict(x))

                with st.spinner("Computing SHAP explanations..."):
                    explanation = explain_prediction(predict_fn, sample, feature_names, background)
                st.success(f"Method: {explanation['method']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Global Feature Importance**")
                    imp_df = pd.DataFrame(explanation["global_importance"])
                    if not imp_df.empty:
                        imp_df = imp_df.sort_values("importance", ascending=True)
                        st.bar_chart(imp_df.set_index("feature"), height=400)
                with col2:
                    st.markdown("**Local Explanations (Row 1)**")
                    if explanation["local_explanations"]:
                        local_df = pd.DataFrame(explanation["local_explanations"][0]["factors"])
                        if not local_df.empty:
                            local_df = local_df.sort_values("shap_value", ascending=True)
                            st.bar_chart(local_df.set_index("feature"), height=400)

                st.markdown("**Local Explanation Details**")
                for local in explanation["local_explanations"]:
                    with st.expander(f"Row {local['row']}"):
                        ldf = pd.DataFrame(local["factors"])
                        st.dataframe(ldf, width="stretch")

                with st.spinner("Computing interaction matrix..."):
                    interaction = feature_interaction_matrix(predict_fn, background, feature_names, max_features=8)
                if interaction.get("matrix"):
                    st.markdown("**Feature Interaction Matrix (SHAP)**")
                    matrix = np.array(interaction["matrix"])
                    names = interaction["names"]
                    fig = {
                        "data": [
                            {
                                "z": matrix.tolist(),
                                "x": names,
                                "y": names,
                                "type": "heatmap",
                                "colorscale": "Viridis",
                                "hoverongaps": False,
                            }
                        ],
                        "layout": {
                            "title": "Mean |SHAP interaction|",
                            "width": 600,
                            "height": 600,
                        },
                    }
                    st.plotly_chart(fig, width='content')

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
                try:
                    imp_df = pd.DataFrame(explanation.get("global_importance", []))
                    if not imp_df.empty:
                        st.bar_chart(imp_df.set_index("feature"), height=400)
                except Exception:
                    pass

    elif page == "What-If Simulator":
        st.subheader("Scenario Simulator")
        baseline_row = latest_input.to_dict()
        c1, c2, c3 = st.columns(3)
        fuel_flow = c1.slider("Fuel flow (kg/s)", 0.0, 5.0,
                              float(baseline_row.get("FuelFlow", 1.0)), 0.01)
        rpm = c2.slider("RPM", 10_000.0, 120_000.0,
                        float(baseline_row.get("RPM", 80_000.0)))
        tamb = c3.slider("Ambient temp (K)", 200.0, 350.0,
                         float(baseline_row.get("Tamb", 288.0)))
        c4, c5, c6 = st.columns(3)
        pamb = c4.slider("Ambient press (Pa)", 40_000.0, 105_000.0,
                         float(baseline_row.get("Pamb", 101_325.0)))
        compressor_eff = c5.slider("Compressor efficiency", 0.3, 1.0, 1.0)
        turbine_eff = c6.slider("Turbine efficiency", 0.3, 1.0, 1.0)
        noise = st.slider("Sensor noise std", 0.0, 0.2, 0.0, 0.01)

        adjustment = ScenarioAdjustment(
            fuel_flow_kg_s=fuel_flow, rpm=rpm,
            ambient_temperature_k=tamb, ambient_pressure_pa=pamb,
            compressor_efficiency=compressor_eff, turbine_efficiency=turbine_eff,
            sensor_noise_std=noise,
        )
        try:
            comparison = ScenarioSimulator().run(baseline_row, adjustment)
            st.session_state["scenario_comparison"] = comparison
            st.session_state["scenario_inputs"] = (baseline_row, adjustment)
            before, after = st.columns(2)
            for label, snapshot, col in (
                ("Before", comparison.baseline, before),
                ("After", comparison.adjusted, after),
            ):
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("Health", f"{snapshot.overall_health:.1%}")
                    st.metric("RUL", f"{snapshot.remaining_useful_life_cycles:.0f} cycles")
                    st.metric("Failure prob.", f"{snapshot.failure_probability:.1%}")
                    st.metric("Thrust", f"{snapshot.thrust_n:.0f} N")
                    st.metric("TSFC", f"{snapshot.tsfc_kg_n_s:.5f} kg/N·s")
                    st.metric("Confidence", f"{snapshot.confidence:.1%}")
            st.json(comparison.delta)
        except ValueError as error:
            st.error(str(error))

    elif page == "Fault Injection":
        st.subheader("Fault Injection Controls")
        selected_faults = st.multiselect("Active faults", [ft.value for ft in FaultType], default=[])
        specs: list[FaultSpec] = []
        for fault_name in selected_faults:
            fault_type = FaultType(fault_name)
            severity = st.slider(f"{fault_name} severity", 0.0, 1.0, 0.5, key=f"sev_{fault_name}")
            target = None
            if fault_type in {FaultType.SENSOR_DRIFT, FaultType.SENSOR_BIAS}:
                target = st.selectbox(
                    f"{fault_name} target", ["T2", "T3", "T4", "P2", "P3", "P4"],
                    key=f"target_{fault_name}",
                )
            specs.append(FaultSpec(fault_type, severity, target))
        injector = FaultInjector(specs)
        twin_fault = DigitalTwin(str(latest_engine_id), estimator_method=estimator_method)
        if Path(model_path).exists():
            twin_fault.load_model(model_path)
        twin_fault.fault_injector = injector
        faulted = twin_fault.batch_predict(
            data[data["EngineID"] == latest_engine_id].sort_values("Cycle")
        )
        st.dataframe(faulted[["Cycle", "CompressorHealth", "CombustorHealth", "TurbineHealth",
                              "OverallHealth", "Thrust", "RULCycles"]], width="stretch")

    elif page == "Root Cause Analysis":
        st.subheader("Root Cause Analysis")
        comparison = st.session_state.get("scenario_comparison")
        if comparison is None:
            st.info("Run the What-If Simulator first.")
        else:
            baseline_row, adjustment = st.session_state["scenario_inputs"]
            baseline_inputs = {
                "FuelFlow": baseline_row.get("FuelFlow", 0.0),
                "RPM": baseline_row.get("RPM", 0.0),
                "Tamb": baseline_row.get("Tamb", 0.0),
                "Pamb": baseline_row.get("Pamb", 0.0),
                "compressor_efficiency": 1.0, "turbine_efficiency": 1.0,
            }
            adjusted_inputs = {
                "FuelFlow": adjustment.fuel_flow_kg_s,
                "RPM": adjustment.rpm,
                "Tamb": adjustment.ambient_temperature_k,
                "Pamb": adjustment.ambient_pressure_pa,
                "compressor_efficiency": adjustment.compressor_efficiency or 1.0,
                "turbine_efficiency": adjustment.turbine_efficiency or 1.0,
            }
            report = analyze_scenario(
                baseline_inputs, adjusted_inputs, comparison.delta["overall_health"]
            )
            st.info(report.summary)
            for factor in report.factors:
                st.write(f"- **{factor.factor}** ({factor.contribution:+.3f}): {factor.explanation}")
            st.markdown("**Causal chain:**")
            st.write(" → ".join(report.causal_chain))

    elif page == "Maintenance":
        st.subheader("Maintenance Recommendation")
        st.metric("Action", str(latest.get("Maintenance", "N/A")),
                  delta=latest.get("RiskLevel", ""))
        st.progress(float(latest["OverallHealth"]), text=f"Health: {latest['OverallHealth']:.1%}")
        st.progress(min(float(latest["FailureProbability"]), 1.0),
                    text=f"Failure Risk: {latest['FailureProbability']:.1%}")
        st.metric("RUL", f"{latest['RULCycles']:.0f} cycles")
        st.metric("Degradation Rate", f"{latest['DegradationRate']:.5f}/cycle")

    elif page == "Maintenance Options":
        st.subheader("Ranked Maintenance Options")
        options = MaintenanceDecisionEngine().generate_options(
            float(latest["OverallHealth"]), float(latest["RULCycles"]),
            float(latest["FailureProbability"]),
        )
        options_df = pd.DataFrame([vars(o) for o in options])
        st.dataframe(options_df.style.highlight_max(["Utility"], color="#bbf7d0"),
                     width="stretch")
        st.success(f"Top recommendation: **{options[0].action}**")
        st.caption(options[0].rationale)

    else:  # Upload & Inference / Settings
        st.dataframe(output, width="stretch")

    st.sidebar.download_button("Export CSV", output.to_csv(index=False),
                                f"twin_predictions_{latest_engine_id}.csv")

except (ValueError, KeyError, Exception) as error:
    st.error(f"Error: {error}")
    import traceback
    st.code(traceback.format_exc())
