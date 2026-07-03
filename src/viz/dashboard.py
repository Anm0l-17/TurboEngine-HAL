"""Professional Streamlit operations dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st
from src.dataset.loader import _COLUMN_ALIASES
from src.digital_twin.engine import DigitalTwin
from src.digital_twin.fleet import rank_fleet
from src.viz.engine_animation import engine_schematic
from src.viz.plots import health_gauge, trend

st.set_page_config(page_title="Turbojet Digital Twin", page_icon="✈", layout="wide")
st.title("Four-Stage Turbojet Digital Twin")
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Engine Health",
        "Performance",
        "RUL",
        "Maintenance",
        "Fleet",
        "Model Explainability",
        "Upload & Inference",
        "Settings",
    ],
)
uploaded = st.sidebar.file_uploader("Sensor dataset", type="csv")
if uploaded is None:
    st.info("Upload an official-schema CSV to begin inference.")
else:
    try:
        data = pd.read_csv(uploaded)
        data = data.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in data.columns})
        model_path = st.sidebar.text_input("Model artifact", "models/best_model.joblib")

        outputs = []
        for engine_id, group in data.groupby("EngineID", sort=False):
            twin = DigitalTwin(str(engine_id))
            if Path(model_path).exists():
                twin.load_model(model_path)
            result = twin.batch_predict(group)
            result["EngineID"] = engine_id
            result["Cycle"] = group["Cycle"].reset_index(drop=True)
            outputs.append(result)
        output = pd.concat(outputs, ignore_index=True)

        latest = output.iloc[-1]
        latest_per_engine = output.sort_values("Cycle").groupby("EngineID", as_index=False).tail(1)

        if page == "Overview":
            a, b, c, d = st.columns(4)
            a.metric("Health", f"{latest['OverallHealth']:.1%}")
            b.metric("Thrust", f"{latest['Thrust']:.0f} N")
            c.metric("RUL", f"{latest['RULCycles']:.0f} cycles")
            d.metric("Risk", str(latest["RiskLevel"]).upper())
            left, right = st.columns(2)
            with left:
                st.plotly_chart(health_gauge(float(latest["OverallHealth"])), width="stretch")
            with right:
                schematic_health = {
                    "CompressorHealth": float(latest["CompressorHealth"]),
                    "CombustorHealth": float(latest["CombustorHealth"]),
                    "TurbineHealth": float(latest["TurbineHealth"]),
                }
                st.plotly_chart(engine_schematic(schematic_health), width="stretch")

        elif page in {"Engine Health", "Performance", "RUL"}:
            columns = (
                ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
                if page == "Engine Health"
                else (
                    ["Thrust", "TSFC"]
                    if page == "Performance"
                    else ["RULCycles", "FailureProbability"]
                )
            )
            single_engine = output[output["EngineID"] == latest["EngineID"]].reset_index(drop=True)
            st.plotly_chart(trend(single_engine, columns), width="stretch")

        elif page == "Maintenance":
            st.subheader(str(latest["Maintenance"]))
            st.write(f"Risk level: {latest['RiskLevel']}")

        elif page == "Fleet":
            fleet_input = latest_per_engine.rename(columns={"EngineID": "engine_id"})
            ranked = rank_fleet(fleet_input)
            st.dataframe(ranked, width="stretch")

        elif page == "Model Explainability":
            twin_for_importance = DigitalTwin()
            if Path(model_path).exists():
                twin_for_importance.load_model(model_path)
            model = twin_for_importance.model
            if model is None:
                st.warning("No model loaded — cannot compute feature importances.")
            else:
                pipeline = model.pipeline
                estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
                if hasattr(estimator, "feature_importances_"):
                    importances = pd.DataFrame(
                        {
                            "Feature": model.feature_names,
                            "Importance": estimator.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)
                    st.bar_chart(importances.set_index("Feature"))
                else:
                    st.info("Loaded model does not expose feature importances.")

        else:
            st.dataframe(output, width="stretch")

        st.download_button("Export predictions", output.to_csv(index=False), "predictions.csv")
    except (ValueError, KeyError) as error:
        st.error(str(error))
