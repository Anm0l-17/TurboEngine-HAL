"""Professional Plotly figures for dashboard, reports, and analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def health_gauge(value: float, title: str = "Overall Health") -> go.Figure:
    """Create a bounded health gauge with threat-level coloring."""
    color = "#15803d" if value >= 0.7 else ("#ca8a04" if value >= 0.3 else "#b91c1c")
    return go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            title={"text": title, "font": {"size": 16}},
            delta={"reference": 50, "increasing": {"color": "#15803d"}, "decreasing": {"color": "#b91c1c"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#fecaca"},
                    {"range": [30, 70], "color": "#fef08a"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 30,
                },
            },
        )
    )


def trend(frame: pd.DataFrame, columns: list[str], title: str = "") -> go.Figure:
    """Plot selected variables against cycle with confidence bands when available."""
    fig = go.Figure()
    for col in columns:
        if col not in frame.columns:
            continue
        lower = f"{col}Lower"
        upper = f"{col}Upper"
        if lower in frame.columns and upper in frame.columns:
            x = frame["Cycle"]
            fig.add_trace(go.Scatter(
                x=x, y=frame[upper], mode="lines",
                line={"width": 0}, showlegend=False, name=f"{col}_upper"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=frame[lower], mode="lines",
                line={"width": 0}, fill="tonexty", fillcolor="rgba(99, 102, 241, 0.15)",
                showlegend=False, name=f"{col}_lower"
            ))
        fig.add_trace(go.Scatter(
            x=frame["Cycle"], y=frame[col], mode="lines+markers",
            name=col, line={"width": 2}, marker={"size": 4},
        ))
    fig.update_layout(
        title=title or None,
        xaxis_title="Cycle",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 40, "r": 20, "t": 40, "b": 60},
    )
    return fig


def health_trajectory_plot(frame: pd.DataFrame) -> go.Figure:
    """Multi-panel health trajectory with per-component breakdown."""
    health_cols = ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
    colors = ["#6366f1", "#06b6d4", "#f59e0b", "#10b981"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=health_cols, vertical_spacing=0.12)
    for i, (col, color) in enumerate(zip(health_cols, colors)):
        row, col_pos = (i // 2) + 1, (i % 2) + 1
        fig.add_trace(
            go.Scatter(x=frame["Cycle"], y=frame[col], mode="lines+markers",
                       name=col, line={"color": color, "width": 2}, marker={"size": 3}),
            row=row, col=col_pos,
        )
        fig.update_xaxes(title_text="Cycle", row=row, col=col_pos)
        fig.update_yaxes(title_text="Health", range=[0, 1], row=row, col=col_pos)
    fig.update_layout(
        template="plotly_white", height=500,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


def pareto_frontier(health: np.ndarray, tsfc: np.ndarray, thrust: np.ndarray) -> go.Figure:
    """Scatter plot of health vs TSFC with thrust as size, showing Pareto frontier."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=health, y=tsfc, mode="markers",
        marker={"size": np.clip(thrust / 5000, 5, 30), "color": thrust,
                "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "Thrust (N)"}},
        text=[f"Thrust: {t:.0f} N<br>Health: {h:.2f}<br>TSFC: {s:.5f}" for t, h, s in zip(thrust, health, tsfc)],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Health vs TSFC (bubble = thrust)",
        xaxis_title="Overall Health", yaxis_title="TSFC (kg/N·s)",
        template="plotly_white", height=500,
    )
    return fig


def calibration_plot(
    predicted: np.ndarray, lower: np.ndarray, upper: np.ndarray, actual: np.ndarray,
    target_name: str = "",
) -> go.Figure:
    """Calibration plot: predicted vs actual with prediction intervals."""
    order = np.argsort(predicted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(predicted)), y=upper[order], mode="lines",
        line={"width": 0}, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(predicted)), y=lower[order], mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor="rgba(99, 102, 241, 0.2)",
        name="90% Prediction Interval",
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(predicted)), y=predicted[order], mode="lines+markers",
        name="Predicted", line={"color": "#6366f1", "width": 2}, marker={"size": 3},
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(predicted)), y=actual[order], mode="markers",
        name="Actual", marker={"color": "#ef4444", "size": 4, "symbol": "x"},
    ))
    fig.update_layout(
        title=f"Calibration: {target_name}" if target_name else "Calibration Plot",
        xaxis_title="Sample (sorted by prediction)", yaxis_title="Value",
        template="plotly_white", height=400,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def correlation_heatmap(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Correlation heatmap of selected columns."""
    corr = frame[columns].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white", height=500,
        xaxis={"tickangle": -45},
    )
    return fig


def engine_schematic(health: dict[str, float]) -> go.Figure:
    """Render component health as a horizontal engine flow path with color bar."""
    names = ["Compressor", "Combustor", "Turbine", "Nozzle"]
    values = [health.get(f"{name}Health", 0) if name != "Nozzle" else (
        health.get("TurbineHealth", 0) * 0.95
    ) for name in names]
    colors = [f"rgb({int(255 * (1 - v))},{int(180 * v + 50)},60)" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        title="Engine Component Health",
        xaxis=dict(range=[0, 1], title="Health", tickformat="%"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white", height=250,
        margin={"l": 100, "r": 20, "t": 30, "b": 30},
    )
    return fig
