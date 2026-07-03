"""Reusable Plotly figures."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def health_gauge(value: float, title: str = "Overall Health") -> go.Figure:
    """Create a bounded health gauge."""
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "#b91c1c"},
                    {"range": [30, 70], "color": "#ca8a04"},
                    {"range": [70, 100], "color": "#15803d"},
                ],
            },
        )
    )


def trend(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Plot selected variables against cycle."""
    return px.line(frame, x="Cycle", y=columns, markers=True)
