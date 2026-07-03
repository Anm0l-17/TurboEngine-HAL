"""Engine schematic state visualization."""

import plotly.graph_objects as go


def engine_schematic(health: dict[str, float]) -> go.Figure:
    """Render component health as a horizontal engine flow path."""
    names = ["Compressor", "Combustor", "Turbine"]
    values = [health.get(f"{name}Health", 0) for name in names]
    colors = [f"rgb({int(255 * (1-v))},{int(180 * v)},60)" for v in values]
    return go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
