"""Multi-engine fleet analytics."""

import pandas as pd


def rank_fleet(latest_states: pd.DataFrame) -> pd.DataFrame:
    """Rank engines from highest to lowest operational risk."""
    required = {"engine_id", "OverallHealth", "FailureProbability", "RULCycles"}
    if not required.issubset(latest_states):
        raise ValueError(f"missing fleet fields: {sorted(required - set(latest_states))}")
    out = latest_states.copy()
    out["RiskScore"] = (
        (1 - out["OverallHealth"]) * 0.45
        + out["FailureProbability"] * 0.4
        + 0.15 / (1 + out["RULCycles"])
    )
    return out.sort_values("RiskScore", ascending=False).reset_index(drop=True)
