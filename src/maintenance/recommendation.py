"""Explainable maintenance recommendation rules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceDecision:
    """Action, urgency, risk, and operational allowance."""

    action: str
    urgency: str
    risk_level: str
    safe_cycles: int
    rationale: str


def recommend(
    health: float, rul_cycles: float, failure_probability: float, weakest_subsystem: str = "engine"
) -> MaintenanceDecision:
    """Map risk indicators to a conservative actionable decision."""
    safe = max(0, int(min(rul_cycles * 0.8, rul_cycles - 5)))
    if health < 0.3 or failure_probability >= 0.7 or rul_cycles <= 10:
        return MaintenanceDecision(
            f"Remove from service; overhaul {weakest_subsystem}",
            "immediate",
            "critical",
            0,
            "Failure threshold reached or imminent",
        )
    if health < 0.55 or failure_probability >= 0.35 or rul_cycles <= 50:
        return MaintenanceDecision(
            f"Schedule inspection of {weakest_subsystem}",
            "within 5 cycles",
            "high",
            safe,
            "Degradation exceeds the intervention band",
        )
    if health < 0.75 or failure_probability >= 0.1:
        return MaintenanceDecision(
            "Increase monitoring frequency",
            "next service window",
            "moderate",
            safe,
            "Early degradation trend detected",
        )
    return MaintenanceDecision(
        "Continue normal operation",
        "routine",
        "low",
        safe,
        "Health and predicted life remain within limits",
    )
