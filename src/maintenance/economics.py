"""Maintenance cost and avoided-loss estimates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceEconomics:
    planned_cost: float
    expected_unplanned_cost: float
    expected_savings: float
    downtime_hours: float


def estimate_economics(
    failure_probability: float,
    planned_cost: float = 50_000,
    failure_cost: float = 500_000,
    downtime_hours: float = 24,
) -> MaintenanceEconomics:
    """Calculate risk-weighted savings of planned intervention."""
    unplanned = failure_probability * failure_cost
    return MaintenanceEconomics(
        planned_cost, unplanned, max(unplanned - planned_cost, 0), downtime_hours
    )
