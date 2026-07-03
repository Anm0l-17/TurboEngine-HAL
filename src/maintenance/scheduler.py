"""Fleet maintenance scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleItem:
    engine_id: str
    due_cycle: int
    priority: float


def schedule(items: list[ScheduleItem], capacity: int | None = None) -> list[ScheduleItem]:
    """Order work by risk priority then due cycle, optionally limiting capacity."""
    ordered = sorted(items, key=lambda item: (-item.priority, item.due_cycle))
    return ordered if capacity is None else ordered[:capacity]
