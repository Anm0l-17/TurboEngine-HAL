"""Execution timing utility."""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Timer:
    """Context manager recording elapsed wall-clock seconds."""

    start: float = field(default=0.0, init=False)
    elapsed: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = perf_counter() - self.start
