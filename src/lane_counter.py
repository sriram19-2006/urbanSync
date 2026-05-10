from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LaneCounter:
    """Tracks vehicle counts by lane."""

    lane_names: tuple[str, ...] = ("north", "east", "south", "west")
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = {lane: 0 for lane in self.lane_names}

    def update(self, lane: str, count: int) -> None:
        self._validate_lane(lane)
        self.counts[lane] = max(0, count)

    def add(self, lane: str, amount: int = 1) -> None:
        self._validate_lane(lane)
        self.counts[lane] = max(0, self.counts[lane] + amount)

    def drain(self, lane: str, amount: int = 1) -> None:
        self._validate_lane(lane)
        self.counts[lane] = max(0, self.counts[lane] - amount)

    def busiest_lane(self) -> str:
        return max(self.counts, key=self.counts.get)

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def _validate_lane(self, lane: str) -> None:
        if lane not in self.counts:
            raise ValueError(f"Unknown lane: {lane}")
