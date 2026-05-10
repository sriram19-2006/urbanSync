from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalDecision:
    green_lane: str
    duration_seconds: int
    reason: str


class SignalController:
    """Chooses which lane should receive a green signal."""

    def __init__(self, min_green: int = 10, max_green: int = 60) -> None:
        self.min_green = min_green
        self.max_green = max_green

    def choose_green_lane(
        self,
        lane_counts: dict[str, int],
        emergency_lane: str | None = None,
    ) -> SignalDecision:
        if emergency_lane is not None:
            return SignalDecision(
                green_lane=emergency_lane,
                duration_seconds=self.max_green,
                reason="emergency vehicle priority",
            )

        if not lane_counts:
            raise ValueError("lane_counts cannot be empty")

        green_lane = max(lane_counts, key=lane_counts.get)
        duration = self._duration_for_count(lane_counts[green_lane])
        return SignalDecision(
            green_lane=green_lane,
            duration_seconds=duration,
            reason="highest vehicle count",
        )

    def _duration_for_count(self, count: int) -> int:
        scaled = self.min_green + count * 3
        return min(self.max_green, max(self.min_green, scaled))
