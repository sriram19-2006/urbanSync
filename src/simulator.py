from __future__ import annotations

from random import Random

from src.emergency import EmergencyManager
from src.lane_counter import LaneCounter
from src.signal_controller import SignalController
from src.utils import clamp


class TrafficSimulator:
    """Runs a deterministic traffic signal simulation."""

    def __init__(self, seed: int = 7) -> None:
        self.random = Random(seed)
        self.lanes = LaneCounter()
        self.signals = SignalController()
        self.emergency = EmergencyManager()

    def run(self, steps: int = 20) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []

        for step in range(1, steps + 1):
            self._simulate_arrivals()

            if step == 6:
                self.emergency.trigger("east", active_for_steps=2)

            decision = self.signals.choose_green_lane(
                self.lanes.snapshot(),
                emergency_lane=self.emergency.current_lane(),
            )
            self.lanes.drain(decision.green_lane, amount=clamp(decision.duration_seconds // 10, 1, 6))

            results.append(
                {
                    "step": step,
                    "green_lane": decision.green_lane,
                    "duration_seconds": decision.duration_seconds,
                    "reason": decision.reason,
                    "lane_counts": self.lanes.snapshot(),
                    "emergency_active": self.emergency.is_active(),
                }
            )

            self.emergency.tick()

        return results

    def _simulate_arrivals(self) -> None:
        for lane in self.lanes.lane_names:
            self.lanes.add(lane, self.random.randint(0, 4))
