from __future__ import annotations

from dataclasses import dataclass

from src.signal_controller import SignalController
from src.utils import clamp


LANES: tuple[str, ...] = ("north", "east", "south", "west")

ARRIVAL_PLAN: tuple[dict[str, int], ...] = (
    {"north": 4, "east": 2, "south": 3, "west": 2},
    {"north": 5, "east": 3, "south": 2, "west": 3},
    {"north": 3, "east": 3, "south": 7, "west": 2},
    {"north": 4, "east": 4, "south": 8, "west": 3},
    {"north": 3, "east": 5, "south": 7, "west": 4},
    {"north": 2, "east": 8, "south": 4, "west": 3},
    {"north": 3, "east": 9, "south": 3, "west": 4},
    {"north": 2, "east": 10, "south": 4, "west": 3},
    {"north": 3, "east": 8, "south": 3, "west": 5},
    {"north": 6, "east": 3, "south": 4, "west": 5},
    {"north": 7, "east": 3, "south": 3, "west": 4},
    {"north": 6, "east": 4, "south": 5, "west": 3},
    {"north": 3, "east": 5, "south": 8, "west": 3},
    {"north": 4, "east": 4, "south": 7, "west": 4},
    {"north": 3, "east": 4, "south": 5, "west": 7},
    {"north": 4, "east": 3, "south": 4, "west": 8},
)


@dataclass(frozen=True)
class DemoStep:
    step: int
    lane_counts: dict[str, int]
    ai_green_lane: str
    fixed_green_lane: str
    duration_seconds: int
    reason: str
    emergency_active: bool
    emergency_lane: str | None
    congestion_level: str
    ai_queue_total: int
    fixed_queue_total: int
    ai_wait_total: int
    fixed_wait_total: int

    @property
    def saved_wait(self) -> int:
        return max(0, self.fixed_wait_total - self.ai_wait_total)


def build_demo_steps() -> list[DemoStep]:
    controller = SignalController(min_green=12, max_green=50)
    ai_queues = {lane: 0 for lane in LANES}
    fixed_queues = {lane: 0 for lane in LANES}
    fixed_cycle = LANES
    ai_wait_total = 0
    fixed_wait_total = 0
    results: list[DemoStep] = []

    for index, arrivals in enumerate(ARRIVAL_PLAN, start=1):
        for lane in LANES:
            ai_queues[lane] += arrivals[lane]
            fixed_queues[lane] += arrivals[lane]

        emergency_lane = "east" if 7 <= index <= 9 else None
        decision = controller.choose_green_lane(ai_queues, emergency_lane=emergency_lane)
        ai_drained = clamp(decision.duration_seconds // 4, 3, 12)
        ai_queues[decision.green_lane] = max(0, ai_queues[decision.green_lane] - ai_drained)

        fixed_green_lane = fixed_cycle[(index - 1) % len(fixed_cycle)]
        fixed_queues[fixed_green_lane] = max(0, fixed_queues[fixed_green_lane] - 7)

        ai_queue_total = sum(ai_queues.values())
        fixed_queue_total = sum(fixed_queues.values())
        ai_wait_total += ai_queue_total
        fixed_wait_total += fixed_queue_total

        results.append(
            DemoStep(
                step=index,
                lane_counts=dict(ai_queues),
                ai_green_lane=decision.green_lane,
                fixed_green_lane=fixed_green_lane,
                duration_seconds=decision.duration_seconds,
                reason=decision.reason,
                emergency_active=emergency_lane is not None,
                emergency_lane=emergency_lane,
                congestion_level=_congestion_level(ai_queue_total),
                ai_queue_total=ai_queue_total,
                fixed_queue_total=fixed_queue_total,
                ai_wait_total=ai_wait_total,
                fixed_wait_total=fixed_wait_total,
            )
        )

    return results


def _congestion_level(total_queue: int) -> str:
    if total_queue >= 31:
        return "Severe"
    if total_queue >= 19:
        return "High"
    if total_queue >= 9:
        return "Moderate"
    return "Low"
