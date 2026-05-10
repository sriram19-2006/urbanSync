from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmergencyVehicle:
    lane: str
    active_for_steps: int = 3


class EmergencyManager:
    """Tracks temporary emergency vehicle priority."""

    def __init__(self) -> None:
        self._vehicle: EmergencyVehicle | None = None

    def trigger(self, lane: str, active_for_steps: int = 3) -> None:
        self._vehicle = EmergencyVehicle(lane=lane, active_for_steps=active_for_steps)

    def current_lane(self) -> str | None:
        if self._vehicle is None:
            return None
        return self._vehicle.lane

    def tick(self) -> None:
        if self._vehicle is None:
            return

        self._vehicle.active_for_steps -= 1
        if self._vehicle.active_for_steps <= 0:
            self._vehicle = None

    def is_active(self) -> bool:
        return self._vehicle is not None
