from __future__ import annotations


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
