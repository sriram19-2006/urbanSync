from __future__ import annotations

import unittest

from src.demo import build_demo_steps
from src.signal_controller import SignalController


class DemoBehaviorTests(unittest.TestCase):
    def test_emergency_gets_green_lane(self) -> None:
        emergency_steps = [step for step in build_demo_steps() if step.emergency_active]

        self.assertGreater(len(emergency_steps), 0)
        self.assertTrue(all(step.ai_green_lane == "east" for step in emergency_steps))
        self.assertTrue(all(step.reason == "emergency vehicle priority" for step in emergency_steps))

    def test_ai_beats_fixed_timer_by_final_step(self) -> None:
        final_step = build_demo_steps()[-1]

        self.assertLess(final_step.ai_wait_total, final_step.fixed_wait_total)
        self.assertGreater(final_step.saved_wait, 0)

    def test_signal_controller_rejects_empty_counts(self) -> None:
        controller = SignalController()

        with self.assertRaises(ValueError):
            controller.choose_green_lane({})


if __name__ == "__main__":
    unittest.main()
