import unittest
from unittest.mock import MagicMock, patch

from atroposlib.orchestration.controller import ScalingController
from atroposlib.orchestration.metrics import WorkloadMetrics


class TestScalingLogic(unittest.TestCase):
    def setUp(self):
        self.controller = ScalingController(
            min_actors=1,
            max_actors=10,
            target_pressure=1.0,
            scaling_threshold=0.2,  # ±0.2
            cooldown_seconds=60,
        )

    def test_hysteresis_no_action(self):
        """Verify no scaling action if pressure is within threshold."""
        metrics = WorkloadMetrics(0, 10, 0, 0.0, 1, 10, 1000.0)  # Pressure = 1.0
        # Pressure is 1.0 (target is 1.0), should stay at 1
        target = self.controller.calculate_desired(metrics, current_actors=1)
        self.assertEqual(target, 1)

        # Pressure is 1.15 (within 0.2 threshold)
        metrics.queue_size = 11.5
        target = self.controller.calculate_desired(metrics, current_actors=1)
        self.assertEqual(target, 1)

    def test_scale_up_with_pending(self):
        """Verify we don't over-provision if pending actors already satisfy the target."""
        metrics = WorkloadMetrics(0, 40, 0, 0.0, 1, 10, 1000.0)  # Pressure = 4.0
        # Target should be 4.
        # But we already have 1 connected + 3 pending = 4 total effective.
        target = self.controller.calculate_desired(
            metrics, current_actors=1, pending_actors=3
        )
        self.assertEqual(target, 1)  # Should not request more

    def test_cooldown_enforcement(self):
        """Verify that scaling actions are blocked during the cooldown period."""
        metrics = WorkloadMetrics(0, 50, 0, 0.0, 1, 10, 1000.0)  # Pressure = 5.0

        # 1. First action
        target = self.controller.calculate_desired(metrics, current_actors=1)
        self.assertEqual(target, 5)

        # 2. Immediate second action (should be blocked by cooldown)
        metrics.timestamp += 10  # Only 10s passed
        metrics.queue_size = 100  # Pressure = 10.0
        target = self.controller.calculate_desired(metrics, current_actors=5)
        self.assertEqual(target, 5)  # Still 5

        # 3. After cooldown
        metrics.timestamp += 60  # 70s passed total
        target = self.controller.calculate_desired(metrics, current_actors=5)
        # Expected is 9 because max_step_change=4 (5 + 4 = 9)
        self.assertEqual(target, 9)

    def test_drain_aware_scale_down(self):
        """Verify we don't scale down more if we are already draining enough actors."""
        metrics = WorkloadMetrics(0, 1, 0, 0.0, 5, 10, 1000.0)  # Pressure = 0.1
        # Target for pressure 0.1 and 5 actors would be 1.
        # But if we are already draining 4 actors (5 - 4 = 1), we don't need a new action.
        target = self.controller.calculate_desired(
            metrics, current_actors=5, draining_actors=4
        )
        self.assertEqual(target, 5)  # No new action, already at target effective count


if __name__ == "__main__":
    unittest.main()
