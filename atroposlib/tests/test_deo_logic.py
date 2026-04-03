import pytest
import time
from atroposlib.orchestration.controller import ScalingController
from atroposlib.orchestration.metrics import WorkloadMetrics

@pytest.fixture
def controller():
    return ScalingController(
        min_actors=1,
        max_actors=10,
        target_pressure=1.0,
        scaling_threshold=0.2,
        cooldown_seconds=0,  # Disable cooldown for most tests
        max_step_change=4
    )

def mock_metrics(pressure: float, timestamp: float = None):
    return WorkloadMetrics(
        current_step=100,
        queue_size=int(pressure * 10),
        total_rollouts=1000,
        unallocated_fraction=0.0,
        num_envs=1,
        batch_size=10,
        timestamp=timestamp or time.time()
    )

def test_initial_scale_up(controller):
    # Pressure 5.0 -> Should scale up
    metrics = mock_metrics(5.0)
    # 1 * 5.0 = 5. Target 5.
    target = controller.calculate_desired(metrics, current_actors=1)
    assert target == 5

def test_step_limiting(controller):
    # Pressure 20.0 -> Target would be 20.
    # Max step is 4. Current is 1. Target should be 5.
    metrics = mock_metrics(20.0)
    target = controller.calculate_desired(metrics, current_actors=1)
    assert target == 5

def test_hysteresis(controller):
    # Target is 1.0. Threshold is 0.2.
    # Pressure 1.1 -> No action (within 1.0 ± 0.2)
    metrics = mock_metrics(1.1)
    target = controller.calculate_desired(metrics, current_actors=5)
    assert target == 5
    
    # Pressure 1.3 -> Action (outside threshold)
    metrics = mock_metrics(1.3)
    # 5 * 1.3 = 6.5 -> ceil = 7
    target = controller.calculate_desired(metrics, current_actors=5)
    assert target == 7

def test_cooldown():
    c = ScalingController(cooldown_seconds=60)
    now = time.time()
    
    # First action sets the timestamp
    metrics = mock_metrics(5.0, timestamp=now)
    c.calculate_desired(metrics, current_actors=1)
    
    # Second action 10s later -> Should hold due to cooldown
    metrics_later = mock_metrics(5.0, timestamp=now + 10)
    target = c.calculate_desired(metrics_later, current_actors=1)
    assert target == 1

def test_pending_actors_compensation(controller):
    # Pressure 5.0. Current 1. Target 5.
    # But if we have 3 pending, we only need to add 1 more.
    # Wait, the controller returns the *Total* target count it thinks we should have.
    # The CLI then decides whether to call set_instance_count.
    
    # If raw_target is 5, but connected=1 and pending=4.
    # Effective = 5. The controller should see that we ALREADY have 5 "in flight".
    metrics = mock_metrics(5.0)
    target = controller.calculate_desired(metrics, current_actors=1, pending_actors=4)
    # Should stay at 1 (because effective is 5)
    assert target == 1
    
    # If pending is only 2. Effective is 3. Target is 5.
    # It should scale up to the target 5.
    target = controller.calculate_desired(metrics, current_actors=1, pending_actors=2)
    assert target == 5

def test_world_bounds(controller):
    # Max is 10. Pressure 100.0.
    metrics = mock_metrics(100.0)
    target = controller.calculate_desired(metrics, current_actors=1)
    # Limited by max_step first (1+4=5)
    assert target == 5
    
    # If current is 9. Pressure 100.0.
    target = controller.calculate_desired(metrics, current_actors=9)
    assert target == 10 # Limited by max_actors
