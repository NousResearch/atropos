"""Tests for the ZMQ sidecar aggregation and leader routing."""

import time
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import zmq


class TestZMQLogReceiver:
    """Tests for the ZMQLogReceiver class."""

    def test_receiver_init_and_bind(self):
        """Test that receiver initializes and binds to port."""
        from atroposlib.utils.logging_client import ZMQLogReceiver

        context = zmq.Context()
        receiver = ZMQLogReceiver(port=5700, context=context)
        assert receiver.port == 5700
        receiver.close()
        context.term()

    def test_receiver_recv_nowait_empty(self):
        """Test that recv_nowait returns None when no data."""
        from atroposlib.utils.logging_client import ZMQLogReceiver

        context = zmq.Context()
        receiver = ZMQLogReceiver(port=5701, context=context)
        result = receiver.recv_nowait()
        assert result is None
        receiver.close()
        context.term()

    def test_receiver_recv_data(self):
        """Test that receiver can receive data from a PUSH socket."""
        from atroposlib.utils.logging_client import ZMQLogReceiver

        context = zmq.Context()
        receiver = ZMQLogReceiver(port=5702, context=context)

        sender = context.socket(zmq.PUSH)
        sender.connect("tcp://localhost:5702")
        time.sleep(0.1)

        test_data = {"metric": 1.0, "_step": 10}
        sender.send_pyobj(test_data)
        time.sleep(0.1)

        result = receiver.recv_nowait()
        assert result == test_data

        sender.close()
        receiver.close()
        context.term()


class TestZMQLogAggregator:
    """Tests for the ZMQLogAggregator class."""

    def test_aggregator_env_registration(self):
        """Test environment registration."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5710)
        aggregator._handle_control_message(
            {
                "_type": "env_register",
                "env_type": "math",
                "instance": "math_0",
                "is_leader": True,
                "leader_receive_port": 5800,
            }
        )
        assert "math_0" in aggregator.registered_envs["math"]
        assert "math" in aggregator.leaders

    def test_aggregator_env_disconnect(self):
        """Test environment disconnection."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5711)
        aggregator.registered_envs["math"].add("math_0")
        aggregator.registered_envs["math"].add("math_1")

        aggregator._handle_control_message(
            {
                "_type": "env_disconnect",
                "env_type": "math",
                "instance": "math_0",
                "was_leader": False,
            }
        )
        assert "math_0" not in aggregator.registered_envs["math"]
        assert "math_1" in aggregator.registered_envs["math"]

    def test_aggregator_all_reported(self):
        """Test _all_reported logic."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5712)
        aggregator.registered_envs["math"] = {"math_0", "math_1", "math_2"}

        key = (10, "math")
        aggregator.env_reported[key] = {"math_0", "math_1"}
        assert not aggregator._all_reported(key)

        aggregator.env_reported[key].add("math_2")
        assert aggregator._all_reported(key)

    def test_aggregator_handle_log_payload(self):
        """Test handling log payloads."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5713)
        aggregator.registered_envs["math"] = {"math_0", "math_1"}

        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "math",
                "_instance": "math_0",
                "accuracy": 0.8,
            }
        )

        key = (10, "math")
        assert key in aggregator.pending_metrics
        assert "math_0" in aggregator.env_reported[key]

    def test_aggregator_aggregation(self):
        """Test metric aggregation."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5714)
        aggregator.registered_envs["math"] = {"math_0", "math_1"}

        # Create a mock leader socket
        mock_socket = MagicMock()
        aggregator.leaders["math"] = {"port": 5800, "socket": mock_socket}

        # Send logs from both instances
        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "math",
                "_instance": "math_0",
                "accuracy": 0.8,
            }
        )
        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "math",
                "_instance": "math_1",
                "accuracy": 0.9,
            }
        )

        # Verify aggregation was triggered (socket.send_pyobj was called)
        mock_socket.send_pyobj.assert_called_once()
        call_args = mock_socket.send_pyobj.call_args
        sent_data = call_args[0][0]

        assert sent_data["_step"] == 10
        assert "math/instances/math_0/accuracy" in sent_data
        assert "math/instances/math_1/accuracy" in sent_data
        assert sent_data["math/aggregated/accuracy_mean"] == pytest.approx(0.85)

    def test_aggregator_timeout(self):
        """Test timeout handling for slow instances."""
        from atroposlib.api.sidecar import AGGREGATION_TIMEOUT, ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5715)
        aggregator.registered_envs["math"] = {"math_0", "math_1"}

        # Create a mock leader socket
        mock_socket = MagicMock()
        aggregator.leaders["math"] = {"port": 5800, "socket": mock_socket}

        # Only one instance reports
        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "math",
                "_instance": "math_0",
                "accuracy": 0.8,
            }
        )

        key = (10, "math")
        aggregator.pending_timestamps[key] = time.time() - AGGREGATION_TIMEOUT - 1

        aggregator._check_timeouts()

        # Verify partial data was sent
        mock_socket.send_pyobj.assert_called_once()

    def test_aggregator_multiple_env_types(self):
        """Test aggregation with multiple environment types."""
        from atroposlib.api.sidecar import ZMQLogAggregator

        aggregator = ZMQLogAggregator(port=5716)
        aggregator.registered_envs["math"] = {"math_0"}
        aggregator.registered_envs["crossword"] = {"crossword_0"}

        mock_math_socket = MagicMock()
        mock_crossword_socket = MagicMock()
        aggregator.leaders["math"] = {"port": 5800, "socket": mock_math_socket}
        aggregator.leaders["crossword"] = {
            "port": 5801,
            "socket": mock_crossword_socket,
        }

        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "math",
                "_instance": "math_0",
                "accuracy": 0.8,
            }
        )
        aggregator._handle_log_payload(
            {
                "_step": 10,
                "_env_type": "crossword",
                "_instance": "crossword_0",
                "accuracy": 0.9,
            }
        )

        # Both should be sent to their respective leaders
        mock_math_socket.send_pyobj.assert_called_once()
        mock_crossword_socket.send_pyobj.assert_called_once()

        math_data = mock_math_socket.send_pyobj.call_args[0][0]
        crossword_data = mock_crossword_socket.send_pyobj.call_args[0][0]

        assert "math/instances/math_0/accuracy" in math_data
        assert "crossword/instances/crossword_0/accuracy" in crossword_data


class TestZMQLogger:
    """Tests for the ZMQLogger class."""

    def test_logger_sends_data(self):
        """Test that ZMQLogger sends data with correct metadata."""
        from atroposlib.utils.logging_client import ZMQLogger

        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://*:5720")

        logger = ZMQLogger(address="tcp://localhost:5720", context=context)
        time.sleep(0.1)

        logger.log(
            {"accuracy": 0.8},
            step=10,
            env_type="math",
            instance_name="math_0",
        )
        time.sleep(0.1)

        data = receiver.recv_pyobj(flags=zmq.NOBLOCK)
        assert data["_step"] == 10
        assert data["_env_type"] == "math"
        assert data["_instance"] == "math_0"
        assert data["accuracy"] == 0.8

        logger.close()
        receiver.close()
        context.term()


class TestLeaderElection:
    """Tests for leader election in server.py."""

    @pytest.fixture
    def app_state(self):
        """Create a mock app state."""

        class MockAppState:
            started = True
            envs = []
            env_leaders = {}
            next_leader_port = 5600
            status_dict = {"step": 0}
            save_checkpoint_interval = 100
            num_steps = 1000
            project = "test-project"
            group = "test-group"
            checkpoint_dir = "/tmp/checkpoints"

        return MockAppState()

    def test_first_instance_becomes_leader(self, app_state):
        """Test that first instance of an env_type becomes leader."""
        # Simulate the logic from register_env endpoint
        desired_name = "math"
        instance_index = len(
            [x for x in app_state.envs if x["desired_name"] == desired_name]
        )
        real_name = f"{desired_name}_{instance_index}"
        registered_id = len(app_state.envs)

        is_leader = desired_name not in app_state.env_leaders
        leader_receive_port = None

        if is_leader:
            leader_receive_port = app_state.next_leader_port
            app_state.next_leader_port += 1
            app_state.env_leaders[desired_name] = {
                "instance": real_name,
                "env_id": registered_id,
                "receive_port": leader_receive_port,
            }

        assert is_leader is True
        assert leader_receive_port == 5600
        assert "math" in app_state.env_leaders

    def test_second_instance_not_leader(self, app_state):
        """Test that second instance is not a leader."""
        # First instance
        app_state.env_leaders["math"] = {
            "instance": "math_0",
            "env_id": 0,
            "receive_port": 5600,
        }
        app_state.envs.append({"desired_name": "math", "real_name": "math_0"})

        # Second instance
        desired_name = "math"
        instance_index = len(
            [x for x in app_state.envs if x["desired_name"] == desired_name]
        )
        real_name = f"{desired_name}_{instance_index}"

        is_leader = desired_name not in app_state.env_leaders
        leader_receive_port = None

        assert is_leader is False
        assert leader_receive_port is None

    def test_different_env_types_have_own_leaders(self, app_state):
        """Test that different env_types each get their own leader."""
        # math leader
        app_state.env_leaders["math"] = {
            "instance": "math_0",
            "env_id": 0,
            "receive_port": 5600,
        }
        app_state.envs.append({"desired_name": "math", "real_name": "math_0"})
        app_state.next_leader_port = 5601

        # crossword - should get its own leader
        desired_name = "crossword"
        instance_index = len(
            [x for x in app_state.envs if x["desired_name"] == desired_name]
        )
        real_name = f"{desired_name}_{instance_index}"
        registered_id = len(app_state.envs)

        is_leader = desired_name not in app_state.env_leaders
        leader_receive_port = None

        if is_leader:
            leader_receive_port = app_state.next_leader_port
            app_state.next_leader_port += 1
            app_state.env_leaders[desired_name] = {
                "instance": real_name,
                "env_id": registered_id,
                "receive_port": leader_receive_port,
            }

        assert is_leader is True
        assert leader_receive_port == 5601
        assert "math" in app_state.env_leaders
        assert "crossword" in app_state.env_leaders
