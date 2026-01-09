"""
Tests for the Verifiers environment implementation.

These tests mock the verifiers library to allow testing without requiring
the actual verifiers package to be installed.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock verifiers before any imports that might use it
mock_vf = MagicMock()
mock_vf.load_environment = MagicMock()
sys.modules["verifiers"] = mock_vf

# Now we can safely import the environment
from atroposlib.envs.base import BaseEnvConfig  # noqa: E402
from environments.verifiers_server import VerifiersEnv, VfEnvConfig  # noqa: E402


def create_mock_reward_func(return_value=1.0):
    """Create a mock reward function with proper signature."""
    def reward_func(completion, answer, **kwargs):
        return return_value
    reward_func.__name__ = "mock_reward_func"
    return reward_func


@pytest.fixture
def mock_rubric():
    """Create a mock rubric with parser and reward functions."""
    rubric = MagicMock()
    rubric.parser = MagicMock()
    rubric.parser.parse_answer = MagicMock(return_value="parsed_answer")
    rubric.get_reward_funcs = MagicMock(return_value=[create_mock_reward_func(), create_mock_reward_func()])
    rubric.get_reward_weights = MagicMock(return_value=[1.0, 1.0])
    rubric.call_reward_func = AsyncMock(return_value=1.0)
    return rubric


@pytest.fixture
def mock_vf_env(mock_rubric):
    """Create a mock Verifiers environment."""
    vf_env = MagicMock()
    vf_env.rubric = mock_rubric
    vf_env.system_prompt = "You are a helpful assistant."
    vf_env.get_dataset = MagicMock(
        return_value=[
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]
    )
    vf_env.get_eval_dataset = MagicMock(
        return_value=[
            {"question": "What is 1+1?", "answer": "2"},
        ]
    )
    return vf_env


@pytest.fixture
def mock_server():
    """Create a mock server for API calls."""
    server = MagicMock()

    # Mock chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = "The answer is 4"
    mock_choice.finish_reason = "stop"

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice] * 8  # group_size responses

    server.chat_completion = AsyncMock(return_value=mock_completion)
    return server


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5] * 20)
    tokenizer.decode = MagicMock(return_value="decoded text")
    return tokenizer


class TestVfEnvConfig:
    """Tests for VfEnvConfig class."""

    def test_config_defaults(self):
        """Test that VfEnvConfig has correct default values."""
        config = VfEnvConfig()
        assert config.vf_env_name == ""
        assert config.env_args == {}
        assert config.group_size == 4  # inherited from BaseEnvConfig
        assert config.max_token_length == 2048
        assert config.reward_threshold == 0.5  # default threshold

    def test_config_custom_values(self):
        """Test VfEnvConfig with custom values."""
        config = VfEnvConfig(
            vf_env_name="wordle",
            env_args={"difficulty": "hard"},
            group_size=16,
            max_token_length=4096,
            reward_threshold=0.7,
        )
        assert config.vf_env_name == "wordle"
        assert config.env_args == {"difficulty": "hard"}
        assert config.group_size == 16
        assert config.max_token_length == 4096
        assert config.reward_threshold == 0.7

    def test_config_inherits_base_env_config(self):
        """Test that VfEnvConfig inherits from BaseEnvConfig."""
        assert issubclass(VfEnvConfig, BaseEnvConfig)


class TestVerifiersEnvInit:
    """Tests for VerifiersEnv initialization."""

    def test_config_init(self, mock_vf_env):
        """Test the config_init class method."""
        mock_vf.load_environment.return_value = mock_vf_env

        env_config, server_configs = VerifiersEnv.config_init()

        assert env_config.group_size == 8
        assert env_config.use_wandb is False
        assert env_config.total_steps == 10
        assert env_config.batch_size == 4
        assert len(server_configs) == 1
        assert server_configs[0].model_name == "gpt-4.1-nano"

    def test_env_has_name_attribute(self):
        """Test that VerifiersEnv has a name class attribute."""
        assert hasattr(VerifiersEnv, "name")
        assert VerifiersEnv.name == "verifiers"


class TestVerifiersEnvSetup:
    """Tests for VerifiersEnv setup method."""

    @pytest.mark.asyncio
    async def test_setup_loads_datasets(self, mock_vf_env):
        """Test that setup properly loads train and test datasets."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.vf_env = mock_vf_env
            env.iter = 0

            await env.setup()

            assert env.train == mock_vf_env.get_dataset()
            assert len(env.test) == 1
            assert env.test[0]["question"] == "What is 1+1?"
            assert env.test[0]["answer"] == "2"
            assert env.iter == 0


class TestVerifiersEnvGetNextItem:
    """Tests for get_next_item method."""

    @pytest.mark.asyncio
    async def test_get_next_item_returns_item(self, mock_vf_env):
        """Test that get_next_item returns the next training item."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.train = [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]
            env.iter = 0

            item1 = await env.get_next_item()
            assert item1["question"] == "Q1"
            assert env.iter == 1

            item2 = await env.get_next_item()
            assert item2["question"] == "Q2"
            assert env.iter == 2

    @pytest.mark.asyncio
    async def test_get_next_item_wraps_around(self, mock_vf_env):
        """Test that get_next_item wraps around when reaching end of dataset."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.train = [{"question": "Q1", "answer": "A1"}]
            env.iter = 0

            item1 = await env.get_next_item()
            item2 = await env.get_next_item()

            assert item1["question"] == "Q1"
            assert item2["question"] == "Q1"  # Should wrap around
            assert env.iter == 2


class TestVerifiersEnvScore:
    """Tests for the score method."""

    @pytest.mark.asyncio
    async def test_score_returns_scored_data_group(
        self, mock_vf_env, mock_rubric, mock_tokenizer
    ):
        """Test that score returns a properly formatted ScoredDataGroup."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.parser = mock_rubric.parser
            env.rubric = mock_rubric
            # Create mock reward functions that return different floats
            # Use side_effect to return different values for different calls
            call_count = [0]

            def mock_reward_func(completion, answer, **kwargs):
                call_count[0] += 1
                # Return high score for first call, low for second
                return 1.0 if call_count[0] == 1 else 0.0

            mock_reward_func.__name__ = "mock_reward_func"
            env.reward_funcs = [mock_reward_func]
            env.reward_scales = [1.0]
            env.tokenizer = mock_tokenizer
            env.config = MagicMock()
            env.config.group_size = 2
            env.config.reward_threshold = 0.5
            env.config.ensure_scores_are_not_same = True
            env.percent_correct_buffer = []

            # Mock tokenize_for_trainer
            with patch(
                "environments.verifiers_server.tokenize_for_trainer"
            ) as mock_tokenize:
                mock_tokenize.return_value = {
                    "tokens": [1, 2, 3, 4, 5] * 20,
                    "masks": [1, 1, 1, 1, 1] * 20,
                }

                rollout_data = [
                    {
                        "messages": (
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ),
                        "answer": "4",
                        "finish_reason": "stop",
                    },
                    {
                        "messages": (
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "5"},
                        ),
                        "answer": "4",
                        "finish_reason": "stop",
                    },
                ]

                result = await env.score(rollout_data)

                assert result is not None
                assert "tokens" in result
                assert "masks" in result
                assert "scores" in result
                assert len(result["tokens"]) == 2
                assert len(result["scores"]) == 2

    @pytest.mark.asyncio
    async def test_score_returns_none_when_all_same(
        self, mock_vf_env, mock_rubric, mock_tokenizer
    ):
        """Test that score returns None when all scores are the same."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.parser = mock_rubric.parser
            env.rubric = mock_rubric
            # All rewards return the same value - use proper signature
            env.reward_funcs = [create_mock_reward_func(1.0)]
            env.reward_scales = [1.0]
            env.tokenizer = mock_tokenizer
            env.config = MagicMock()
            env.config.group_size = 2
            env.config.reward_threshold = 0.5
            env.config.ensure_scores_are_not_same = True
            env.percent_correct_buffer = []

            with patch(
                "environments.verifiers_server.tokenize_for_trainer"
            ) as mock_tokenize:
                mock_tokenize.return_value = {
                    "tokens": [1, 2, 3, 4, 5] * 20,
                    "masks": [1, 1, 1, 1, 1] * 20,
                }

                rollout_data = [
                    {
                        "messages": (
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ),
                        "answer": "4",
                        "finish_reason": "stop",
                    },
                    {
                        "messages": (
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ),
                        "answer": "4",
                        "finish_reason": "stop",
                    },
                ]

                result = await env.score(rollout_data)

                assert result is None


class TestVerifiersEnvCollectTrajectories:
    """Tests for collect_trajectories method."""

    @pytest.mark.asyncio
    async def test_collect_trajectories_generates_completions(
        self, mock_vf_env, mock_server, mock_rubric, mock_tokenizer
    ):
        """Test that collect_trajectories generates multiple completions."""
        mock_vf.load_environment.return_value = mock_vf_env

        # Mock the vf_env class name to indicate single-turn
        mock_vf_env_single = MagicMock()
        mock_vf_env_single.__class__.__name__ = "SingleTurnEnv"

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.vf_env = mock_vf_env_single
            env.server = mock_server
            env.system_prompt = "You are helpful"
            env.config = MagicMock()
            env.config.group_size = 4
            env.config.max_token_length = 2048
            env.parser = mock_rubric.parser
            env.rubric = mock_rubric
            env.reward_funcs = [create_mock_reward_func()]
            env.reward_scales = [1.0]
            env.tokenizer = mock_tokenizer
            env.percent_correct_buffer = []

            # Mock the score method to return a valid ScoredDataGroup
            with patch.object(env, "score") as mock_score:
                mock_score.return_value = {
                    "tokens": [[1, 2, 3]],
                    "masks": [[1, 1, 1]],
                    "scores": [1.0],
                }

                item = {"question": "What is 2+2?", "answer": "4"}
                result, backlog = await env.collect_trajectories(item)

                # Verify chat_completion was called with correct params
                mock_server.chat_completion.assert_called_once()
                call_kwargs = mock_server.chat_completion.call_args[1]
                assert call_kwargs["n"] == 4
                assert call_kwargs["max_tokens"] == 2048

                # Verify score was called
                mock_score.assert_called_once()

                # Verify result structure
                assert result is not None
                assert backlog == []


class TestVerifiersEnvWandbLog:
    """Tests for wandb_log method."""

    @pytest.mark.asyncio
    async def test_wandb_log_calculates_percent_correct(self):
        """Test that wandb_log calculates percent_correct from buffer."""
        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.percent_correct_buffer = [1.0, 1.0, 0.0, 1.0]
            env.eval_metrics = []
            env.config = MagicMock()
            env.config.use_wandb = False

            # Mock the parent wandb_log
            with patch(
                "atroposlib.envs.base.BaseEnv.wandb_log", new_callable=AsyncMock
            ) as mock_parent_log:
                await env.wandb_log({})

                # Check that percent_correct was calculated
                call_args = mock_parent_log.call_args[0][0]
                assert "train/percent_correct" in call_args
                assert call_args["train/percent_correct"] == 0.75  # 3/4

                # Buffer should be cleared
                assert env.percent_correct_buffer == []

    @pytest.mark.asyncio
    async def test_wandb_log_handles_empty_buffer(self):
        """Test that wandb_log handles empty percent_correct_buffer."""
        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.percent_correct_buffer = []
            env.eval_metrics = []
            env.config = MagicMock()
            env.config.use_wandb = False

            with patch(
                "atroposlib.envs.base.BaseEnv.wandb_log", new_callable=AsyncMock
            ) as mock_parent_log:
                await env.wandb_log({})

                # Should not add percent_correct when buffer is empty
                call_args = mock_parent_log.call_args[0][0]
                assert "train/percent_correct" not in call_args


class TestVerifiersEnvEvaluate:
    """Tests for evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_runs_on_test_set(self, mock_vf_env, mock_server):
        """Test that evaluate runs evaluation on the test set."""
        mock_vf.load_environment.return_value = mock_vf_env

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.test = [{"question": "What is 1+1?", "answer": "2"}]
            env.server = mock_server
            env.config = MagicMock()
            env.config.max_token_length = 2048
            env.config.data_dir_to_save_evals = None
            env.system_prompt = "You are helpful"
            env.eval_metrics = []

            # Mock rollout_and_score_eval
            with patch.object(env, "rollout_and_score_eval") as mock_rollout:
                mock_rollout.return_value = {
                    "score": 1.0,
                    "sample": {"question": "What is 1+1?", "score": 1},
                }

                # Mock evaluate_log
                with patch.object(
                    env, "evaluate_log", new_callable=AsyncMock
                ) as mock_log:
                    result = await env.evaluate()

                    assert result["eval/avg_total_score"] == 1.0
                    mock_rollout.assert_called_once()
                    mock_log.assert_called_once()


class TestVerifiersEnvIntegration:
    """Integration tests for the full environment flow."""

    @pytest.mark.asyncio
    async def test_full_training_loop_flow(
        self, mock_vf_env, mock_server, mock_rubric, mock_tokenizer
    ):
        """Test a simplified training loop flow."""
        mock_vf.load_environment.return_value = mock_vf_env

        # Mock the vf_env class name to indicate single-turn
        mock_vf_env_single = MagicMock()
        mock_vf_env_single.__class__.__name__ = "SingleTurnEnv"
        mock_vf_env_single.get_dataset = mock_vf_env.get_dataset
        mock_vf_env_single.get_eval_dataset = mock_vf_env.get_eval_dataset

        with patch.object(VerifiersEnv, "__init__", lambda self, *args, **kwargs: None):
            env = VerifiersEnv.__new__(VerifiersEnv)
            env.vf_env = mock_vf_env_single
            env.server = mock_server
            env.system_prompt = "You are helpful"
            env.config = MagicMock()
            env.config.group_size = 2
            env.config.max_token_length = 2048
            env.parser = mock_rubric.parser
            env.rubric = mock_rubric
            env.reward_funcs = [create_mock_reward_func()]
            env.reward_scales = [1.0]
            env.tokenizer = mock_tokenizer
            env.percent_correct_buffer = []
            env.iter = 0

            # Setup
            await env.setup()
            assert len(env.train) == 2
            assert len(env.test) == 1

            # Get next item
            item = await env.get_next_item()
            assert item["question"] == "What is 2+2?"

            # Mock score to return valid data
            with patch.object(env, "score") as mock_score:
                mock_score.return_value = {
                    "tokens": [[1, 2, 3]],
                    "masks": [[1, 1, 1]],
                    "scores": [1.0],
                }

                # Collect trajectories
                result, backlog = await env.collect_trajectories(item)
                assert result is not None
                assert backlog == []
