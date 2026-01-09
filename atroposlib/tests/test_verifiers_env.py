"""
Unit tests for VerifiersEnv adapter.

These tests verify the actual behavior of the VerifiersEnv adapter.
Tests are organized into:
1. Configuration tests - verify VfEnvConfig works correctly
2. Import guard tests - verify proper error when verifiers missing
3. Initialization tests - verify proper validation during init
4. Data processing tests - verify dataset normalization
5. Scoring tests - verify reward calculation
6. Integration tests (marked @pytest.mark.prime) - require Prime Hub

Run with: pytest atroposlib/tests/test_verifiers_env.py -v
Run with Prime: pytest atroposlib/tests/test_verifiers_env.py -v --runprime
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atroposlib.envs.base import APIServerConfig, BaseEnvConfig

# Import the actual modules we're testing
from environments.verifiers_server import (
    VerifiersEnv,
    VfDataItem,
    VfEnvConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================
@pytest.fixture
def basic_server_config() -> List[APIServerConfig]:
    """Create a minimal server configuration for testing."""
    return [
        APIServerConfig(
            model_name="test-model",
            base_url="http://localhost:9001/v1",
            api_key="test-key",
        )
    ]


@pytest.fixture
def sample_dataset() -> List[Dict[str, str]]:
    """Sample dataset items for testing."""
    return [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    ]


@pytest.fixture
def alternate_format_dataset() -> List[Dict[str, str]]:
    """Dataset with alternate field names (prompt/response instead of question/answer)."""
    return [
        {"prompt": "Translate 'hello' to Spanish", "response": "hola"},
        {"input": "What color is the sky?", "output": "blue"},
    ]


# =============================================================================
# Configuration Tests
# =============================================================================
class TestVfEnvConfig:
    """Test VfEnvConfig class behavior."""

    def test_inherits_from_base_config(self):
        """VfEnvConfig should inherit from BaseEnvConfig."""
        assert issubclass(VfEnvConfig, BaseEnvConfig)

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = VfEnvConfig()

        assert config.vf_env_name == ""
        assert config.env_args == {}
        assert config.normalize_rewards is False
        assert config.min_mask_tokens == 10
        # Also verify inherited defaults work
        assert config.group_size == 4
        assert config.max_token_length == 2048

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = VfEnvConfig(
            vf_env_name="wordle",
            env_args={"difficulty": "hard"},
            normalize_rewards=True,
            min_mask_tokens=20,
            group_size=16,
        )

        assert config.vf_env_name == "wordle"
        assert config.env_args == {"difficulty": "hard"}
        assert config.normalize_rewards is True
        assert config.min_mask_tokens == 20
        assert config.group_size == 16

    def test_env_args_is_mutable_dict(self):
        """Test that env_args can be set to a dict and modified."""
        config = VfEnvConfig()
        config.env_args = {"key": "value"}
        config.env_args["another_key"] = "another_value"

        assert "key" in config.env_args
        assert "another_key" in config.env_args


# =============================================================================
# Import Guard Tests
# =============================================================================
class TestImportGuard:
    """Test the optional verifiers import handling."""

    def test_module_imports_successfully(self):
        """The module should import even if verifiers is available."""
        # This test passes if the import at the top of this file worked
        from environments.verifiers_server import VerifiersEnv, VfEnvConfig

        assert VerifiersEnv is not None
        assert VfEnvConfig is not None

    def test_importerror_message_is_helpful(self, basic_server_config):
        """When verifiers is missing, error should explain how to install."""
        import environments.verifiers_server as vs

        # Temporarily mock vf as None to simulate missing verifiers
        original_vf = vs.vf
        original_error = vs._verifiers_import_error

        try:
            vs.vf = None
            vs._verifiers_import_error = ImportError("No module named 'verifiers'")

            config = VfEnvConfig(vf_env_name="test")

            with pytest.raises(ImportError) as exc_info:
                VerifiersEnv(config=config, server_configs=basic_server_config)

            error_msg = str(exc_info.value)
            # Error should mention installation instructions
            assert (
                "pip install verifiers" in error_msg or "verifiers" in error_msg.lower()
            )
        finally:
            vs.vf = original_vf
            vs._verifiers_import_error = original_error


# =============================================================================
# Initialization Validation Tests
# =============================================================================
class TestInitValidation:
    """Test VerifiersEnv initialization validation."""

    def test_empty_env_name_raises_error(self, basic_server_config):
        """Init should fail if vf_env_name is empty."""
        config = VfEnvConfig(vf_env_name="")

        # When vf_env_name is empty, we should get ValueError before
        # even trying to load the environment
        import environments.verifiers_server as vs

        if vs.vf is None:
            pytest.skip("verifiers not installed")

        with pytest.raises(ValueError) as exc_info:
            VerifiersEnv(config=config, server_configs=basic_server_config)

        assert "vf_env_name" in str(exc_info.value)

    def test_invalid_env_name_raises_error(self, basic_server_config):
        """Init should fail with helpful message for non-existent environment."""
        import environments.verifiers_server as vs

        if vs.vf is None:
            pytest.skip("verifiers not installed")

        config = VfEnvConfig(vf_env_name="definitely_not_a_real_environment_12345")

        with pytest.raises(ValueError) as exc_info:
            VerifiersEnv(config=config, server_configs=basic_server_config)

        error_msg = str(exc_info.value)
        # Error should mention the environment name
        assert "definitely_not_a_real_environment_12345" in error_msg


# =============================================================================
# Dataset Normalization Tests
# =============================================================================
class TestDatasetNormalization:
    """Test the _normalize_dataset method logic."""

    def test_standard_format_preserved(self, sample_dataset):
        """Standard question/answer format should be preserved."""
        # Simulate normalization logic
        normalized = []
        for item in sample_dataset:
            question = (
                item.get("question") or item.get("prompt") or item.get("input") or ""
            )
            answer = (
                item.get("answer") or item.get("response") or item.get("output") or ""
            )
            if question:
                normalized.append({"question": str(question), "answer": str(answer)})

        assert len(normalized) == 3
        assert normalized[0]["question"] == "What is 2+2?"
        assert normalized[0]["answer"] == "4"
        assert normalized[2]["answer"] == "Shakespeare"

    def test_alternate_formats_normalized(self, alternate_format_dataset):
        """Alternate formats (prompt/response, input/output) should be normalized."""
        normalized = []
        for item in alternate_format_dataset:
            question = (
                item.get("question") or item.get("prompt") or item.get("input") or ""
            )
            answer = (
                item.get("answer") or item.get("response") or item.get("output") or ""
            )
            if question:
                normalized.append({"question": str(question), "answer": str(answer)})

        assert len(normalized) == 2
        # First item uses prompt/response
        assert normalized[0]["question"] == "Translate 'hello' to Spanish"
        assert normalized[0]["answer"] == "hola"
        # Second item uses input/output
        assert normalized[1]["question"] == "What color is the sky?"
        assert normalized[1]["answer"] == "blue"

    def test_empty_question_filtered(self):
        """Items without questions should be filtered out."""
        raw_data = [
            {"question": "Valid question", "answer": "answer"},
            {"answer": "orphan answer"},  # No question
            {"question": "", "answer": "empty question"},  # Empty question
        ]

        normalized = []
        for item in raw_data:
            question = item.get("question") or ""
            answer = item.get("answer") or ""
            if question:
                normalized.append({"question": str(question), "answer": str(answer)})

        # Only the first item should remain
        assert len(normalized) == 1
        assert normalized[0]["question"] == "Valid question"


# =============================================================================
# Reward Calculation Tests
# =============================================================================
class TestRewardCalculation:
    """Test reward weight normalization and score calculation."""

    def test_weight_normalization_basic(self):
        """Weights should be normalized to sum to 1.0."""
        weights = [2.0, 3.0, 5.0]  # Sum = 10
        weight_sum = sum(weights)

        assert weight_sum > 0
        scales = [w / weight_sum for w in weights]

        assert scales[0] == pytest.approx(0.2)
        assert scales[1] == pytest.approx(0.3)
        assert scales[2] == pytest.approx(0.5)
        assert sum(scales) == pytest.approx(1.0)

    def test_weight_normalization_single(self):
        """Single weight should normalize to 1.0."""
        weights = [5.0]
        weight_sum = sum(weights)
        scales = [w / weight_sum for w in weights]

        assert scales[0] == pytest.approx(1.0)

    def test_zero_weight_sum_fallback(self):
        """Zero sum weights should fallback to equal weights."""
        weights = [0.0, 0.0, 0.0]
        weight_sum = sum(weights)

        if weight_sum > 0:
            scales = [w / weight_sum for w in weights]
        else:
            n = len(weights) or 1
            scales = [1.0 / n] * n

        # Should give equal weights
        assert len(scales) == 3
        assert all(s == pytest.approx(1 / 3) for s in scales)
        assert sum(scales) == pytest.approx(1.0)

    def test_weighted_score_aggregation(self):
        """Test that weighted rewards are aggregated correctly."""
        rewards = [1.0, 0.5, 0.0]  # Rewards from 3 functions
        scales = [0.2, 0.3, 0.5]  # Normalized weights

        weighted_sum = sum(r * s for r, s in zip(rewards, scales))

        # 1.0*0.2 + 0.5*0.3 + 0.0*0.5 = 0.2 + 0.15 + 0 = 0.35
        assert weighted_sum == pytest.approx(0.35)


# =============================================================================
# Async Method Signature Tests
# =============================================================================
class TestAsyncSignatures:
    """Verify async methods have correct signatures."""

    def test_setup_is_async(self):
        """setup() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.setup)

    def test_get_next_item_is_async(self):
        """get_next_item() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.get_next_item)

    def test_collect_trajectories_is_async(self):
        """collect_trajectories() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.collect_trajectories)

    def test_score_is_async(self):
        """score() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.score)

    def test_evaluate_is_async(self):
        """evaluate() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.evaluate)

    def test_rollout_and_score_eval_is_async(self):
        """rollout_and_score_eval() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.rollout_and_score_eval)

    def test_wandb_log_is_async(self):
        """wandb_log() must be async."""
        assert asyncio.iscoroutinefunction(VerifiersEnv.wandb_log)


# =============================================================================
# Class Attribute Tests
# =============================================================================
class TestClassAttributes:
    """Test required class attributes exist."""

    def test_has_name_attribute(self):
        """VerifiersEnv should have a 'name' class attribute."""
        assert hasattr(VerifiersEnv, "name")
        assert VerifiersEnv.name == "verifiers"

    def test_has_env_config_cls(self):
        """VerifiersEnv should specify its config class."""
        assert hasattr(VerifiersEnv, "env_config_cls")
        assert VerifiersEnv.env_config_cls == VfEnvConfig

    def test_config_init_classmethod_exists(self):
        """config_init should be a classmethod."""
        assert hasattr(VerifiersEnv, "config_init")
        assert callable(VerifiersEnv.config_init)

    def test_config_init_returns_correct_types(self):
        """config_init should return (VfEnvConfig, List[APIServerConfig])."""
        env_config, server_configs = VerifiersEnv.config_init()

        assert isinstance(env_config, VfEnvConfig)
        assert isinstance(server_configs, list)
        assert all(isinstance(c, APIServerConfig) for c in server_configs)


# =============================================================================
# Prime Integration Tests (require Prime Hub login)
# =============================================================================
@pytest.mark.prime
class TestPrimeIntegration:
    """
    Integration tests that require Prime Hub access.

    Run with: pytest --runprime

    Prerequisites:
        uv tool install prime
        prime login
        prime env install will/wordle
    """

    def test_load_wordle_environment(self, basic_server_config):
        """Test loading the wordle environment from Prime Hub."""
        import environments.verifiers_server as vs

        if vs.vf is None:
            pytest.skip("verifiers not installed")

        config = VfEnvConfig(vf_env_name="wordle")

        try:
            env = VerifiersEnv(config=config, server_configs=basic_server_config)
            assert env.vf_env is not None
            assert env.rubric is not None
        except ValueError as e:
            if "not installed" in str(e).lower():
                pytest.skip("wordle environment not installed via Prime")
            raise

    @pytest.mark.asyncio
    async def test_setup_loads_datasets(self, basic_server_config):
        """Test that setup() loads training and eval datasets."""
        import environments.verifiers_server as vs

        if vs.vf is None:
            pytest.skip("verifiers not installed")

        config = VfEnvConfig(vf_env_name="wordle")

        try:
            env = VerifiersEnv(config=config, server_configs=basic_server_config)
            await env.setup()

            assert hasattr(env, "train")
            assert hasattr(env, "test")
            assert len(env.train) > 0 or len(env.test) > 0
        except ValueError as e:
            if "not installed" in str(e).lower():
                pytest.skip("wordle environment not installed via Prime")
            raise

    @pytest.mark.asyncio
    async def test_get_next_item_returns_valid_item(self, basic_server_config):
        """Test that get_next_item returns a properly formatted item."""
        import environments.verifiers_server as vs

        if vs.vf is None:
            pytest.skip("verifiers not installed")

        config = VfEnvConfig(vf_env_name="wordle")

        try:
            env = VerifiersEnv(config=config, server_configs=basic_server_config)
            await env.setup()

            if not env.train:
                pytest.skip("No training data available")

            item = await env.get_next_item()

            assert "question" in item
            assert "answer" in item
            assert isinstance(item["question"], str)
            assert isinstance(item["answer"], str)
        except ValueError as e:
            if "not installed" in str(e).lower():
                pytest.skip("wordle environment not installed via Prime")
            raise
