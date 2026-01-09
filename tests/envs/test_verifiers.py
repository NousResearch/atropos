from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.verifiers import VerifiersEnv, VfEnvConfig

# Mock data
MOCK_ITEM = {"question": "What is 2+2?", "answer": "4"}
MOCK_RESPONSE_CONTENT = "The answer is 4."
MOCK_VF_ENV_NAME = "mock/env"


@pytest.fixture
def mock_verifiers_module():
    with patch("atroposlib.envs.verifiers.vf") as mock_vf:
        # Mock environment setup
        mock_env = MagicMock()
        mock_vf.load_environment.return_value = mock_env

        # Mock rubric and parser
        mock_rubric = MagicMock()
        mock_env.rubric = mock_rubric
        mock_parser = MagicMock()
        mock_rubric.parser = mock_parser

        # Mock datasets
        mock_env.get_dataset.return_value = [MOCK_ITEM]
        mock_env.get_eval_dataset.return_value = [MOCK_ITEM]

        # Mock rewards
        mock_reward_func = MagicMock()  # Represents a function
        mock_rubric.get_reward_funcs.return_value = [mock_reward_func]
        mock_rubric.get_reward_weights.return_value = [1.0]

        # Mock reward call
        async def async_reward(*args, **kwargs):
            return 1.0

        mock_rubric.call_reward_func = AsyncMock(side_effect=async_reward)

        # Mock parser call
        mock_parser.parse_answer.return_value = "4"

        yield mock_vf


@pytest.fixture
def env_config():
    return VfEnvConfig(vf_env_name=MOCK_VF_ENV_NAME)


@pytest.fixture
def server_config():
    return [
        APIServerConfig(
            model_name="test-model", base_url="http://test", api_key="sk-test"
        )
    ]


@pytest.mark.asyncio
async def test_verifiers_env_initialization(
    mock_verifiers_module, env_config, server_config
):
    env = VerifiersEnv(config=env_config, server_configs=server_config)

    # Check if load_environment was called
    mock_verifiers_module.load_environment.assert_called_once_with(MOCK_VF_ENV_NAME)

    # Check if setup loads data
    # Mock methods on the instance to avoid network calls
    env.register_env = AsyncMock()
    env.setup_wandb = AsyncMock()

    await env.setup()

    assert len(env.train_data) == 1
    assert len(env.test_data) == 1
    assert env.iter == 0


@pytest.mark.asyncio
async def test_verifiers_env_collect_trajectory(
    mock_verifiers_module, env_config, server_config
):
    env = VerifiersEnv(config=env_config, server_configs=server_config)

    # Mock server response
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = MOCK_RESPONSE_CONTENT
    mock_completion.choices[0].finish_reason = "stop"

    env.server = MagicMock()
    env.server.chat_completion = AsyncMock(return_value=mock_completion)

    # Mock tokenizer
    env.tokenizer = MagicMock()
    env.tokenizer.encode.return_value = [1, 2, 3, 4]
    env.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4]

    # Call collect_trajectory
    scored_item, backlog = await env.collect_trajectory(MOCK_ITEM)

    assert backlog == []
    assert scored_item is not None
    assert scored_item["scores"] == 1.0
    assert scored_item["tokens"] == [1, 2, 3, 4]
    assert scored_item["masks"] == [1, 1, 1, 1]

    # Validate structure
    assert "messages" in scored_item
    messages = scored_item["messages"]
    assert len(messages) == 3  # system, user, assistant
    assert messages[-1]["content"] == MOCK_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_verifiers_env_get_next_item(
    mock_verifiers_module, env_config, server_config
):
    env = VerifiersEnv(config=env_config, server_configs=server_config)

    # Mock methods on the instance
    env.register_env = AsyncMock()
    env.setup_wandb = AsyncMock()

    await env.setup()

    item = await env.get_next_item()
    assert item == MOCK_ITEM

    # Iter should increment
    assert env.iter == 1
