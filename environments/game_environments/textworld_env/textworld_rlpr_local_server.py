#!/usr/bin/env python3
"""
Local test runner for the RLPR TextWorld environment (agent + VR-CLI flow).
"""

import asyncio
import logging
import os
from types import SimpleNamespace
from typing import Any, List

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.agents.atropos_agent import AtroposAgentConfig
from environments.game_environments.textworld_env.textworld_env_rlpr import (
    TextWorldEnv as TextWorldEnvRLPR,
)
from environments.game_environments.textworld_env.textworld_env_rlpr import (
    TextWorldEnvConfig as TextWorldEnvRLPRConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting RLPR TextWorld environment local debug runner")

    # Keep the config light for local tests
    tokenizer_name = os.getenv(
        "TOKENIZER_NAME",
        os.getenv("HF_TOKENIZER", "NousResearch/DeepHermes-3-Mistral-24B-Preview"),
    )
    env_config = TextWorldEnvRLPRConfig(
        tokenizer_name=tokenizer_name,
        group_size=2,  # keep small for local testing
        use_wandb=False,
        wandb_name="textworld_rlpr_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=32768,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        max_steps=3,
        # For debugging, include messages in SDGs for inspection
        include_messages=True,
    )

    # Tighten agent generation for faster local runs
    env_config.atropos_agent_config = AtroposAgentConfig(
        max_tokens_per_completion=128,
        temperature=0.7,
    )

    # Single server config: you can swap to local SGLang/OpenAI-compatible server
    mock_mode = os.getenv("MOCK_SERVER", "0") == "1"
    server_configs = [
        APIServerConfig(
            model_name=os.getenv("TEST_MODEL", "DeepHermes-3-Mistral-24B-Preview-q8"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            num_requests_for_eval=0,
        )
    ]

    env = TextWorldEnvRLPR(
        config=env_config, server_configs=server_configs, slurm=False, testing=False
    )

    if not mock_mode:
        await env.setup()

    # Optional Mock Server to test rollout loop without network
    if mock_mode:

        class _MockChoice:
            def __init__(self, text: str, logprobs: Any = None):
                self.text = text
                self.logprobs = logprobs

        class _MockLogProbs:
            def __init__(self, n: int = 256):
                # Uniform logprobs; sufficient for perplexity calculation
                self.token_logprobs = [-1.0] * n

        class _MockCompletion:
            def __init__(self, text: str, with_logprobs: bool = False):
                lp = _MockLogProbs(512) if with_logprobs else None
                self.choices = [_MockChoice(text=text, logprobs=lp)]
                self.usage = SimpleNamespace(
                    prompt_tokens=256, completion_tokens=len(text)
                )

        class MockServer:
            async def completion(
                self,
                prompt: str,
                max_tokens: int = 0,
                echo: bool = False,
                logprobs: int = 0,
                **kwargs,
            ):
                # Produce a minimal valid tool call with think/memory
                action = "look"
                response = (
                    "<think>Thinking about the next best action.</think>\n"
                    "<memory>Explored room; consider checking surroundings.</memory>\n"
                    f'<tool_call>{{"name": "execute_command", "arguments": {{"command": "{action}", "expected_outcome": "I see the room description"}}}}</tool_call>'
                )
                return _MockCompletion(response, with_logprobs=echo and logprobs)

            async def chat_completion(self, **kwargs):
                # Not used on RLPR path; provide stub
                return self.completion(prompt="", max_tokens=0)

        env.server = MockServer()  # type: ignore

    # Create one episode and run it
    item = await env.get_next_item()
    if not item:
        logger.error("Failed to get an RLPR textworld episode")
        return

    logger.info(f"Running RLPR episode: {item.get('episode_id', 'unknown')}")
    sdgs, _ = await env.collect_trajectories(item)

    if isinstance(sdgs, list):
        logger.info(f"Collected {len(sdgs)} ScoredDataGroups (turns)")
    elif isinstance(sdgs, dict):
        logger.info("Collected 1 ScoredDataGroup")
    else:
        logger.warning("No ScoredDataGroups returned from collect_trajectories")

    await env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
