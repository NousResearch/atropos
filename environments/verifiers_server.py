"""
Verifiers Training Environment for Atropos

Supports TWO modes:
- serve: RL training with local inference server (requires ManagedServer for logprobs)
- process: SFT data generation with ANY API (OpenAI, Claude, local, etc.)

Usage:
  # RL Training (requires local vLLM/SGLang server)
  python verifiers_server.py serve \
      --env.vf_env_name "will/wordle" \
      --openai.base_url http://localhost:9001/v1 \
      --slurm false

  # SFT Data Generation with OpenAI GPT-4o
  python verifiers_server.py process \
      --env.vf_env_name "will/wordle" \
      --env.data_path_to_save_groups gpt4o_sft_data.jsonl \
      --env.total_steps 100 \
      --env.group_size 4 \
      --openai.model_name gpt-4o \
      --openai.base_url https://api.openai.com/v1

  # SFT Data Generation with local server
  python verifiers_server.py process \
      --env.vf_env_name "will/wordle" \
      --env.data_path_to_save_groups local_sft_data.jsonl \
      --openai.base_url http://localhost:9001/v1

  # Evaluation (uses ManagedServer by default, falls back to direct API in process mode)
  python verifiers_server.py evaluate \
      --env.vf_env_name "will/wordle" \
      --openai.base_url http://localhost:9001/v1

To install a Verifiers/Prime environment:
1. uv tool install prime
2. prime login
3. prime env install will/wordle (or any owner/environment)
Docs: https://docs.primeintellect.ai/tutorials-environments/install
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from openai import AsyncOpenAI
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class VfEnvConfig(BaseEnvConfig):
    """
    Configuration for the Verifiers environments.
    """

    vf_env_name: str = ""
    env_args: Dict[str, Any] = Field(default_factory=dict)


class VerifiersEnv(BaseEnv):
    name = "verifiers"
    env_config_cls = VfEnvConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        self.rubric = self.vf_env.rubric

        self.parser = self.rubric.parser

        # Handle both single Rubric and RubricGroup (composite)
        # RubricGroup has empty funcs/weights at top level - must extract from individual rubrics
        if hasattr(self.rubric, "rubrics"):
            self.reward_funcs = []
            self.reward_weights = []
            for rubric in self.rubric.rubrics:
                self.reward_funcs.extend(rubric.funcs)
                self.reward_weights.extend(rubric.weights)
        else:
            self.reward_funcs = self.rubric.funcs
            self.reward_weights = self.rubric.weights

        total = sum(self.reward_weights) if self.reward_weights else 1.0
        self.reward_scales = [weight / total for weight in self.reward_weights]
        self.system_prompt = self.vf_env.system_prompt

    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, List[APIServerConfig]]:
        env_config = VfEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=4,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="verifiers",
        )
        # Default config for local inference server (vLLM, SGLang, TRL)
        # For SFT data generation with OpenAI, override via CLI:
        #   --openai.base_url https://api.openai.com/v1 --openai.model_name gpt-4o
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url="https://api.openai.com/v1",
                api_key="x",
                num_requests_for_eval=4,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate percent_correct from buffer
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        self.percent_correct_buffer = list()

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        train_data = self.vf_env.get_dataset()
        # Only load columns we need to avoid memory bloat
        columns_to_keep = ["question", "answer", "info"]
        available_columns = [c for c in columns_to_keep if c in train_data.column_names]
        self.train = train_data.select_columns(available_columns).to_list()
        test_data = self.vf_env.get_eval_dataset()
        available_test_columns = [
            c for c in columns_to_keep if c in test_data.column_names
        ]
        self.test = test_data.select_columns(available_test_columns).to_list()
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _compute_score(self, completion_messages: List[Dict], answer: str) -> float:
        """Compute score using verifiers reward functions."""
        rewards = []
        for func in self.reward_funcs:
            reward = func(
                parser=self.parser,
                completion=completion_messages,
                answer=answer,
            )
            rewards.append(reward)
        weighted_rewards = [r * self.reward_scales[j] for j, r in enumerate(rewards)]
        return sum(weighted_rewards)

    async def rollout_and_score_eval(
        self, question: str, answer: str, **kwargs
    ) -> dict:
        """
        Rollout and score for evaluation.
        Uses ManagedServer in serve mode, direct API calls in process mode.
        """
        system_prompt = kwargs.get("system_prompt")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        is_process_mode = getattr(self, "process_mode", False)

        if is_process_mode:
            # Process mode: use direct API call (works with any API)
            completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
            )
        else:
            # Serve mode: use ManagedServer for token tracking
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                completion = await managed.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.0,
                )

        response_content = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_content})

        answer_parsed = self.parser.parse_answer(completion=response_content)

        score = self._compute_score(messages, answer)

        sample = {
            "messages": messages,
            "question": question,
            "gold_answer": answer,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": score,
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(
                    item["question"], item["answer"], system_prompt=self.system_prompt
                )
            )
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        avg_total_score = sum(scores) / len(scores)

        end_time = time.time()

        self.eval_metrics.append(("eval/avg_total_score", avg_total_score))

        eval_metrics = {"eval/avg_total_score": avg_total_score}

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

        return eval_metrics

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        """
        Collect trajectories - switches between:
        - SFT data generation (process mode): Any API, no logprobs needed
        - RL training (serve mode): Local server with logprobs
        """
        is_process_mode = getattr(self, "process_mode", False)

        if is_process_mode:
            return await self._collect_trajectories_for_sft(item)
        else:
            return await self._collect_trajectories_for_rl(item)

    async def _collect_trajectories_for_sft(
        self, item: Dict[str, Any]
    ) -> Tuple[ScoredDataGroup, list]:
        """
        SFT data generation mode - works with ANY API (OpenAI, Claude, local).
        Does NOT require logprobs or local server.

        Uses verifiers rollout() for multi-turn environments and tokenize_for_trainer
        to tokenize completions with your training tokenizer.
        """
        question = item["question"]
        answer = item["answer"]

        # Build initial messages
        initial_messages: List[Dict[str, str]] = []
        if self.system_prompt:
            initial_messages.append({"role": "system", "content": self.system_prompt})
        initial_messages.append({"role": "user", "content": question})

        # Create AsyncOpenAI client directly from server config (no ManagedServer needed)
        server_config = self.server.servers[0].config
        client = AsyncOpenAI(
            api_key=server_config.api_key,
            base_url=server_config.base_url,
            timeout=server_config.timeout,
        )

        # Sampling args - use max_completion_tokens for newer models like gpt-5
        sampling_args = {
            "temperature": 1.0,
            "max_completion_tokens": self.config.max_token_length,
        }

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []

        # Semaphore for scoring (required by rubric.score_rollout)
        score_sem = asyncio.Semaphore(1)

        # Run rollouts in parallel for group_size
        async def run_single_rollout(example_id: int):
            # Pass through any info from the dataset item (e.g., docker_image for SWE envs)
            item_info = item.get("info", {})
            rollout_input = {
                "prompt": initial_messages,
                "answer": answer,
                "example_id": example_id,
                "task": self.config.vf_env_name,
                "info": item_info,
            }
            state = await self.vf_env.rollout(
                input=rollout_input,
                client=client,
                model=server_config.model_name,
                sampling_args=sampling_args,
            )
            # Score the rollout using verifiers rubric (computes reward from test output)
            # This is needed because vf_env.rollout() doesn't call score_rollout
            await self.rubric.score_rollout(state, score_sem=score_sem)
            return state

        # Run group_size rollouts in parallel
        rollout_tasks = [run_single_rollout(i) for i in range(self.config.group_size)]
        states = await asyncio.gather(*rollout_tasks)

        for state in states:
            # Extract completion messages from state
            completion_messages = list(state.get("prompt", [])) + list(
                state.get("completion", [])
            )
            # Ensure all message contents are strings (not None)
            # This can happen with tool call messages that have content: null
            completion_messages = [
                {**msg, "content": msg.get("content") or ""}
                for msg in completion_messages
            ]

            # Get reward from verifiers scoring (set by rubric.score_rollout above)
            score = state.get("reward", 0.0)

            # Determine finish reason from last trajectory step
            trajectory = state.get("trajectory", [])
            if trajectory:
                finish_reason = trajectory[-1]["response"].choices[0].finish_reason
            else:
                finish_reason = "stop"

            # Use tokenize_for_trainer for tokenization
            # train_on_all_assistant_turns=True ensures ALL assistant turns are unmasked
            # for multi-turn environments, not just the last message
            tokenized = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=completion_messages,
                include_messages=True,
                finish_reason=finish_reason,
                train_on_all_assistant_turns=True,
            )

            scored_data["tokens"].append(tokenized["tokens"])
            scored_data["masks"].append(tokenized["masks"])
            scored_data["messages"].append(completion_messages)
            scored_data["scores"].append(score)

        # Track scores for wandb logging
        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []

    async def _collect_trajectories_for_rl(
        self, item: Dict[str, Any]
    ) -> Tuple[ScoredDataGroup, list]:
        """
        RL training mode - requires local inference server for logprobs.
        Uses AtroposManagedClient with vf_env.rollout() for both single-turn and multi-turn.
        """
        from atroposlib.envs.server_handling.atropos_managed_client import (
            AtroposManagedClient,
        )

        question = item["question"]
        answer = item["answer"]
        item_info = item.get("info", {})

        initial_messages: List[Dict[str, str]] = []
        if self.system_prompt:
            initial_messages.append({"role": "system", "content": self.system_prompt})
        initial_messages.append({"role": "user", "content": question})

        sampling_args = {
            "temperature": 1.0,
            "max_completion_tokens": self.config.max_token_length,
        }

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["inference_logprobs"] = []

        # Semaphore for scoring (required by rubric.score_rollout)
        score_sem = asyncio.Semaphore(1)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            client = AtroposManagedClient(
                managed_server=managed,
                model=self.server_configs[0].model_name,
            )

            # Run group_size rollouts sequentially (ManagedServer state must be reset between)
            for i in range(self.config.group_size):
                client.reset()

                rollout_input = {
                    "prompt": initial_messages,
                    "answer": answer,
                    "example_id": i,
                    "task": self.config.vf_env_name,
                    "info": item_info,
                }

                state = await self.vf_env.rollout(
                    input=rollout_input,
                    client=client,
                    model=self.server_configs[0].model_name,
                    sampling_args=sampling_args,
                )

                # Score the rollout (computes reward from test output)
                await self.rubric.score_rollout(state, score_sem=score_sem)

                tokens, masks, logprobs, score = self._extract_from_state(state)
                scored_data["tokens"].append(tokens)
                scored_data["masks"].append(masks)
                scored_data["inference_logprobs"].append(logprobs)
                scored_data["scores"].append(score)

        # Track scores for wandb logging
        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []

    def _extract_from_state(
        self, state: Any
    ) -> Tuple[List[int], List[int], List[float], float]:
        """
        Extract tokens/masks/logprobs/score from a single rollout state.

        Handles the mask convention conversion:
        - Verifiers: prompt_mask=0, completion_mask=1
        - Atropos: masked_tokens=-100 (prompt), token_id (completion)
        """
        all_tokens: List[int] = []
        all_masks: List[int] = []
        all_logprobs: List[float] = []

        trajectory = state.get("trajectory", [])

        for step in trajectory:
            tokens = step["tokens"]

            prompt_ids = tokens["prompt_ids"]
            completion_ids = tokens["completion_ids"]
            completion_logprobs = tokens["completion_logprobs"]

            all_tokens.extend(prompt_ids)
            all_tokens.extend(completion_ids)

            all_masks.extend([-100] * len(prompt_ids))
            all_masks.extend(completion_ids)

            all_logprobs.extend([1.0] * len(prompt_ids))
            all_logprobs.extend(completion_logprobs)

        reward = state["reward"]

        return all_tokens, all_masks, all_logprobs, reward


if __name__ == "__main__":
    VerifiersEnv.cli()
