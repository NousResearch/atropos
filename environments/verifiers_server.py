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

import time
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
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
        self.reward_funcs = self.rubric.funcs
        self.reward_weights = self.rubric.weights
        self.reward_scales = [
            weight / sum(self.reward_weights) for weight in self.reward_weights
        ]
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
        self.train = train_data.select_columns(["question", "answer"]).to_list()
        test_data = self.vf_env.get_eval_dataset()
        self.test = test_data.select_columns(["question", "answer"]).to_list()
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
        Collect trajectories - automatically switches between:
        - ManagedServer (for RL training with local server - requires logprobs)
        - tokenize_for_trainer (for SFT datagen with any API - no logprobs needed)
        """
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Check if we're in process mode (SFT data generation)
        is_process_mode = getattr(self, "process_mode", False)

        if is_process_mode:
            return await self._collect_trajectories_for_sft(messages, answer)
        else:
            return await self._collect_trajectories_for_training(messages, answer)

    async def _collect_trajectories_for_sft(
        self, messages: List[Dict], answer: str
    ) -> Tuple[ScoredDataGroup, list]:
        """
        SFT data generation mode - works with ANY API (OpenAI, Claude, local).
        Does NOT require logprobs or local server.

        Uses tokenize_for_trainer to tokenize completions with your training
        tokenizer, so the resulting data is ready for fine-tuning your target model.
        """
        completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=1.0,
        )

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []
        # Note: No inference_logprobs - not needed/available for SFT

        for choice in completions.choices:
            response = choice.message.content or ""
            finish_reason = choice.finish_reason or ""

            # Build full conversation for scoring and tokenization
            completion_messages = messages + [
                {"role": "assistant", "content": response}
            ]

            # Score using verifiers reward funcs
            score = self._compute_score(completion_messages, answer)

            # Use tokenize_for_trainer for tokenization
            # This uses YOUR training tokenizer (e.g., Qwen, Llama), not the API's tokenizer
            # So GPT-4o responses get tokenized for your target model
            tokenized = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=completion_messages,
                include_messages=True,
                finish_reason=finish_reason,
            )

            scored_data["tokens"].append(tokenized["tokens"])
            scored_data["masks"].append(tokenized["masks"])
            scored_data["messages"].append(completion_messages)
            scored_data["scores"].append(score)

        # Track scores for wandb logging
        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []

    async def _collect_trajectories_for_training(
        self, messages: List[Dict], answer: str
    ) -> Tuple[ScoredDataGroup, list]:
        """
        RL training mode - requires local inference server.
        Uses ManagedServer for proper token/logprob alignment.

        The inference_logprobs are required for policy gradient methods like
        GRPO, PPO, REINFORCE, etc.
        """
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completions = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )
            state = managed.get_state()
            nodes = state["nodes"]

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["inference_logprobs"] = []  # Required for RL training!

        for i, choice in enumerate(completions.choices):
            response = choice.message.content or ""

            # Build full conversation for scoring
            completion_messages = messages + [
                {"role": "assistant", "content": response}
            ]

            # Score using verifiers reward funcs
            score = self._compute_score(completion_messages, answer)

            # Use ManagedServer's properly aligned tokens/masks/logprobs
            node = nodes[i]
            scored_data["tokens"].append(node.tokens)
            scored_data["masks"].append(node.masked_tokens)
            scored_data["inference_logprobs"].append(node.logprobs)
            scored_data["scores"].append(score)

        # Track scores for wandb logging
        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []


if __name__ == "__main__":
    VerifiersEnv.cli()
