#
# To install a Verifiers/Prime environment:
# 1. uv tool install prime
# 2. prime login
# 3. prime env install will/wordle (or any owner/environment)
#
# Or just run with --env.vf_env_name <env> and it will auto-install!
#
import logging
import os
import random
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Union

from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Import verifiers with guard for optional dependency
try:
    import verifiers as vf
except ImportError:
    vf = None

logger = logging.getLogger(__name__)


class VfEnvConfig(BaseEnvConfig):
    vf_env_name: str = ""
    env_args: Dict = {}
    reward_threshold: float = 0.5  # Configurable threshold for binary rewards


class VerifiersEnv(BaseEnv):
    name = "verifiers"

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        if vf is None:
            raise ImportError(
                "verifiers package is required for VerifiersEnv. "
                "Install with: pip install 'atroposlib[verifiers]'"
            )

        super().__init__(config, server_configs, slurm, testing)
        self.eval_metrics = list()
        self.percent_correct_buffer = list()

        self.vf_env = self._load_or_install_environment(
            config.vf_env_name, **config.env_args
        )
        self.rubric = self.vf_env.rubric

        self.parser = self.rubric.parser
        # Compatibility layer for public/private API (changed in verifiers >= 0.1.9)
        self.reward_funcs = self._get_rubric_reward_funcs()
        self.reward_weights = self._get_rubric_reward_weights()
        total_weight = sum(self.reward_weights) if self.reward_weights else 1.0
        self.reward_scales = (
            [weight / total_weight for weight in self.reward_weights]
            if self.reward_weights
            else []
        )
        self.system_prompt = self.vf_env.system_prompt

    def _load_or_install_environment(self, env_name: str, **env_args):
        """Load environment, auto-installing via prime CLI if not found.

        Args:
            env_name: Environment name. Can be either:
                - Full format: "owner/env" (e.g., "will/wordle")
                - Short format: "env" (e.g., "wordle")
                For auto-install, use full format "owner/env".
        """
        # Parse env_name - extract module name for loading
        # Format can be "owner/env_name" or just "env_name"
        module_name = env_name.split("/")[-1] if "/" in env_name else env_name

        try:
            return vf.load_environment(module_name, **env_args)
        except (ValueError, ImportError) as e:
            if "Could not import" in str(e) or "No module named" in str(e):
                logger.info(
                    "Environment '%s' not found, attempting to install via prime CLI",
                    env_name,
                )

                # For installation, we need the full "owner/env" format
                if "/" not in env_name:
                    raise RuntimeError(
                        f"Environment '{env_name}' not found. "
                        f"For auto-install, use full format: --env.vf_env_name owner/{env_name} "
                        f"(e.g., will/wordle)"
                    ) from e

                try:
                    # Try to install via prime CLI
                    result = subprocess.run(
                        ["prime", "env", "install", env_name],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if result.returncode == 0:
                        logger.info("Successfully installed environment '%s'", env_name)
                        # Try loading again after install (use module name)
                        return vf.load_environment(module_name, **env_args)
                    else:
                        logger.error(
                            "Failed to install environment '%s': %s",
                            env_name,
                            result.stderr or result.stdout,
                        )
                        raise RuntimeError(
                            f"Failed to install environment '{env_name}'. "
                            f"Make sure you're logged in with 'prime login'. "
                            f"Error: {result.stderr or result.stdout}"
                        ) from e
                except FileNotFoundError:
                    raise RuntimeError(
                        f"Environment '{env_name}' not found and 'prime' CLI is not "
                        "installed. Install with: uv tool install prime"
                    ) from e
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"Timeout installing environment '{env_name}'. "
                        "Try manually: prime env install " + env_name
                    ) from e
            else:
                raise

    def _get_rubric_reward_funcs(self) -> List:
        """Get reward functions with compatibility for different verifiers versions."""
        if hasattr(self.rubric, "get_reward_funcs"):
            return self.rubric.get_reward_funcs()
        elif hasattr(self.rubric, "_get_reward_funcs"):
            return self.rubric._get_reward_funcs()
        else:
            logger.warning("Could not find reward_funcs method on rubric")
            return []

    def _get_rubric_reward_weights(self) -> List:
        """Get reward weights with compatibility for different verifiers versions."""
        if hasattr(self.rubric, "get_reward_weights"):
            return self.rubric.get_reward_weights()
        elif hasattr(self.rubric, "_get_reward_weights"):
            return self.rubric._get_reward_weights()
        else:
            logger.warning("Could not find reward_weights method on rubric")
            return []

    def _call_reward_func(self, func, completion, answer, prompt=None, **kwargs):
        """Call a reward function with appropriate arguments based on its signature."""
        import inspect

        sig = inspect.signature(func)
        params = sig.parameters

        # Build kwargs based on what the function accepts
        call_kwargs = {}
        if "parser" in params:
            call_kwargs["parser"] = self.parser
        if "completion" in params:
            call_kwargs["completion"] = completion
        if "answer" in params:
            call_kwargs["answer"] = answer
        if "prompt" in params:
            call_kwargs["prompt"] = prompt

        # Add any extra kwargs the function might accept
        for key, value in kwargs.items():
            if key in params:
                call_kwargs[key] = value

        return func(**call_kwargs)

    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, List[APIServerConfig]]:
        env_config = VfEnvConfig(
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8010",
            total_steps=10,
            batch_size=4,
            steps_per_eval=1,
            max_token_length=2048,
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=4,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate percent_correct if buffer has data
        if len(self.percent_correct_buffer) > 0:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = self.vf_env.get_dataset()
        test_data = self.vf_env.get_eval_dataset()
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
        self.iter = 0

    async def rollout_and_score_eval(
        self, question: str, answer: str, **kwargs
    ) -> dict:
        state = kwargs["state"] if "state" in kwargs else None
        info = kwargs["info"] if "info" in kwargs else None
        system_prompt = kwargs["system_prompt"] if "system_prompt" in kwargs else None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
        )

        # Defensive check for completion response
        if not completion.choices:
            logger.warning("No choices returned from eval chat completion")
            return {"score": 0.0, "sample": {"error": "No completion choices"}}

        choice = completion.choices[0]
        if not choice.message or choice.message.content is None:
            logger.warning("Empty message content in eval completion")
            return {"score": 0.0, "sample": {"error": "Empty completion content"}}

        response_content = choice.message.content
        messages.append({"role": "assistant", "content": response_content})

        # PARSE HERE WITH VF PARSER
        answer_parsed = self.parser.parse_answer(completion=response_content)

        # USE REWARD FUNC HERE TO GET SCORE
        rewards = []
        for func in self.reward_funcs:
            try:
                reward = self._call_reward_func(
                    func=func,
                    completion=messages,
                    answer=answer,
                    prompt=question,
                    info=info,
                    state=state,
                )
                # Handle async functions
                if hasattr(reward, "__await__"):
                    reward = await reward
                rewards.append(reward if reward is not None else 0.0)
            except Exception as e:
                logger.warning(
                    "Reward function %s failed: %s", getattr(func, "__name__", func), e
                )
                rewards.append(0.0)

        weighted_rewards = [
            reward * self.reward_scales[i] if i < len(self.reward_scales) else reward
            for i, reward in enumerate(rewards)
        ]

        score = sum(weighted_rewards)

        sample = {
            "messages": messages,
            "question": question,
            "gold_answer": answer,
            # "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": choice.finish_reason,
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

    async def get_next_item(self) -> Item:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """Collect multiple trajectories for a single item and score them."""
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Generate multiple completions at once
        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        to_score = []
        if not chat_completions.choices:
            logger.warning("No choices returned from chat completion")
            return None, []

        for choice in chat_completions.choices:
            # Defensive check for choice content
            if not choice.message or choice.message.content is None:
                logger.warning("Empty message content in completion choice")
                continue

            response_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": choice.message.content},
            ]
            to_score.append(
                {
                    "messages": response_messages,
                    "answer": answer,
                    "finish_reason": choice.finish_reason,
                }
            )

        scored_data = await self.score(to_score)
        return scored_data, []

    async def score(
        self, rollout_group_data: List[Dict]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score a group of rollouts using the Verifiers rubric."""
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            response_content = item["messages"][-1]["content"]
            answer = item["answer"]

            # Parse the model's answer using the Verifiers parser
            _ = self.parser.parse_answer(completion=response_content)

            # Calculate rewards using the rubric's reward functions
            rewards = []
            for func in self.reward_funcs:
                try:
                    reward = self._call_reward_func(
                        func=func,
                        prompt=item["messages"][1]["content"],  # user message
                        completion=item["messages"],
                        answer=answer,
                    )
                    # Handle async functions
                    if hasattr(reward, "__await__"):
                        reward = await reward
                    rewards.append(reward if reward is not None else 0.0)
                except Exception as e:
                    logger.warning(
                        "Reward function %s failed: %s",
                        getattr(func, "__name__", func),
                        e,
                    )
                    rewards.append(0.0)

            # Calculate weighted score
            weighted_score = sum(r * w for r, w in zip(rewards, self.reward_scales))

            # Normalize messages to list for tokenizer compatibility
            messages_list = list(item["messages"])

            # Tokenize the messages for training
            out_dict = tokenize_for_trainer(
                self.tokenizer, messages_list, item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Skip examples with too few valid tokens
            if len([1 for m in masks if m != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            # Convert weighted score to reward (-1 to 1 range based on threshold)
            reward_value = (
                1.0 if weighted_score >= self.config.reward_threshold else -1.0
            )
            scores["scores"].append(reward_value)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Track percent correct in buffer for wandb logging
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Return None if all scores are the same (no learning signal)
        if len(scores["scores"]) == 0:
            return None
        if all(s == scores["scores"][0] for s in scores["scores"]):
            return None

        return scores


if __name__ == "__main__":
    VerifiersEnv.cli()
