from __future__ import annotations

import os
import time
from typing import Any, Tuple, Union

import verifiers as vf
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.server_baseline import ServerBaseline
from atroposlib.type_definitions import EvaluationSample, Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


def normalize_vf_env_id(env_id: str) -> str:
    env_id = (env_id or "").strip()
    if not env_id:
        raise ValueError(
            "env.vf_env_name must be set (Prime Env Hub id: 'owner/environment-name@version')."
        )
    env_id = env_id.split("@", 1)[0].strip()
    if "/" in env_id:
        env_id = env_id.rsplit("/", 1)[-1].strip()
    if not env_id:
        raise ValueError(
            "env.vf_env_name must contain an environment name like 'owner/environment-name'."
        )
    return env_id


def sanitize_messages(messages: list[dict[str, Any]]) -> list[Message]:
    sanitized: list[Message] = []
    for msg in messages:
        role = msg.get("role")
        if role == "developer":
            role = "system"
        if role == "agent":
            role = "assistant"
        if role not in ("system", "user", "assistant", "tool"):
            continue
        sanitized.append({"role": role, "content": msg.get("content", "")})
    return sanitized


def infer_model_name(
    server_configs: Union[ServerBaseline, list[APIServerConfig], APIServerConfig],
) -> str:
    if isinstance(server_configs, list) and server_configs:
        return server_configs[0].model_name
    if isinstance(server_configs, APIServerConfig):
        return server_configs.model_name
    return getattr(server_configs, "model_name", "model")


class _ChatCompletionsProxy:
    def __init__(self, server: Any, split: str):
        self._server = server
        self._split = split

    async def create(self, *args, **kwargs):
        if args:
            raise TypeError("Only keyword arguments are supported.")
        messages = kwargs.pop("messages", None)
        if messages is None:
            raise TypeError("Missing required kwarg: messages")
        kwargs.pop("model", None)
        if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return await self._server.chat_completion(
            messages=messages,
            split=self._split,
            **kwargs,
        )


class _CompletionsProxy:
    def __init__(self, server: Any, split: str):
        self._server = server
        self._split = split

    async def create(self, *args, **kwargs):
        if args:
            raise TypeError("Only keyword arguments are supported.")
        prompt = kwargs.pop("prompt", None)
        if prompt is None:
            raise TypeError("Missing required kwarg: prompt")
        kwargs.pop("model", None)
        return await self._server.completion(
            prompt=prompt,
            split=self._split,
            **kwargs,
        )


class _ChatProxy:
    def __init__(self, server: Any, split: str):
        self.completions = _ChatCompletionsProxy(server, split)


class AtroposOpenAIProxy:
    def __init__(self, server: Any, split: str):
        self.chat = _ChatProxy(server, split)
        self.completions = _CompletionsProxy(server, split)


class VfEnvConfig(BaseEnvConfig):
    vf_env_name: str = Field(
        default="",
        description=(
            "Prime Env Hub environment id. Accepts 'owner/environment-name@version' (Prime docs) "
            "or an installed verifiers env id like 'environment-name'."
        ),
    )
    env_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs forwarded to verifiers.load_environment(env_id, **env_args).",
    )
    train_on_all_assistant_turns: bool = Field(
        default=True,
        description=(
            "If True, unmask all assistant/agent turns (recommended for multi-turn verifiers envs)."
        ),
    )
    temperature: float | None = Field(
        default=None,
        description="Optional override for sampling temperature during training rollouts.",
    )
    eval_temperature: float | None = Field(
        default=0.0,
        description="Optional override for sampling temperature during evaluate().",
    )
    eval_num_examples: int = Field(
        default=128,
        description="Number of eval examples to run per evaluate() call (-1 = full eval dataset).",
    )


class VerifiersEnv(BaseEnv):
    name = "verifiers"
    env_config_cls = VfEnvConfig

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: Union[ServerBaseline, list[APIServerConfig], APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        if self.config.group_size <= 1 and self.config.ensure_scores_are_not_same:
            self.config.ensure_scores_are_not_same = False

        self.eval_metrics: list[tuple[str, float]] = []

        vf_env_id = normalize_vf_env_id(config.vf_env_name)
        try:
            self.vf_env = vf.load_environment(vf_env_id, **config.env_args)
        except Exception as e:
            raise RuntimeError(
                "Failed to load verifiers environment (Prime Env Hub). Install it first, e.g.:\n"
                f"  prime env install {config.vf_env_name}@latest --with pip\n"
                f"Underlying error: {e}"
            ) from e

        print(
            "[verifiers] loaded Prime Env Hub environment: "
            f"hub_id='{config.vf_env_name}' resolved_env_id='{vf_env_id}' "
            f"env_type='{type(self.vf_env).__module__}.{type(self.vf_env).__name__}'"
        )

        self._train_client = AtroposOpenAIProxy(server=self.server, split="train")
        self._eval_client = AtroposOpenAIProxy(server=self.server, split="eval")
        self._model_name = infer_model_name(server_configs)

        self.iter = 0
        self.train = None
        self.eval_ds = None

    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, list[APIServerConfig]]:
        env_config = VfEnvConfig(
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=16,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        self.train = self.vf_env.get_dataset()
        self.eval_ds = self.vf_env.get_eval_dataset()
        self.iter = 0

    def _sampling_args(self, *, temperature_override: float | None) -> dict[str, Any]:
        base: dict[str, Any] = dict(getattr(self.vf_env, "sampling_args", {}) or {})
        base["n"] = 1
        base["max_tokens"] = self.config.max_token_length
        if temperature_override is not None:
            base["temperature"] = temperature_override
        return base

    async def collect_trajectories(self, item: Item) -> tuple[Any, list[Item]]:
        assert isinstance(item, dict)

        prompt = item.get("prompt")
        if prompt is None:
            raise KeyError(
                "verifiers dataset items are expected to have a 'prompt' column; got keys: "
                f"{sorted(item.keys())}"
            )

        info = item.get("info") or {}
        if not isinstance(info, dict):
            info = {"info": info}

        rollout_input: dict[str, Any] = {
            "prompt": prompt,
            "answer": str(item.get("answer", "")),
            "task": str(item.get("task", "default")),
            "info": info,
            "example_id": int(item.get("example_id", 0) or 0),
        }

        inputs = [dict(rollout_input) for _ in range(self.config.group_size)]
        sampling_args = self._sampling_args(
            temperature_override=self.config.temperature
        )

        outputs = await self.vf_env.generate(
            inputs,
            client=self._train_client,
            model=self._model_name,
            sampling_args=sampling_args,
            use_tqdm=False,
        )

        tokens_list: list[list[int]] = []
        masks_list: list[list[int]] = []
        scores_list: list[float] = []
        messages_list: list[list[Message] | None] = []

        for state, p, c, reward in zip(
            outputs["state"],
            outputs["prompt"],
            outputs["completion"],
            outputs["reward"],
        ):
            score = float(reward or 0.0)
            finish_reason = "length" if bool(state.get("is_truncated")) else ""

            if isinstance(p, list) and isinstance(c, list):
                full_messages = sanitize_messages(p + c)
                tok = tokenize_for_trainer(
                    self.tokenizer,
                    full_messages,
                    include_messages=self.config.include_messages,
                    train_on_all_assistant_turns=self.config.train_on_all_assistant_turns,
                    finish_reason=finish_reason,
                )
                tokens_list.append(tok["tokens"])
                masks_list.append(tok["masks"])
                messages_list.append(tok.get("messages"))
            elif isinstance(p, str) and isinstance(c, str):
                prompt_tokens = self.tokenizer.encode(p)
                if prompt_tokens and prompt_tokens[-1] == self.tokenizer.eos_token_id:
                    prompt_tokens = prompt_tokens[:-1]
                full_tokens = self.tokenizer.encode(p + c)
                masks = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens) :]
                tokens_list.append(full_tokens)
                masks_list.append(masks)
                if self.config.include_messages:
                    messages_list.append(
                        [
                            {"role": "user", "content": p},
                            {"role": "assistant", "content": c},
                        ]
                    )
                else:
                    messages_list.append(None)
            else:
                return None, [item]

            scores_list.append(score)

        group: ScoredDataGroup = {
            "tokens": tokens_list,
            "masks": masks_list,
            "scores": scores_list,
        }
        if self.config.include_messages:
            group["messages"] = messages_list

        return group, []

    async def evaluate(self, *args, **kwargs):
        if self.eval_ds is None:
            return {}

        start_time = time.time()
        sampling_args = self._sampling_args(
            temperature_override=self.config.eval_temperature
        )

        outputs = await self.vf_env.evaluate(
            client=self._eval_client,
            model=self._model_name,
            sampling_args=sampling_args,
            num_examples=self.config.eval_num_examples,
            max_concurrent=max(1, int(self.config.max_eval_workers)),
            max_concurrent_generation=max(1, int(self.config.max_eval_workers)),
            max_concurrent_scoring=max(1, int(self.config.max_eval_workers)),
            use_tqdm=False,
        )

        rewards = [float(r or 0.0) for r in outputs["reward"]]
        avg_total_score = (sum(rewards) / len(rewards)) if rewards else 0.0
        eval_metrics = {"eval/avg_total_score": avg_total_score}

        samples: list[EvaluationSample] = []
        for p, c, a, r, state in zip(
            outputs["prompt"],
            outputs["completion"],
            outputs["answer"],
            outputs["reward"],
            outputs["state"],
        ):
            finish_reason = "length" if bool(state.get("is_truncated")) else None
            if isinstance(p, list) and isinstance(c, list):
                msgs: list[dict[str, Any]] = sanitize_messages(p + c)
            elif isinstance(p, str) and isinstance(c, str):
                msgs = [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c},
                ]
            else:
                msgs = []

            samples.append(
                {
                    "messages": msgs,
                    "gold_answer": str(a),
                    "score": int(r) if float(r or 0.0).is_integer() else None,
                    "correct": bool(r),
                    "finish_reason": finish_reason,
                }
            )
            if len(samples) >= 20:
                break

        end_time = time.time()
        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.max_token_length,
            },
        )
        return eval_metrics

    async def get_next_item(self) -> Item:
        assert self.train is not None
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item
