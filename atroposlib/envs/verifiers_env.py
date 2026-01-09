from __future__ import annotations

import asyncio
import inspect
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
from atroposlib.envs.verifiers_openai_proxy import AtroposOpenAIProxy
from atroposlib.envs.verifiers_utils import (
    infer_model_name,
    last_assistant_text,
    normalize_vf_env_id,
    reward_scales,
    sanitize_messages,
    weighted_sum,
)
from atroposlib.type_definitions import EvaluationSample, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


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
                "Failed to load verifiers environment. Ensure you've installed it via Prime, e.g.:\n"
                "  prime login\n"
                f"  prime env install {config.vf_env_name}@latest --with pip\n"
                f"Then rerun with env.vf_env_name='{config.vf_env_name}' (or just '{vf_env_id}').\n"
                f"Underlying error: {e}"
            ) from e

        print(
            "[verifiers] loaded Prime Env Hub environment: "
            f"hub_id='{config.vf_env_name}' resolved_env_id='{vf_env_id}' "
            f"env_type='{type(self.vf_env).__module__}.{type(self.vf_env).__name__}'"
        )

        self.rubric = self.vf_env.rubric
        self.parser = self.rubric.parser

        self._supports_rubric_score_group = callable(
            getattr(self.rubric, "score_group", None)
        )
        self._supports_rubric_score_rollout = callable(
            getattr(self.rubric, "score_rollout", None)
        )

        self.reward_funcs: list[Any] = []
        self.reward_weights: list[float] = []
        self.reward_scales: list[float] = []
        if not (
            self._supports_rubric_score_group or self._supports_rubric_score_rollout
        ):
            get_funcs = getattr(self.rubric, "get_reward_funcs", None) or getattr(
                self.rubric, "_get_reward_funcs", None
            )
            get_weights = getattr(self.rubric, "get_reward_weights", None) or getattr(
                self.rubric, "_get_reward_weights", None
            )
            if callable(get_funcs):
                self.reward_funcs = list(get_funcs())
            if callable(get_weights):
                self.reward_weights = list(map(float, get_weights()))
            self.reward_scales = reward_scales(self.reward_weights)

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

    async def _score_states(self, states: list[dict[str, Any]]) -> None:
        if not states:
            return

        score_sem = asyncio.Semaphore(int(getattr(self.vf_env, "max_workers", 512)))
        if self._supports_rubric_score_group:
            await self.rubric.score_group(states, score_sem=score_sem)
            return

        if self._supports_rubric_score_rollout:
            await asyncio.gather(
                *[
                    self.rubric.score_rollout(state, score_sem=score_sem)
                    for state in states
                ]
            )
            return

        if not (
            self.reward_funcs
            and callable(getattr(self.rubric, "call_reward_func", None))
        ):
            for state in states:
                state["reward"] = float(state.get("reward") or 0.0)
            return

        async def score_one(state: dict[str, Any]) -> None:
            prompt = state.get("prompt")
            completion = state.get("completion")
            answer = str(state.get("answer", ""))
            task = str(state.get("task", "default"))
            info = state.get("info") or {}
            if not isinstance(info, dict):
                info = {"info": info}

            rewards = await asyncio.gather(
                *[
                    self.rubric.call_reward_func(
                        func=func,
                        prompt=prompt,
                        completion=completion,
                        answer=answer,
                        state=state,
                        task=task,
                        info=info,
                    )
                    for func in self.reward_funcs
                ]
            )
            state["reward"] = weighted_sum(
                list(map(float, rewards)), self.reward_scales
            )

        await asyncio.gather(*[score_one(state) for state in states])

    def _score_from_state(self, state: dict[str, Any]) -> float:
        reward = state.get("reward", 0.0)
        if reward is None:
            reward = 0.0
        try:
            return float(reward)
        except Exception:
            return 0.0

    async def _rollout(
        self,
        *,
        item: Item,
        prompt: Any,
        answer: str,
        task: str,
        info: dict[str, Any],
        sampling_args: dict[str, Any],
        split: str,
    ) -> tuple[Any, dict[str, Any]]:
        client = self._train_client if split == "train" else self._eval_client

        base_kwargs: dict[str, Any] = {
            "client": client,
            "model": self._model_name,
            "prompt": prompt,
            "answer": answer,
            "task": task,
            "info": info,
            "sampling_args": sampling_args,
        }

        rollout_sig = inspect.signature(self.vf_env.rollout)
        params = list(rollout_sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]

        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        accepted_names = {
            p.name
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        kwargs = (
            base_kwargs
            if accepts_kwargs
            else {k: v for k, v in base_kwargs.items() if k in accepted_names}
        )

        input_first = bool(params) and params[0].name in {
            "input",
            "item",
            "row",
            "example",
        }

        result = (
            await self.vf_env.rollout(item, **kwargs)
            if input_first
            else await self.vf_env.rollout(**kwargs)
        )

        if isinstance(result, tuple) and len(result) == 2:
            completion, state = result
        elif isinstance(result, dict):
            state = result
            completion = state.get("completion")
        else:
            raise TypeError(
                f"Unsupported verifiers env rollout return type: {type(result)}"
            )

        if not isinstance(state, dict):
            raise TypeError(f"Rollout state must be dict, got: {type(state)}")

        return completion, state

    def _sampling_args(self, *, temperature_override: float | None) -> dict[str, Any]:
        base: dict[str, Any] = dict(getattr(self.vf_env, "sampling_args", {}) or {})
        base["n"] = 1
        base["max_tokens"] = self.config.max_token_length
        if temperature_override is not None:
            base["temperature"] = temperature_override
        return base

    def _extract_prompt_answer_task_info(
        self, item: dict[str, Any]
    ) -> tuple[Any, str, str, dict[str, Any]]:
        prompt: Any | None = None

        if item.get("prompt") is not None:
            prompt = item["prompt"]

        if prompt is None:
            question = item.get("question") or item.get("input") or item.get("problem")
            if question is None:
                raise KeyError(
                    "verifiers dataset item missing prompt keys; expected one of: "
                    "'prompt', 'question', 'input', 'problem'."
                )

            format_prompt = getattr(self.vf_env, "format_prompt", None)
            if callable(format_prompt) and isinstance(question, str):
                prompt = format_prompt(
                    question,
                    system_prompt=getattr(self.vf_env, "system_prompt", None),
                    few_shot=getattr(self.vf_env, "few_shot", None),
                )
            else:
                prompt = question

        answer = str(
            item.get("answer") or item.get("output") or item.get("response") or ""
        )
        task = str(item.get("task", "default"))
        info = item.get("info") or {}
        if not isinstance(info, dict):
            info = {"info": info}
        return prompt, answer, task, info

    async def _rollout_state(
        self,
        *,
        item: Item,
        prompt: Any,
        answer: str,
        task: str,
        info: dict[str, Any],
        sampling_args: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        completion, state = await self._rollout(
            item=item,
            prompt=prompt,
            answer=answer,
            task=task,
            info=info,
            sampling_args=sampling_args,
            split=split,
        )

        if completion is None:
            completion = state.get("completion")

        if "prompt" not in state or state.get("prompt") is None:
            state["prompt"] = prompt
        state.setdefault("answer", answer)
        state.setdefault("task", task)
        state.setdefault("info", info)
        state["completion"] = completion
        return state

    def _finish_reason_from_state(self, state: dict[str, Any]) -> str:
        try:
            trajectory = state.get("trajectory") or []
            if trajectory:
                response = trajectory[-1].get("response")
                if response is not None and getattr(response, "choices", None):
                    return str(response.choices[0].finish_reason or "")
        except Exception:
            return ""
        return ""

    def _scored_item_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        score = self._score_from_state(state)
        finish_reason = self._finish_reason_from_state(state)

        prompt = state.get("prompt")
        completion = state.get("completion")
        message_type = getattr(self.vf_env, "message_type", "chat")

        if (
            message_type == "chat"
            and isinstance(prompt, list)
            and isinstance(completion, list)
        ):
            full_messages = sanitize_messages(prompt + completion)
            tok = tokenize_for_trainer(
                self.tokenizer,
                full_messages,
                include_messages=self.config.include_messages,
                train_on_all_assistant_turns=self.config.train_on_all_assistant_turns,
                finish_reason=finish_reason,
            )
            return {
                "tokens": tok["tokens"],
                "masks": tok["masks"],
                "scores": float(score),
                "messages": tok.get("messages"),
            }

        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise TypeError(
                "Unsupported verifiers env rollout output for completion-style tokenization: "
                f"prompt={type(prompt)} completion={type(completion)}"
            )

        prompt_tokens = self.tokenizer.encode(prompt)
        if prompt_tokens and prompt_tokens[-1] == self.tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]
        full_tokens = self.tokenizer.encode(prompt + completion)
        masks = [-100 for _ in range(len(prompt_tokens))] + full_tokens[
            len(prompt_tokens) :
        ]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        return {
            "tokens": full_tokens,
            "masks": masks,
            "scores": float(score),
            "messages": messages if self.config.include_messages else None,
        }

    async def collect_trajectories(self, item: Item) -> tuple[Any, list[Item]]:
        assert isinstance(item, dict)
        prompt, answer, task, info = self._extract_prompt_answer_task_info(item)
        sampling_args = self._sampling_args(
            temperature_override=self.config.temperature
        )

        rollout_tasks = [
            self._rollout_state(
                item=item,
                prompt=prompt,
                answer=answer,
                task=task,
                info=info,
                sampling_args=sampling_args,
                split="train",
            )
            for _ in range(self.config.group_size)
        ]
        rollout_results = await asyncio.gather(*rollout_tasks, return_exceptions=True)

        backlog: list[Item] = []
        states: list[dict[str, Any]] = []
        for res in rollout_results:
            if isinstance(res, Exception):
                backlog.append(item)
            else:
                states.append(res)

        if len(states) != self.config.group_size:
            return None, backlog

        await self._score_states(states)

        items: list[dict[str, Any]] = []
        for state in states:
            try:
                items.append(self._scored_item_from_state(state))
            except Exception:
                backlog.append(item)

        if len(items) != self.config.group_size:
            return None, backlog

        group: ScoredDataGroup = {
            "tokens": [it["tokens"] for it in items],
            "masks": [it["masks"] for it in items],
            "scores": [it["scores"] for it in items],
        }
        if self.config.include_messages:
            group["messages"] = [it.get("messages") for it in items]

        return group, backlog

    async def evaluate(self, *args, **kwargs):
        if self.eval_ds is None:
            return {}

        start_time = time.time()

        eval_num = self.config.eval_num_examples
        if eval_num > 0:
            eval_ds = self.eval_ds.select(range(min(eval_num, len(self.eval_ds))))
        else:
            eval_ds = self.eval_ds

        scores: list[float] = []
        samples: list[EvaluationSample] = []

        async def eval_one(row: dict[str, Any]):
            prompt, answer, task, info = self._extract_prompt_answer_task_info(row)
            state = await self._rollout_state(
                item=row,
                prompt=prompt,
                answer=answer,
                task=task,
                info=info,
                sampling_args=self._sampling_args(
                    temperature_override=self.config.eval_temperature
                ),
                split="eval",
            )
            await self._score_states([state])
            score = self._score_from_state(state)
            completion = state.get("completion")
            prompt_for_scoring = state.get("prompt", prompt)
            finish_reason = self._finish_reason_from_state(state)

            question = row.get("question") or row.get("input") or row.get("problem")
            if question is None and isinstance(prompt_for_scoring, list):
                for msg in reversed(prompt_for_scoring):
                    if msg.get("role") == "user":
                        question = msg.get("content")
                        break

            parsed = ""
            if (
                getattr(self.vf_env, "message_type", "chat") == "chat"
                and isinstance(prompt_for_scoring, list)
                and isinstance(completion, list)
            ):
                parsed = self.parser.parse_answer(
                    completion=last_assistant_text(completion)
                )
                full_messages = sanitize_messages(prompt_for_scoring + completion)
            elif isinstance(completion, str):
                parsed = self.parser.parse_answer(completion=completion)
                full_messages = [
                    {"role": "user", "content": prompt_for_scoring},
                    {"role": "assistant", "content": completion},
                ]
            else:
                full_messages = []

            sample: EvaluationSample = {
                "messages": full_messages,
                "question": str(question) if question is not None else None,
                "gold_answer": answer,
                "model_parsed": str(parsed) if parsed is not None else None,
                "score": int(score) if float(score).is_integer() else None,
                "correct": bool(score),
                "finish_reason": finish_reason,
            }

            return float(score), sample

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for row in eval_ds:
            queue.put_nowait(row)

        async def worker():
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                s, sample = await eval_one(row)
                scores.append(s)
                samples.append(sample)
                queue.task_done()

        max_workers = max(1, int(self.config.max_eval_workers))
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(max_workers, queue.qsize()))
        ]
        await asyncio.gather(*workers)

        avg_total_score = (sum(scores) / len(scores)) if scores else 0.0
        eval_metrics = {"eval/avg_total_score": avg_total_score}

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
