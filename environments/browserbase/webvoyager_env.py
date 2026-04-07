from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from pydantic import Field

from atroposlib.envs.base import APIServerConfig
from atroposlib.type_definitions import Message

from .browserbase_env import BrowserbaseEnv, BrowserbaseEnvConfig

WEBVOYAGER_SYSTEM_PROMPT = """You are a browser automation agent using Browserbase and Stagehand.

Available tools:
- navigate(url): Navigate to a URL
- observe(instruction): Find likely actions matching an instruction
- act(instruction): Execute a natural-language browser action
- extract(instruction, schema_json): Extract structured page data

Use tools to complete the web task efficiently. When the task is complete, stop
calling tools and answer with a concise final summary of the result."""

WEBVOYAGER_JUDGE_PROMPT = """You are evaluating whether a browser automation agent successfully completed a web task.

Task Description:
```
{question}
```

Agent's Actions and Final State:
```
{response}
```

Based on the agent's actions and final state, evaluate whether the task was successfully completed.

Consider:
1. Did the agent navigate to the correct website/page?
2. Did the agent perform the required actions (search, filter, click, fill forms, etc.)?
3. Did the agent reach a state that satisfies the task requirements?
4. Did the agent provide the requested information if applicable?

Respond "yes" if the task was successfully completed, "no" if it was not completed or only partially completed."""


class WebVoyagerBrowserbaseEnvConfig(BrowserbaseEnvConfig):
    dataset_path: str = Field(
        default="",
        description="Path to a WebVoyager JSONL file.",
    )
    num_examples: int = Field(
        default=-1,
        description="Number of examples to load. Use -1 for all.",
    )
    web_filter: Optional[str] = Field(
        default=None,
        description="Optional website filter matching the dataset's web_name/website field.",
    )
    system_prompt: str = Field(
        default=WEBVOYAGER_SYSTEM_PROMPT,
        description="System prompt for the Browserbase agent.",
    )
    judge_model_name: str = Field(
        default="gpt-4o-mini",
        description="Model used to judge task completion.",
    )
    judge_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the judge model API.",
    )
    judge_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable containing the judge API key.",
    )
    judge_temperature: float = Field(
        default=0.0,
        description="Judge temperature.",
    )
    judge_max_tokens: int = Field(
        default=256,
        description="Max tokens for the judge response.",
    )


class WebVoyagerBrowserbaseEnv(BrowserbaseEnv):
    name = "webvoyager_browserbase"
    env_config_cls = WebVoyagerBrowserbaseEnvConfig

    def __init__(
        self,
        config: WebVoyagerBrowserbaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        self.testing = testing
        super().__init__(config, server_configs, slurm=slurm, testing=testing)
        self._tasks: List[Dict[str, Any]] = []
        self._iter = 0
        self.judge_client: Optional[openai.AsyncOpenAI] = None

        judge_key = os.getenv(config.judge_api_key_env)
        if judge_key:
            self.judge_client = openai.AsyncOpenAI(
                api_key=judge_key,
                base_url=config.judge_base_url,
            )
        elif not testing:
            raise ValueError(
                f"Missing judge API key in env var {config.judge_api_key_env}."
            )

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[WebVoyagerBrowserbaseEnvConfig, List[APIServerConfig]]:
        env_config = WebVoyagerBrowserbaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-4B",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=4,
            steps_per_eval=100,
            max_token_length=1024,
            ensure_scores_are_not_same=False,
            wandb_name="webvoyager-browserbase",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-4B",
                base_url="http://localhost:9001/v1",
                api_key="x",
                server_type="vllm",
            )
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"WebVoyager dataset not found at {dataset_path}. Set env.dataset_path."
            )
        self._tasks = self._load_webvoyager_dataset(
            dataset_path=dataset_path,
            num_examples=self.config.num_examples,
            web_filter=self.config.web_filter,
        )
        if not self._tasks:
            raise ValueError(
                "WebVoyager dataset produced no tasks. Check dataset_path, num_examples, and web_filter."
            )
        self._iter = 0

    async def get_next_item(self) -> Dict[str, Any]:
        task = self._tasks[self._iter % len(self._tasks)]
        self._iter += 1
        return task

    async def build_initial_messages(self, item: Dict[str, Any]) -> List[Message]:
        user_prompt = (
            f"Task: {item['question']}\n"
            f"Start URL: {item['start_url']}\n\n"
            "Begin by navigating to the start URL with navigate(url). Use tools as "
            "needed to complete the task. When the task is complete, stop calling tools "
            "and respond with a concise final answer that describes the final state or "
            "the requested information."
        )
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def score_episode(
        self,
        item: Dict[str, Any],
        messages: List[Message],
        runtime: Any,
        termination_reason: str,
    ) -> Tuple[float, Dict[str, Any]]:
        transcript = self.render_episode_transcript(messages)
        judge_prompt = WEBVOYAGER_JUDGE_PROMPT.format(
            question=item["question"],
            response=transcript,
        )
        judge_response = await self._judge_completion(judge_prompt)
        score = 1.0 if self._judge_says_yes(judge_response) else 0.0

        final_answer = ""
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        if assistant_messages:
            final_answer = assistant_messages[-1].get("content") or ""

        overrides = {
            "task_id": item.get("task_id"),
            "website": item.get("website"),
            "termination_reason": termination_reason,
            "judge_response": judge_response,
            "final_answer": final_answer,
            "num_messages": len(messages),
        }
        return score, overrides

    async def teardown(self) -> None:
        await super().teardown()
        if self.judge_client is not None:
            await self.judge_client.close()
            self.judge_client = None

    def _load_webvoyager_dataset(
        self,
        dataset_path: Path,
        num_examples: int,
        web_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        with dataset_path.open("r") as handle:
            for line in handle:
                item = json.loads(line)
                website = item.get("website") or item.get("web_name")
                if web_filter and website != web_filter:
                    continue
                tasks.append(
                    {
                        "question": item.get("question") or item["ques"],
                        "start_url": item.get("start_url")
                        or item.get("web")
                        or item.get("website"),
                        "task_id": item.get("task_id") or item.get("id"),
                        "website": website,
                    }
                )

        if num_examples > 0:
            tasks = tasks[:num_examples]
        return tasks

    async def _judge_completion(self, judge_prompt: str) -> str:
        if self.judge_client is None:
            raise RuntimeError("Judge client is not configured.")

        kwargs: Dict[str, Any] = {
            "model": self.config.judge_model_name,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": self.config.judge_temperature,
        }
        if self.config.judge_max_tokens > 0:
            kwargs["max_tokens"] = self.config.judge_max_tokens

        completion = await self.judge_client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def _judge_says_yes(self, judge_response: str) -> bool:
        if re.search(r"\byes\b", judge_response, flags=re.IGNORECASE):
            return True
        if re.search(r"\bno\b", judge_response, flags=re.IGNORECASE):
            return False
        return False

    def render_episode_transcript(self, messages: List[Message]) -> str:
        rendered: List[str] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue

            if role == "assistant" and message.get("tool_calls"):
                tool_calls = json.dumps(message["tool_calls"], indent=2)
                content = message.get("content") or ""
                rendered.append(f"assistant: {content}\nTool Calls:\n{tool_calls}")
                continue

            content = message.get("content")
            if isinstance(content, list):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = content or ""

            if role == "tool":
                name = message.get("name", "tool")
                rendered.append(f"tool[{name}]: {content_str}")
            else:
                rendered.append(f"{role}: {content_str}")
        return "\n\n".join(rendered)
