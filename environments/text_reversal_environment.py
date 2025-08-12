import json
import random
import re
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Thinking-enabled system prompt
thinking_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class TextReversalEnvConfig(BaseEnvConfig):
    """Config for TextReversalEnv.

    - use_thinking: if True, we prepend a thinking system message; otherwise, no system message is used
      and the model should answer directly without thinking.
    - dataset_name: allows overriding the dataset, defaults to PrimeIntellect/Reverse-Text-SFT
    - eval_dataset_name: if provided, use this dataset as a static eval set; otherwise sample from train
    - test_set_size: number of held-out examples for eval (used only when eval_dataset_name is None)
    - max_train_token_length: max tokens for training generation
    - max_eval_token_length: max tokens for eval generation
    """

    # Whether to include the thinking system prompt
    use_thinking: bool = False

    # Dataset name and splitting behavior
    dataset_name: str = "PrimeIntellect/Reverse-Text-SFT"
    eval_dataset_name: Optional[str] = None
    test_set_size: int = 100

    # Separate max token lengths for train and eval
    max_train_token_length: int = 1024 * 16
    max_eval_token_length: int = 1024 * 32


class TextReversalEnv(BaseEnv):
    env_config_cls = TextReversalEnvConfig

    def __init__(
        self,
        config: TextReversalEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List[List[Tuple[str, float, str]]] = []

        # Internal dataset storage after processing
        self.train = None
        self.test = None
        self.iter = 0

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[TextReversalEnvConfig, List[APIServerConfig]]:
        env_config = TextReversalEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers_per_node=16,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="text_reversal_env",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            # Env-specific
            use_thinking=False,
            dataset_name="PrimeIntellect/Reverse-Text-SFT",
            test_set_size=100,
            eval_dataset_name=None,
            max_train_token_length=1024 * 16,
            max_eval_token_length=1024 * 32,
        )

        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]

        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics: Dict):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "expected_output"])
            for group in self.rollouts_for_wandb:
                for text, score, expected in group:
                    table.add_data(text, score, expected)
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / max(1, len(self.percent_correct_buffer))
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = []
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Load `PrimeIntellect/Reverse-Text-SFT` and produce train/test splits.

        Each dataset entry is expected to contain:
          - prompt: list of messages including a system and a user message
          - completion: list with a single assistant message containing the reversed text

        We preprocess into a list of dicts with keys:
          - system_content
          - user_content
          - expected_assistant
        """
        dataset_name = getattr(
            self.config, "dataset_name", "PrimeIntellect/Reverse-Text-SFT"
        )
        eval_dataset_name = getattr(self.config, "eval_dataset_name", None)
        try:
            full_dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

        # Convert to plain list of dicts for simple handling
        processed_items: List[Dict[str, str]] = []
        for idx, row in enumerate(full_dataset):
            try:
                system_text, user_text, expected_text = self._extract_fields(row)
                if not user_text or not expected_text:
                    continue
                processed_items.append(
                    {
                        "system_content": system_text or "",
                        "user_content": user_text,
                        "expected_assistant": expected_text,
                    }
                )
            except Exception:
                # Skip malformed entries
                continue

        if len(processed_items) == 0:
            raise RuntimeError(
                "No valid items parsed from dataset. Verify dataset schema for PrimeIntellect/Reverse-Text-SFT."
            )

        random.Random(42).shuffle(processed_items)

        # Build train and test according to presence of a static eval dataset
        if eval_dataset_name:
            # Use full processed_items as train
            self.train = processed_items

            # Load eval dataset separately
            try:
                eval_raw = load_dataset(eval_dataset_name, split="train")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load eval dataset '{eval_dataset_name}': {e}"
                )

            eval_items: List[Dict[str, str]] = []
            for row in eval_raw:
                try:
                    sys_t, usr_t, exp_t = self._extract_fields(row)
                    if not usr_t or not exp_t:
                        continue
                    eval_items.append(
                        {
                            "system_content": sys_t or "",
                            "user_content": usr_t,
                            "expected_assistant": exp_t,
                        }
                    )
                except Exception:
                    continue

            if not eval_items:
                raise RuntimeError(
                    f"No valid items parsed from eval dataset '{eval_dataset_name}'."
                )
            self.test = eval_items
        else:
            # Sample a fixed-size eval set from the primary dataset
            test_size = min(self.config.test_set_size, max(1, len(processed_items)))
            if len(processed_items) <= 1:
                self.train = processed_items
                self.test = processed_items
            else:
                self.test = processed_items[:test_size]
                self.train = processed_items[test_size:]

        self.iter = 0

    def _extract_fields(
        self, row: Dict
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract (system_text, user_text, expected_text) from a raw dataset row.

        Expected forms:
          - row["prompt"]: list of {role, content} dicts including system and user
          - row["completion"]: list with one assistant dict {role: "assistant", content: ...}

        We tolerate dict or list forms for both for robustness.
        """

        system_text: Optional[str] = None
        user_text: Optional[str] = None
        expected_text: Optional[str] = None

        prompt_field = row.get("prompt")
        if isinstance(prompt_field, list):
            for m in prompt_field:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content", "")
                if role == "system" and system_text is None:
                    system_text = content
                elif role == "user" and user_text is None:
                    user_text = content
        elif isinstance(prompt_field, dict):
            # Some data might represent a single message for system, with user elsewhere
            if prompt_field.get("role") == "system":
                system_text = prompt_field.get("content", "")

        # Expected completion
        completion_field = row.get("completion")
        if isinstance(completion_field, list):
            for m in completion_field:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    expected_text = m.get("content", "")
                    break
        elif isinstance(completion_field, dict):
            if completion_field.get("role") == "assistant":
                expected_text = completion_field.get("content", "")

        return system_text, user_text, expected_text

    def _build_messages(self, system_text: str, user_text: str) -> List[Dict[str, str]]:
        """Construct chat messages according to the requirement:
        - If thinking is enabled, include a system message with thinking instructions.
        - Do NOT use dataset system message as system; instead, prepend it to the user
          content followed by two newlines, then the dataset user content.
        """
        messages: List[Dict[str, str]] = []
        if getattr(self.config, "use_thinking", False):
            messages.append({"role": "system", "content": thinking_system_prompt})

        combined_user_content = (system_text or "").strip()
        if combined_user_content:
            combined_user_content = f"{combined_user_content}\n\n{user_text}"
        else:
            combined_user_content = user_text

        messages.append({"role": "user", "content": combined_user_content})
        return messages

    @staticmethod
    def _strip_think_and_trailing(text: str) -> str:
        """Return content after the first closing </think> tag if present; otherwise the full text.
        Trims surrounding whitespace.
        """
        if not text:
            return ""
        close_match = re.search(r"</think>", text, flags=re.IGNORECASE)
        if close_match:
            return text[close_match.end() :].strip()
        return text.strip()

    async def rollout_and_score_eval(self, test_item: Dict) -> float:
        messages = self._build_messages(
            test_item.get("system_content", ""), test_item.get("user_content", "")
        )

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=getattr(
                self.config, "max_eval_token_length", self.config.max_token_length
            ),
            temperature=0.2,
            split="eval",
        )

        model_text = completion.choices[0].text
        model_answer = self._strip_think_and_trailing(model_text)
        expected = (test_item.get("expected_assistant", "") or "").strip()
        return 1.0 if model_answer == expected else 0.0

    async def evaluate(self, *args, **kwargs):
        if not self.test or len(self.test) == 0:
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        eval_tasks = [self.rollout_and_score_eval(item) for item in self.test]
        scores = await tqdm_asyncio.gather(*eval_tasks)
        percent_correct = sum(scores) / len(scores) if scores else 0.0
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        # item: (prompt_messages_tuple, expected_text_string)
        prompt_messages_list = [dict(m) for m in item[0]]

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages_list, add_generation_prompt=True, tokenize=False
        )

        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=getattr(
                self.config, "max_train_token_length", self.config.max_token_length
            ),
            temperature=0.8,
        )

        to_score: List[Tuple[Tuple[Dict, ...], str]] = []
        for choice in completions.choices:
            trajectory_messages = [dict(m) for m in item[0]]
            trajectory_messages.append({"role": "assistant", "content": choice.text})
            to_score.append((tuple(trajectory_messages), item[1]))

        scored = await self.score(to_score)
        return scored, []

    async def score(
        self, rollout_group_data: List[Tuple[Tuple[Dict, ...], str]]
    ) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        if not rollout_group_data:
            return None

        expected_text = rollout_group_data[0][1]
        if expected_text is None:
            return None

        random.shuffle(rollout_group_data)

        for trajectory, expected in rollout_group_data:
            model_response = trajectory[-1]["content"]
            model_answer = self._strip_think_and_trailing(model_response)
            reward = 1.0 if (model_answer.strip() == (expected or "").strip()) else 0.0

            out = tokenize_for_trainer(self.tokenizer, [dict(m) for m in trajectory])
            tokens = out["tokens"]
            masks = out["masks"]

            if sum(1 for m in masks if m != -100) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success metrics
        for r in scores["scores"]:
            self.percent_correct_buffer.append(1.0 if r >= 1.0 else 0.0)

        if not scores["tokens"]:
            return None

        # If all scores are identical, skip to keep learning signal
        if len(set(scores["scores"])) <= 1 and len(scores["scores"]) > 1:
            return None

        return scores

    async def get_next_item(self) -> Item:
        if not self.train:
            raise RuntimeError("Training data not initialized")

        item = self.train[self.iter % len(self.train)]
        self.iter += 1

        messages = self._build_messages(
            item.get("system_content", ""), item.get("user_content", "")
        )

        prompt_tuple = tuple(frozenset(m.items()) for m in messages)
        answer_text = item.get("expected_assistant", "")
        return (prompt_tuple, answer_text)

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,
        item: Item = None,
    ):
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = len(scored_data["tokens"])

        expected_text = item[1] if item else ""
        group_rows = []
        for i in range(min(num_keep, len(scored_data["tokens"]))):
            decoded = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=False
            )
            score_val = scored_data["scores"][i]
            group_rows.append((decoded, score_val, expected_text))

        self.rollouts_for_wandb.append(group_rows)
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)


if __name__ == "__main__":
    TextReversalEnv.cli()
