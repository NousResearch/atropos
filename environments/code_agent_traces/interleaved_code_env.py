"""
InterleavedCodeEnv - Atropos Environment for Interleaved Code Reasoning

This environment trains models to solve coding problems using
interleaved thinking: Think → Code → Think → Code → Verify

Key features:
- Extends BaseEnv for full Atropos integration
- Uses [THINK]/[CODE]/[VERIFY] markers for structured reasoning
- Executes code locally with test verification
- Rewards based on test pass rate

Usage:
    python interleaved_code_env.py serve --config config.yaml
    python interleaved_code_env.py process --env--total_steps 100
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Local executor
from local_executor import execute_code_safe

logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS_PER_TURN = 1024
MAX_ROLLOUT_TURNS = 5
DEBUG = os.getenv("DEBUG", "0") == "1"


# ============================================================================
# System prompt for interleaved reasoning
# ============================================================================

INTERLEAVED_SYSTEM_PROMPT = """You are an expert Python programmer who thinks step-by-step while coding.

You MUST interleave your reasoning with code using these markers:

[THINK] Your reasoning about the problem or next step
[CODE]
your_python_code_here()
[/CODE]
[VERIFY] Trace through your solution with test cases

Example:
[THINK] I need a hash map to track seen numbers for O(n) lookup
[CODE]
def two_sum(nums, target):
    seen = {}
[/CODE]
[THINK] Now iterate and check for complement
[CODE]
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
[/CODE]
[VERIFY]
Test: [2,7,11,15], target=9
- i=0: num=2, comp=7, seen={} → add 2:0
- i=1: num=7, comp=2, seen={2:0} → found! return [0,1] ✓
[/VERIFY]

IMPORTANT:
- Use [THINK] before EVERY code block to explain your reasoning
- Use [THINK] WAIT when you catch a potential bug
- Use [VERIFY] at the end to trace through your solution
- Return ONLY the function definition, no test code"""


class InterleavedCodeEnvConfig(BaseEnvConfig):
    """Configuration for InterleavedCodeEnv."""

    dataset_name: str = Field(
        default="openai/openai_humaneval",
        description="HuggingFace dataset for coding problems",
    )
    dataset_split: str = Field(
        default="test",
        description="Dataset split to use",
    )
    max_problems: int = Field(
        default=500,
        description="Maximum number of problems to load",
    )
    code_timeout: float = Field(
        default=10.0,
        description="Timeout for code execution in seconds",
    )
    partial_credit: bool = Field(
        default=True,
        description="Give partial credit for partial test passes",
    )
    think_bonus: float = Field(
        default=0.1,
        description="Bonus reward for using [THINK] markers",
    )
    verify_bonus: float = Field(
        default=0.1,
        description="Bonus reward for using [VERIFY] marker",
    )


class InterleavedCodeEnv(BaseEnv):
    """
    Atropos environment for training interleaved code reasoning.

    Trains models to solve coding problems using Think → Code → Verify pattern.
    """

    name = "interleaved_code"
    env_config_cls = InterleavedCodeEnvConfig

    def __init__(
        self,
        config: InterleavedCodeEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: InterleavedCodeEnvConfig = config
        self.train_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.problem_idx = 0
        self.percent_correct_buffer: List[float] = []
        self.think_counts: List[int] = []
        self.verify_counts: List[int] = []

    @classmethod
    def config_init(cls):
        """Initialize default configuration."""
        cfg = InterleavedCodeEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=256,
            steps_per_eval=50,
            max_token_length=4096,
            inference_weight=1.0,
            wandb_name="interleaved_code",
        )
        servers = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
            )
        ]
        return cfg, servers

    async def setup(self):
        """Load the coding dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        try:
            # Try loading HumanEval or similar
            if "humaneval" in self.config.dataset_name.lower():
                ds = load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    trust_remote_code=True,
                )
                # Convert to our format
                problems = []
                for item in ds:
                    problems.append({
                        "task_id": item.get("task_id", f"task_{len(problems)}"),
                        "prompt": item.get("prompt", ""),
                        "canonical_solution": item.get("canonical_solution", ""),
                        "test": item.get("test", ""),
                        "entry_point": item.get("entry_point", "solution"),
                    })
            else:
                # Generic dataset loading
                ds = load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                )
                problems = list(ds)

            # Limit problems
            if len(problems) > self.config.max_problems:
                problems = problems[: self.config.max_problems]

            # Split into train/test
            random.shuffle(problems)
            split_idx = int(len(problems) * 0.9)
            self.train_data = Dataset.from_list(problems[:split_idx])
            self.test_data = Dataset.from_list(problems[split_idx:])

            logger.info(
                f"Loaded {len(self.train_data)} train, {len(self.test_data)} test problems"
            )

        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}. Using built-in problems.")
            self._use_builtin_problems()

    def _use_builtin_problems(self):
        """Use built-in problems if dataset loading fails."""
        problems = [
            {
                "task_id": "two_sum",
                "prompt": "def two_sum(nums: List[int], target: int) -> List[int]:\n    \"\"\"Return indices of two numbers that add up to target.\"\"\"\n",
                "test": {
                    "fn_name": "two_sum",
                    "inputs": [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6]],
                    "outputs": [[0, 1], [1, 2], [0, 1]],
                },
                "entry_point": "two_sum",
            },
            {
                "task_id": "is_palindrome",
                "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is palindrome (alphanumeric only, case insensitive).\"\"\"\n",
                "test": {
                    "fn_name": "is_palindrome",
                    "inputs": [["A man, a plan, a canal: Panama"], ["race a car"], [""]],
                    "outputs": [True, False, True],
                },
                "entry_point": "is_palindrome",
            },
            {
                "task_id": "max_subarray",
                "prompt": "def max_subarray(nums: List[int]) -> int:\n    \"\"\"Find contiguous subarray with largest sum.\"\"\"\n",
                "test": {
                    "fn_name": "max_subarray",
                    "inputs": [[[-2, 1, -3, 4, -1, 2, 1, -5, 4]], [[1]], [[-1]]],
                    "outputs": [6, 1, -1],
                },
                "entry_point": "max_subarray",
            },
            {
                "task_id": "reverse_linked_list",
                "prompt": "def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:\n    \"\"\"Reverse a singly linked list.\"\"\"\n",
                "test": {
                    "fn_name": "reverse_list",
                    "inputs": [],  # Would need special handling
                    "outputs": [],
                },
                "entry_point": "reverse_list",
            },
            {
                "task_id": "valid_parentheses",
                "prompt": "def is_valid(s: str) -> bool:\n    \"\"\"Check if parentheses string is valid.\"\"\"\n",
                "test": {
                    "fn_name": "is_valid",
                    "inputs": [["()"], ["()[]{}"], ["(]"], ["([)]"], ["{[]}"]],
                    "outputs": [True, True, False, False, True],
                },
                "entry_point": "is_valid",
            },
        ]

        random.shuffle(problems)
        split_idx = int(len(problems) * 0.8)
        self.train_data = Dataset.from_list(problems[:split_idx])
        self.test_data = Dataset.from_list(problems[split_idx:])

    async def get_next_item(self):
        """Get next problem for training."""
        if self.train_data is None:
            await self.setup()

        # Cycle through problems
        idx = self.problem_idx % len(self.train_data)
        self.problem_idx += 1

        problem = self.train_data[idx]
        return self._prepare_item(problem)

    def _prepare_item(self, problem: Dict) -> Tuple:
        """Prepare a problem as a message tuple."""
        prompt = problem.get("prompt", "")
        task_id = problem.get("task_id", "unknown")

        user_content = f"""Solve this coding problem step-by-step using [THINK], [CODE], and [VERIFY] markers:

{prompt}

Remember:
- Use [THINK] before each code block
- Use [VERIFY] to trace through your solution
- Return ONLY the function, no test code"""

        messages = [
            {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Convert to hashable tuple for caching
        messages_tuple = tuple(frozenset(m.items()) for m in messages)

        # Include test info
        test_info = problem.get("test", {})
        if isinstance(test_info, str):
            test_info = {"raw_test": test_info}

        return (messages_tuple, json.dumps({
            "task_id": task_id,
            "entry_point": problem.get("entry_point", "solution"),
            "test": test_info,
        }))

    def _parse_response(self, text: str) -> Tuple[str, int, int, bool]:
        """
        Parse interleaved response to extract code and count markers.

        Returns: (code, think_count, has_verify)
        """
        code_parts = []
        think_count = 0
        has_verify = False

        # Count [THINK] markers
        think_count = len(re.findall(r'\[THINK\]', text, re.IGNORECASE))

        # Check for [VERIFY]
        has_verify = bool(re.search(r'\[VERIFY\]', text, re.IGNORECASE))

        # Extract code from [CODE]...[/CODE] blocks
        code_pattern = r'\[CODE\](.*?)\[/CODE\]'
        matches = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            code = match.strip()
            if code:
                code_parts.append(code)

        # Fallback: try markdown code blocks
        if not code_parts:
            md_pattern = r'```python\s*(.*?)```'
            matches = re.findall(md_pattern, text, re.DOTALL)
            for match in matches:
                code = match.strip()
                if code:
                    code_parts.append(code)

        full_code = '\n'.join(code_parts)
        return full_code, think_count, has_verify

    def _execute_and_score(
        self, code: str, test_info: Dict, think_count: int, has_verify: bool
    ) -> float:
        """Execute code and calculate reward."""
        if not code.strip():
            return -1.0  # No code generated

        # Base score from test execution
        if test_info and test_info.get("fn_name"):
            try:
                results, metadata = execute_code_safe(
                    code, test_info, timeout=self.config.code_timeout
                )

                if "error" in metadata:
                    base_score = -1.0
                else:
                    passed = sum(results)
                    total = len(results)

                    if total == 0:
                        base_score = 0.0
                    elif self.config.partial_credit:
                        # Partial credit: -1 to 1 based on pass rate
                        base_score = -1.0 + (2.0 * passed / total)
                    else:
                        # Binary: all or nothing
                        base_score = 1.0 if passed == total else -1.0

            except Exception as e:
                logger.warning(f"Execution error: {e}")
                base_score = -1.0
        else:
            # No tests available, just check if code is syntactically valid
            try:
                compile(code, "<string>", "exec")
                base_score = 0.0  # Valid syntax but no tests
            except SyntaxError:
                base_score = -1.0

        # Apply bonuses for good reasoning structure
        bonus = 0.0
        if think_count >= 2:
            bonus += self.config.think_bonus
        if has_verify:
            bonus += self.config.verify_bonus

        # Clamp final score to [-1, 1]
        final_score = max(-1.0, min(1.0, base_score + bonus))

        return final_score

    async def collect_trajectory(self, item) -> Tuple[Optional[Dict], List]:
        """Collect a single trajectory for one rollout."""
        messages_tuple, expected_raw = item
        expected = json.loads(expected_raw) if isinstance(expected_raw, str) else expected_raw

        # Convert back to messages
        prompt_msgs = [dict(r) for r in messages_tuple]

        # Generate completion
        prompt_txt = self.tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False
        )

        try:
            completion = await self.server.completion(
                prompt=prompt_txt,
                max_tokens=MAX_TOKENS_PER_TURN * MAX_ROLLOUT_TURNS,
                temperature=0.7,
                n=1,
            )
            raw_response = completion.choices[0].text or ""
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return None, []

        # Parse response
        code, think_count, has_verify = self._parse_response(raw_response)

        # Get test info
        test_info = expected.get("test", {})

        # Calculate score
        score = self._execute_and_score(code, test_info, think_count, has_verify)

        # Track metrics
        self.percent_correct_buffer.append(max(0, score))
        self.think_counts.append(think_count)
        self.verify_counts.append(1 if has_verify else 0)

        # Build full context for tokenization
        assistant_msg = {"role": "assistant", "content": raw_response}
        full_ctx = prompt_msgs + [assistant_msg]

        # Tokenize
        tok = tokenize_for_trainer(self.tokenizer, full_ctx)

        if DEBUG:
            logger.info(f"Task: {expected.get('task_id')}, Score: {score:.2f}, "
                       f"Thinks: {think_count}, Verify: {has_verify}")

        return {
            "tokens": tok["tokens"],
            "masks": tok["masks"],
            "scores": score,
            "messages": full_ctx,
        }, []

    async def evaluate(self, *args, **kwargs):
        """Evaluate on test set."""
        if self.test_data is None or len(self.test_data) == 0:
            logger.warning("No test data available for evaluation")
            return

        logger.info(f"Running evaluation on {len(self.test_data)} problems")

        correct = 0
        total = 0
        total_thinks = 0
        total_verifies = 0

        for problem in self.test_data:
            item = self._prepare_item(problem)
            result, _ = await self.collect_trajectory(item)

            if result and result["scores"] > 0:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_thinks = sum(self.think_counts[-total:]) / total if total > 0 else 0
        avg_verifies = sum(self.verify_counts[-total:]) / total if total > 0 else 0

        metrics = {
            "eval/accuracy": accuracy,
            "eval/avg_think_count": avg_thinks,
            "eval/verify_rate": avg_verifies,
            "eval/num_problems": total,
        }

        logger.info(f"Eval results: accuracy={accuracy:.2%}, thinks={avg_thinks:.1f}")

        await self.wandb_log(metrics)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add custom metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
            self.percent_correct_buffer = []

        if self.think_counts:
            wandb_metrics["train/avg_think_count"] = (
                sum(self.think_counts) / len(self.think_counts)
            )
            self.think_counts = []

        if self.verify_counts:
            wandb_metrics["train/verify_rate"] = (
                sum(self.verify_counts) / len(self.verify_counts)
            )
            self.verify_counts = []

        await super().wandb_log(wandb_metrics)


# CLI entry point
if __name__ == "__main__":
    InterleavedCodeEnv.cli()
