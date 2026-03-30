"""
Code Debug Environment for Atropos

Trains LLMs to debug and fix buggy Python functions.
Uses the HumanEvalPack dataset (HumanEvalFix subset) with execution-based verification
against ground-truth test cases.

Environment pattern follows sql_query_env for consistency.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from code_executor import (
    count_test_results,
    execute_code_with_tests,
    extract_boxed_code,
)
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

logger = logging.getLogger(__name__)

# System prompt following established Atropos patterns
SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

SYSTEM_PROMPT += """You are an expert Python debugger. Given a buggy Python function \
and its test cases, identify the bug and provide the corrected function.

You are allocated a maximum of 2048 tokens, please strive to use less.

Provide your corrected function inside \\boxed{} like this:
\\boxed{def function_name(args):
    # your corrected code here
    return result
}

Important:
- Keep the same function signature
- Only fix the bug, don't rewrite the function from scratch unless necessary
- Ensure the function passes all provided test cases
- Do not include test cases or the check function in your answer

End your answer with \\boxed{your corrected function here}"""


class CodeDebugItem(TypedDict):
    """Type definition for a HumanEvalFix dataset item."""

    task_id: str
    prompt: str
    buggy_solution: str
    canonical_solution: str
    test: str
    entry_point: str


def format_debug_prompt(item: CodeDebugItem) -> str:
    """Format the buggy code and context into a prompt for the model."""
    buggy_code = item["prompt"] + item["buggy_solution"]
    # Show test structure without revealing the exact assertions
    return (
        f"Here is a buggy Python function:\n\n"
        f"```python\n{buggy_code}```\n\n"
        f"The function `{item['entry_point']}` has a bug. "
        f"It fails its test cases.\n\n"
        f"Please identify the bug, fix it, and provide the corrected "
        f"function inside \\boxed{{}}."
    )


class CodeDebugEnv(BaseEnv):
    """
    Environment for training LLMs to debug Python code.

    Uses the HumanEvalFix dataset. The model receives a buggy function
    and must output the corrected version. Scoring is done by executing
    the fixed code against the original test suite.
    """

    name = "code_debug"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.partial_fix_buffer = list()
        self.raw_score_buffer = list()

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="code_debug",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log custom metrics to WandB."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log percent fully correct (all tests pass)
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        # Log average raw score (includes partial credit)
        if self.raw_score_buffer:
            wandb_metrics["train/avg_score"] = sum(self.raw_score_buffer) / len(
                self.raw_score_buffer
            )

        # Log partial fix rate (code runs but doesn't pass all tests)
        if self.partial_fix_buffer:
            wandb_metrics["train/partial_fix_rate"] = sum(
                self.partial_fix_buffer
            ) / len(self.partial_fix_buffer)

        self.percent_correct_buffer = list()
        self.raw_score_buffer = list()
        self.partial_fix_buffer = list()

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Load the HumanEvalPack dataset (HumanEvalFix) and prepare train/test splits."""
        from datasets import load_dataset

        logger.info("Loading HumanEvalPack (python) dataset...")
        dataset = load_dataset("bigcode/humanevalpack", "python", split="test")

        all_items: List[CodeDebugItem] = []
        for row in dataset:
            all_items.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["declaration"],
                    "buggy_solution": row["buggy_solution"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
            )

        logger.info("Loaded %d problems", len(all_items))

        # Verify a few items actually work with canonical solutions
        verified = 0
        for item in all_items[:10]:
            code = item["prompt"] + item["canonical_solution"]
            passed, _ = execute_code_with_tests(code, item["test"], item["entry_point"])
            if passed:
                verified += 1
        logger.info("Verified %d/10 canonical solutions execute correctly", verified)

        # Split 80/20 train/test
        random.shuffle(all_items)
        split_idx = int(len(all_items) * 0.8)
        self.train = all_items[:split_idx]
        self.test = all_items[split_idx:]

        logger.info("Train: %d, Test: %d", len(self.train), len(self.test))
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _score_fix(
        self, generated_code: str, item: CodeDebugItem
    ) -> Tuple[float, bool]:
        """
        Score a generated fix by execution against test cases.

        Returns:
            Tuple of (score, is_partial_fix):
            - score: 1.0 if all tests pass, proportional for partial,
                     -1.0 if no improvement or compilation error
            - is_partial_fix: True if some but not all tests pass
        """
        if not generated_code:
            return -1.0, False

        # Run the fixed code against tests
        all_passed, error = execute_code_with_tests(
            generated_code, item["test"], item["entry_point"]
        )

        if all_passed:
            return 1.0, False

        # Check for partial credit - how many tests pass?
        passed, total = count_test_results(
            generated_code, item["test"], item["entry_point"]
        )

        if total == 0:
            return -1.0, False

        # Also check how the buggy code does
        buggy_code = item["prompt"] + item["buggy_solution"]
        buggy_passed, buggy_total = count_test_results(
            buggy_code, item["test"], item["entry_point"]
        )

        # Score based on improvement over buggy code
        if passed > buggy_passed:
            # Partial improvement: scale between -0.5 and 0.9
            improvement_ratio = (passed - buggy_passed) / max(1, total - buggy_passed)
            score = -0.5 + 1.4 * improvement_ratio
            return score, True
        elif passed == buggy_passed and passed > 0:
            # No improvement but code at least runs
            return -0.5, True
        else:
            # Made things worse or code doesn't compile
            return -1.0, False

    async def rollout_and_score_eval(self, item: CodeDebugItem) -> dict:
        """Rollout and score a single evaluation item."""
        user_content = format_debug_prompt(item)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.6,
            )
            response_content = completion.choices[0].message.content

        # Extract and score generated fix
        generated_code = extract_boxed_code(response_content)
        score, is_partial = self._score_fix(generated_code, item)
        correct = score == 1.0

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_content},
            ],
            "question": f"Fix bug in {item['entry_point']}",
            "gold_answer": item["canonical_solution"],
            "model_parsed": generated_code or "(no code extracted)",
            "score": 1 if correct else 0,
            "correct": correct,
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": 1 if correct else 0, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set."""
        import time

        start_time = time.time()

        eval_tasks = []
        for item in self.test[:100]:
            eval_tasks.append(self.rollout_and_score_eval(item))
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores) if scores else 0

        end_time = time.time()

        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.6,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(
        self, item: CodeDebugItem
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        """Generate code fixes for a buggy function."""
        user_content = format_debug_prompt(item)
        user_message = {"role": "user", "content": user_content}

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    user_message,
                ],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "item": item,
                    "finish_reason": chat_completion.finish_reason,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score generated code fixes by execution against test cases."""
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()

        item = rollout_group_data[0]["item"]

        random.shuffle(rollout_group_data)

        for rollout in rollout_group_data:
            response_content = rollout["messages"][-1]["content"]

            # Extract fixed code from \boxed{}
            generated_code = extract_boxed_code(response_content)

            # Score by execution
            reward, is_partial = self._score_fix(generated_code, item)
            self.partial_fix_buffer.append(1 if is_partial else 0)

            tokens = rollout["tokens"]
            masks = rollout["masks"]
            logprobs = rollout["logprobs"]

            # Remove obviously bad examples (too short responses)
            if len([1 for m in masks if m != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["scores"]:
            return None

        for score in scores["scores"]:
            self.percent_correct_buffer.append(1.0 if score >= 1.0 else 0.0)
            self.raw_score_buffer.append(max(score, 0.0))

        # Apply length penalty when all scores are perfect
        if all(score == 1.0 for score in scores["scores"]):
            token_lengths = [len(t) for t in scores["tokens"]]
            if max(token_lengths) == 0:
                return None

            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5

            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    scores["scores"].append(1.0)
                else:
                    pct = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    pct = min(pct, 1.0)
                    scores["scores"].append(1.0 - pct)

        # If all scores are same, add small noise to break ties
        # This prevents infinite retry loops in process mode while
        # maintaining meaningful relative ordering for training
        if all(scores["scores"][0] == s for s in scores["scores"]):
            scores["scores"] = [
                s + random.uniform(-0.01, 0.01) for s in scores["scores"]
            ]

        return scores

    async def get_next_item(self) -> CodeDebugItem:
        """Get the next training item."""
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    CodeDebugEnv.cli()
