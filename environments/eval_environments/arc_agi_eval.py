"""
ARC-AGI 2 Evaluation Environment for Atropos

This environment evaluates models on the ARC-AGI 2 benchmark - testing
abstract reasoning and pattern recognition with grid-based visual puzzles.

Dataset: arc-agi-community/arc-agi-2
Paper: https://arcprize.org/guide

ARC-AGI 2 tests:
- Abstract reasoning
- Pattern recognition and transformation
- Visual/spatial reasoning
- Few-shot learning from examples
- Pixel-perfect grid output

The model is shown training examples (input → output grid transformations)
and must apply the learned pattern to a test input to produce the correct output grid.

Metrics:
- Accuracy (pixel-perfect match of output grid)

Supports optional thinking mode with <think></think> tags.
Answer must be provided in <answer></answer> tags as a JSON 2D array.
"""

import ast
import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    ANSWER_TAG_PATTERN,
    create_system_content,
    get_default_thinking_prompt,
    save_eval_results,
    validate_thinking_format,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
)


class ARCAGIEvalConfig(BaseEnvConfig):
    """Configuration for ARC-AGI 2 evaluation environment."""

    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=True,
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    # Dataset configuration
    dataset_name: str = Field(
        default="arc-agi-community/arc-agi-2",
        description="HuggingFace dataset name for ARC-AGI 2.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation (train or test).",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for model generation.",
    )

    eval_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for evaluation responses. Set to 0 for provider default.",
    )

    # Prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to append after thinking prompt.",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retries for failed API calls.",
    )

    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retry attempts in seconds.",
    )

    min_response_length: int = Field(
        default=1,
        ge=1,
        description="Minimum response length to consider valid.",
    )

    # Debug configuration
    full_debug: bool = Field(
        default=False,
        description="Enable verbose debug logging.",
    )


class ARCAGIEvalEnv(BaseEnv):
    """
    ARC-AGI 2 Evaluation Environment for Atropos.

    Evaluates models on abstract reasoning with grid-based pattern puzzles.
    """

    name = "arc_agi_eval"
    env_config_cls = ARCAGIEvalConfig

    def __init__(
        self,
        config: ARCAGIEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: ARCAGIEvalConfig = config
        self.eval_metrics = []

    @classmethod
    def config_init(cls) -> Tuple[ARCAGIEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for CLI usage."""
        config = ARCAGIEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=4096,
            wandb_name="arc_agi_eval",
            data_path_to_save_groups=None,
            eval_max_tokens=0,
        )
        server_config = APIServerConfig(
            model_name="Hermes-3-Llama-3.1-8B",
            base_url="http://localhost:8000/v1",
            api_key="x",
            num_requests_for_eval=256,  # Fewer concurrent requests due to longer responses
        )
        return config, [server_config]

    async def setup(self):
        """Load the ARC-AGI 2 dataset."""
        print("\nARC-AGI 2 Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(
                f"  Thinking prompt: {get_default_thinking_prompt(self.config.custom_thinking_prompt)[:80]}..."
            )

        # Load dataset
        self.dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.eval_split,
            trust_remote_code=True,
        )

        self.eval_items = list(self.dataset)
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert a 2D grid to a multi-line JSON string."""
        lines = []
        for row in grid:
            lines.append(json.dumps(row))
        return "\n".join(lines)

    def _format_prompt(self, item: Dict) -> Tuple[str, List[List[int]]]:
        """
        Format an ARC-AGI 2 item into a prompt.

        Returns the formatted prompt and the gold answer grid.
        """
        # Build training examples
        training_pairs = item["fewshots"]
        training_examples = ""

        for i, pair in enumerate(training_pairs):
            training_examples += f"--Example {i + 1}--\n\n"
            training_examples += "INPUT:\n"
            training_examples += self._grid_to_string(pair["input"]) + "\n\n"
            training_examples += "OUTPUT:\n"
            training_examples += self._grid_to_string(pair["output"]) + "\n\n"

        # Test input
        test_input = self._grid_to_string(item["question"][0]["input"])
        gold_output = item["question"][0]["output"]

        # Build the prompt
        query = """You are solving an ARC-AGI puzzle. You will be shown training examples
where an input grid is transformed into an output grid following a specific pattern or rule.

Your task is to:
1. Analyze the training examples to understand the transformation pattern
2. Apply that same pattern to the test input
3. Produce the correct output grid

Each grid is a 2D array of integers from 0-9, where each number represents a different color.

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Analyze the pattern in the training examples, then apply it to the test input.

IMPORTANT: Provide your final answer as a JSON 2D array inside <answer></answer> tags.
The answer should contain ONLY the JSON array, nothing else.

Example format:
<answer>
[[0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]]
</answer>
""".format(
            training_examples=training_examples, test_input=test_input
        )

        return query, gold_output

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode configuration."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt,
        )

    def _is_valid_grid(self, grid: Any) -> bool:
        """Check if grid is a valid 2D list of integers 0-9."""
        if not isinstance(grid, list) or len(grid) == 0:
            return False
        if not all(isinstance(row, list) for row in grid):
            return False
        # Check all rows have same length
        row_len = len(grid[0])
        if row_len == 0:
            return False
        for row in grid:
            if len(row) != row_len:
                return False
            if not all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row):
                return False
        return True

    def _parse_grid_from_string(self, text: str) -> Optional[List[List[int]]]:
        """
        Parse a 2D grid from a string.

        Tries multiple parsing strategies:
        1. Direct JSON parse of the whole text
        2. ast.literal_eval (handles Python list syntax)
        3. Extract rows line by line
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strategy 1: Direct JSON parse
        try:
            grid = json.loads(text)
            if self._is_valid_grid(grid):
                return grid
        except json.JSONDecodeError:
            pass

        # Strategy 2: ast.literal_eval (handles Python repr format)
        try:
            grid = ast.literal_eval(text)
            if self._is_valid_grid(grid):
                return grid
        except (ValueError, SyntaxError):
            pass

        # Strategy 3: Find the nested array pattern
        # Look for [[...], [...], ...]
        nested_pattern = r"\[\s*\[[\d,\s\[\]]+\]\s*\]"
        matches = re.findall(nested_pattern, text, re.DOTALL)

        for match in matches:
            try:
                grid = ast.literal_eval(match)
                if self._is_valid_grid(grid):
                    return grid
            except Exception:
                continue

        # Strategy 4: Extract rows one per line
        # Look for lines like [0, 1, 2, 3]
        row_pattern = r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]"
        rows = re.findall(row_pattern, text)

        if rows:
            try:
                grid = [json.loads(row) for row in rows]
                if self._is_valid_grid(grid):
                    return grid
            except Exception:
                pass

        return None

    def _extract_answer(self, response: str) -> Tuple[Optional[List[List[int]]], str]:
        """
        Extract the grid answer from the model's response.

        Looks for content inside <answer></answer> tags after </think> (if thinking mode).
        """
        # Get content after </think> if in thinking mode
        if self.config.thinking_mode:
            is_valid, content_after_think = validate_thinking_format(response, True)
            if is_valid:
                response_to_parse = content_after_think
            else:
                response_to_parse = response
        else:
            response_to_parse = response

        # Try <answer></answer> tags first
        answer_match = ANSWER_TAG_PATTERN.search(response_to_parse)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            grid = self._parse_grid_from_string(answer_content)
            if grid:
                return grid, "answer_tag"
            else:
                if self.config.full_debug:
                    print(
                        f"    Found answer tag but couldn't parse grid: {answer_content[:100]}..."
                    )
                return None, "answer_tag_parse_failed"

        # Fallback: Try to find grid anywhere in response
        grid = self._parse_grid_from_string(response_to_parse)
        if grid:
            return grid, "fallback_grid_search"

        return None, "no_match"

    def _grids_match(
        self, pred_grid: List[List[int]], gold_grid: List[List[int]]
    ) -> bool:
        """Check if two grids are pixel-perfect matches."""
        if pred_grid is None or gold_grid is None:
            return False
        if len(pred_grid) != len(gold_grid):
            return False
        for pred_row, gold_row in zip(pred_grid, gold_grid):
            if len(pred_row) != len(gold_row):
                return False
            if pred_row != gold_row:
                return False
        return True

    async def _generate_with_retry(
        self, messages: List[Dict], item_id: str
    ) -> Optional[str]:
        """Generate response with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                api_params = {
                    "model": self.server_configs[0].model_name,
                    "messages": messages,
                    "temperature": self.config.eval_temperature,
                }
                if self.config.eval_max_tokens > 0:
                    api_params["max_tokens"] = self.config.eval_max_tokens

                response = await self.client.chat.completions.create(**api_params)

                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()
                    if len(content) >= self.config.min_response_length:
                        return content

            except Exception as e:
                if self.config.full_debug:
                    print(f"  Error on item {item_id} attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return None

    async def _evaluate_single_item(self, item: Dict, idx: int) -> Dict:
        """Evaluate a single ARC-AGI 2 item."""
        # Format prompt
        prompt, gold_grid = self._format_prompt(item)

        # Build messages
        messages = []
        system_content = self._create_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        # Generate response
        response = await self._generate_with_retry(messages, str(idx))

        if response is None:
            return {
                "index": idx,
                "is_correct": False,
                "extracted_grid": None,
                "gold_grid": gold_grid,
                "extraction_method": "generation_failed",
                "error": "Failed to generate response",
            }

        # Extract answer
        extracted_grid, extraction_method = self._extract_answer(response)

        # Score - pixel perfect match
        is_correct = self._grids_match(extracted_grid, gold_grid)

        result = {
            "index": idx,
            "is_correct": is_correct,
            "extracted_grid": extracted_grid,
            "gold_grid": gold_grid,
            "extraction_method": extraction_method,
            "num_training_examples": len(item["fewshots"]),
            "input_grid_size": f"{len(item['question'][0]['input'])}x{len(item['question'][0]['input'][0])}",
            "output_grid_size": (
                f"{len(gold_grid)}x{len(gold_grid[0])}" if gold_grid else "unknown"
            ),
        }

        if self.config.full_debug:
            result["response"] = response
            result["prompt"] = prompt

        return result

    async def evaluate(self, *args, **kwargs):
        """Run the full ARC-AGI 2 evaluation."""
        print("\n" + "=" * 60)
        print("Starting ARC-AGI 2 Evaluation (Generative/Reasoning Mode)")
        print("=" * 60)
        print(f"  Total puzzles: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print("=" * 60)

        # Evaluate all items
        tasks = [
            self._evaluate_single_item(item, idx)
            for idx, item in enumerate(self.eval_items)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating ARC-AGI 2")

        # Calculate metrics
        total = len(results)

        if total == 0:
            print("Warning: No evaluation results obtained")
            return

        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Extraction method breakdown
        method_counts = {}
        for r in results:
            method = r.get("extraction_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        # Grid size stats
        successful_extractions = sum(
            1 for r in results if r["extracted_grid"] is not None
        )

        # Print summary
        print("\n" + "=" * 60)
        print("ARC-AGI 2 Evaluation Results")
        print("=" * 60)
        print(f"  Total puzzles: {total}")
        print(f"  Correct (pixel-perfect): {correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print("-" * 60)
        print(
            f"  Successful grid extractions: {successful_extractions}/{total} ({successful_extractions/total:.1%})"
        )
        print("-" * 60)
        print("  Extraction Methods:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {method}: {count} ({count/total:.1%})")
        print("=" * 60)

        # Save results
        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "correct": correct,
            "successful_extractions": successful_extractions,
            "extraction_rate": successful_extractions / total if total > 0 else 0.0,
            "extraction_methods": method_counts,
        }

        save_eval_results(self.config.data_dir_to_save_evals, metrics, results)

        self.eval_metrics = [
            {
                "accuracy": accuracy,
                "total": total,
                "extraction_rate": successful_extractions / total if total > 0 else 0.0,
            }
        ]

    async def wandb_log(self, step: int):
        """Log metrics to wandb."""
        if self.eval_metrics and wandb.run is not None:
            for metric in self.eval_metrics:
                wandb.log(metric, step=step)

    # Required BaseEnv interface methods
    async def get_next_item(self):
        return None

    async def collect_trajectories(self, *args, **kwargs):
        return []

    async def score(self, *args, **kwargs):
        return []


if __name__ == "__main__":
    ARCAGIEvalEnv.cli()
