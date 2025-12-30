"""
IFEval (Instruction Following Evaluation) Environment for Atropos

This environment evaluates models on the IFEval benchmark - testing their
ability to follow specific formatting and structural instructions.

Dataset: google/IFEval
Paper: https://arxiv.org/abs/2311.07911

Unlike factual QA benchmarks, IFEval tests instruction following by checking
if responses satisfy specific constraints like:
- Keyword requirements (must include/exclude certain words)
- Length constraints (number of sentences, paragraphs, words)
- Format constraints (JSON, bullet lists, sections, titles)
- Language constraints (respond in specific language)
- Case constraints (all caps, lowercase)
- Start/end constraints (begin/end with specific text)

Metrics:
- prompt_level_strict_acc: All instructions followed exactly
- prompt_level_loose_acc: All instructions followed (with variations tried)
- inst_level_strict_acc: Per-instruction accuracy (strict)
- inst_level_loose_acc: Per-instruction accuracy (loose)

Supports optional thinking mode with <think></think> tags.
"""

import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from eval_helpers import (
    create_system_content,
    get_default_thinking_prompt,
    format_reasoning_debug_info,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)

# Import IFEval instructions from local module (ported from lighteval)
try:
    from ifeval_instructions import instructions_registry

    IFEVAL_AVAILABLE = True
except ImportError:
    try:
        # Try relative import if running from different directory
        from .ifeval_instructions import instructions_registry

        IFEVAL_AVAILABLE = True
    except ImportError:
        IFEVAL_AVAILABLE = False
        print(
            "Warning: Could not import IFEval instructions. Make sure ifeval_instructions module exists."
        )


class IFEvalConfig(BaseEnvConfig):
    """Configuration for IFEval evaluation environment."""

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
        default="google/IFEval",
        description="HuggingFace dataset name for IFEval.",
    )

    eval_split: str = Field(
        default="train",
        description="Dataset split to use for evaluation. IFEval only has 'train' split.",
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
        description="Custom system prompt to append after thinking prompt (if thinking_mode) or use directly.",
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


class IFEvalEnv(BaseEnv):
    """
    IFEval Evaluation Environment for Atropos.

    Evaluates models on instruction-following capabilities using the IFEval benchmark.

    Key features:
    - Tests 25+ types of instruction constraints
    - Strict and loose evaluation modes
    - Prompt-level and instruction-level metrics
    - Optional thinking mode with <think></think> tags
    """

    name = "ifeval_eval"
    env_config_cls = IFEvalConfig

    def __init__(
        self,
        config: IFEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: IFEvalConfig = config

        if not IFEVAL_AVAILABLE:
            raise ImportError(
                "IFEval instructions not available. Install langdetect: pip install langdetect"
            )

        # Initialize metrics tracking
        self.eval_metrics = []

        # Pre-compile regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt."""
        return get_default_thinking_prompt(self.config.custom_thinking_prompt)

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt,
        )

    @classmethod
    def config_init(cls) -> Tuple[IFEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = IFEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="ifeval_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            # IFEval specific defaults
            dataset_name="google/IFEval",
            eval_split="train",
            eval_temperature=0.6,
            eval_max_tokens=0,
            thinking_mode=True,
        )

        server_configs = [
            APIServerConfig(
                model_name="Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9000/v1",
                api_key=os.getenv("OPENAI_API_KEY", "none"),
                num_max_requests_at_once=32,
                num_requests_for_eval=1024,
            ),
        ]

        return env_config, server_configs

    async def setup(self) -> None:
        """Load the IFEval dataset and prepare for evaluation."""
        print("\nIFEval Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            thinking_prompt = self._get_thinking_prompt()
            if thinking_prompt:
                print(f"  Thinking prompt: {thinking_prompt[:100]}...")
            else:
                print(f"  Thinking prompt: None (using API reasoning mode only)")

        # Load IFEval dataset
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.eval_split,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")

            # Show sample structure
            if self.eval_data and self.config.full_debug:
                sample = self.eval_data[0]
                print(f"  Sample fields: {list(sample.keys())}")
                print(
                    f"  Sample instruction_id_list: {sample.get('instruction_id_list', [])[:3]}..."
                )

        except Exception as e:
            print(f"Error loading IFEval dataset: {e}")
            raise

        # Analyze instruction distribution
        instruction_counts = {}
        for item in self.eval_data:
            for instr_id in item.get("instruction_id_list", []):
                instruction_counts[instr_id] = instruction_counts.get(instr_id, 0) + 1

        print(f"\n  Instruction types found: {len(instruction_counts)}")
        if self.config.full_debug:
            for instr_id, count in sorted(
                instruction_counts.items(), key=lambda x: -x[1]
            )[:10]:
                print(f"    {instr_id}: {count}")

        self.all_eval_items = self.eval_data
        self.iter = 0

    def _validate_thinking_format(self, response: str) -> Tuple[bool, str]:
        """Validate thinking format and extract content after </think> tags."""
        if not self.config.thinking_mode:
            return True, response

        think_open_count = len(self._think_pattern.findall(response))
        think_close_count = len(self._think_close_pattern.findall(response))

        if think_open_count != 1 or think_close_count != 1:
            return False, response

        match = self._think_content_pattern.search(response)
        if match:
            return True, match.group(1).strip()
        else:
            return False, response

    def _extract_thinking_content(self, response: str) -> Optional[str]:
        """Extract the content inside <think></think> tags."""
        match = self._thinking_extract_pattern.search(response)
        if match:
            return match.group(1).strip()
        return None

    def _preprocess_response(self, response: str) -> List[str]:
        """
        Preprocess response for loose evaluation.
        Creates variations by removing first/last lines and asterisks.
        Matches lighteval's _preprocess_response.
        """
        all_responses = []
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
        return all_responses

    def _check_instructions(
        self,
        response: str,
        instruction_id_list: List[str],
        kwargs_list: List[Dict[str, Any]],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Check if response follows all instructions.

        Returns dict with strict and loose results for each instruction.
        """
        # Get all response variations for loose evaluation
        all_responses = self._preprocess_response(response)

        is_following_list_strict = []
        is_following_list_loose = []
        instruction_results = []

        for index, instruction_id in enumerate(instruction_id_list):
            try:
                instruction_cls = instructions_registry.INSTRUCTION_DICT.get(
                    instruction_id
                )
                if instruction_cls is None:
                    # Unknown instruction - skip
                    if self.config.full_debug:
                        print(f"    Unknown instruction: {instruction_id}")
                    continue

                instruction = instruction_cls(instruction_id)

                # Build instruction with kwargs (remove None values)
                task_kwargs = {
                    k: v for k, v in kwargs_list[index].items() if v is not None
                }
                instruction.build_description(**task_kwargs)

                # Some instructions need the prompt
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=prompt)

                # Strict check
                strict_pass = False
                if response.strip() and instruction.check_following(response):
                    strict_pass = True
                is_following_list_strict.append(strict_pass)

                # Loose check - try all variations
                loose_pass = False
                for r in all_responses:
                    if r.strip() and instruction.check_following(r):
                        loose_pass = True
                        break
                is_following_list_loose.append(loose_pass)

                instruction_results.append(
                    {
                        "instruction_id": instruction_id,
                        "strict_pass": strict_pass,
                        "loose_pass": loose_pass,
                    }
                )

            except Exception as e:
                if self.config.full_debug:
                    print(f"    Error checking instruction {instruction_id}: {e}")
                is_following_list_strict.append(False)
                is_following_list_loose.append(False)
                instruction_results.append(
                    {
                        "instruction_id": instruction_id,
                        "strict_pass": False,
                        "loose_pass": False,
                        "error": str(e),
                    }
                )

        return {
            "prompt_level_strict": (
                all(is_following_list_strict) if is_following_list_strict else False
            ),
            "prompt_level_loose": (
                all(is_following_list_loose) if is_following_list_loose else False
            ),
            "inst_level_strict": is_following_list_strict,
            "inst_level_loose": is_following_list_loose,
            "instruction_results": instruction_results,
            "num_instructions": len(instruction_id_list),
        }

    async def get_next_item(self):
        """Get next item for training (not used in eval-only environment)."""
        self.iter += 1
        if self.all_eval_items:
            item = self.all_eval_items[self.iter % len(self.all_eval_items)]
            return item
        return None

    async def collect_trajectories(self, item):
        """Collect trajectories (not used in eval-only environment)."""
        return None, []

    async def score(self, rollout_group_data):
        """Score rollouts (not used in eval-only environment)."""
        return None

    async def rollout_and_score_eval(self, eval_item: Dict) -> Dict:
        """Evaluate a single IFEval prompt."""
        try:
            prompt = eval_item.get("prompt", "")
            instruction_id_list = eval_item.get("instruction_id_list", [])
            kwargs_list = eval_item.get("kwargs", [])

            if not prompt or not instruction_id_list:
                return {"result": None, "sample": None}

            # Build messages for model
            messages = []
            system_content = self._create_system_content()
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})

            # Get model response with retry logic
            model_response = None
            finish_reason = None
            for attempt in range(self.config.max_retries):
                try:
                    completion_kwargs = {
                        "messages": messages,
                        "n": 1,
                        "temperature": self.config.eval_temperature,
                        "split": "eval",
                    }
                    if self.config.eval_max_tokens > 0:
                        completion_kwargs["max_tokens"] = self.config.eval_max_tokens

                    if self.config.full_debug:
                        print(f"\n  [API Call] Sending request (attempt {attempt + 1})...")
                        print(f"    Temperature: {completion_kwargs.get('temperature')}")
                        print(f"    Max tokens: {completion_kwargs.get('max_tokens', 'not set (unlimited)')}")
                        print(f"    Thinking mode: {self.config.thinking_mode}")
                        print(f"    Reasoning effort: {self.config.reasoning_effort}")
                        # Show extra_body that will be injected by ServerManager
                        if self.config.thinking_mode or self.config.reasoning_effort:
                            print(f"    (ServerManager will inject reasoning extra_body)")

                    _api_start = time.time()
                    completion = await self.server.chat_completion(**completion_kwargs)
                    _api_elapsed = time.time() - _api_start

                    # Log reasoning token usage if full_debug is enabled
                    if self.config.full_debug and completion:
                        print(f"  [API Response] Received in {_api_elapsed:.2f}s")
                        print(format_reasoning_debug_info(completion))

                    if completion.choices and completion.choices[0].message.content:
                        model_response = completion.choices[0].message.content
                        finish_reason = getattr(
                            completion.choices[0], "finish_reason", None
                        )

                        if (
                            len(model_response.strip())
                            >= self.config.min_response_length
                        ):
                            break
                        elif attempt < self.config.max_retries - 1:
                            if self.config.full_debug:
                                print("  Response too short, retrying...")
                            await asyncio.sleep(self.config.retry_delay)

                except Exception as e:
                    print(
                        f"  API Error (attempt {attempt + 1}/{self.config.max_retries}): {type(e).__name__}: {e}"
                    )
                    if hasattr(e, "response"):
                        try:
                            print(
                                f"    Response: {e.response.text[:500] if hasattr(e.response, 'text') else e.response}"
                            )
                        except Exception:
                            pass
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        print(f"  Failed after {self.config.max_retries} attempts")
                        return {"result": None, "sample": None}

            if not model_response:
                return {"result": None, "sample": None}

            # Handle thinking mode - extract content after </think> for evaluation
            thinking_format_valid, response_for_eval = self._validate_thinking_format(
                model_response
            )

            # Extract thinking content for logging
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)

            # Check instructions
            check_result = self._check_instructions(
                response=response_for_eval,
                instruction_id_list=instruction_id_list,
                kwargs_list=kwargs_list,
                prompt=prompt,
            )

            # Build sample record
            sample = {
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "instruction_id_list": instruction_id_list,
                "model_response": model_response,
                "response_for_eval": (
                    response_for_eval[:1000] + "..."
                    if len(response_for_eval) > 1000
                    else response_for_eval
                ),
                "prompt_level_strict": check_result["prompt_level_strict"],
                "prompt_level_loose": check_result["prompt_level_loose"],
                "inst_level_strict": check_result["inst_level_strict"],
                "inst_level_loose": check_result["inst_level_loose"],
                "num_instructions": check_result["num_instructions"],
                "finish_reason": finish_reason,
                "response_length": len(model_response),
                "thinking_mode": self.config.thinking_mode,
                "thinking_format_valid": thinking_format_valid,
            }

            if self.config.thinking_mode:
                sample["thinking_content"] = (
                    thinking_content[:500] + "..."
                    if thinking_content and len(thinking_content) > 500
                    else thinking_content
                )

            if self.config.full_debug:
                strict_status = "✓" if check_result["prompt_level_strict"] else "✗"
                loose_status = "✓" if check_result["prompt_level_loose"] else "✗"
                print(
                    f"  [{strict_status}/{loose_status}] {len(instruction_id_list)} instructions"
                )

            return {"result": check_result, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"result": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run IFEval evaluation."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting IFEval Evaluation (Instruction Following)")
        print(f"{'='*60}")
        print(f"  Total prompts: {len(self.all_eval_items)}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating IFEval")

            valid_results = [
                r
                for r in results
                if r and r.get("sample") is not None and r.get("result") is not None
            ]

            if not valid_results:
                print("Warning: No valid evaluation results obtained")
                return

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            return

        end_time = time.time()

        # Compute metrics
        samples = [r["sample"] for r in valid_results]
        total_count = len(valid_results)

        # Prompt-level metrics
        prompt_strict_count = sum(
            1 for s in samples if s.get("prompt_level_strict", False)
        )
        prompt_loose_count = sum(
            1 for s in samples if s.get("prompt_level_loose", False)
        )

        prompt_strict_acc = (
            prompt_strict_count / total_count if total_count > 0 else 0.0
        )
        prompt_loose_acc = prompt_loose_count / total_count if total_count > 0 else 0.0

        # Instruction-level metrics
        all_inst_strict = []
        all_inst_loose = []
        for s in samples:
            all_inst_strict.extend(s.get("inst_level_strict", []))
            all_inst_loose.extend(s.get("inst_level_loose", []))

        inst_strict_acc = (
            sum(all_inst_strict) / len(all_inst_strict) if all_inst_strict else 0.0
        )
        inst_loose_acc = (
            sum(all_inst_loose) / len(all_inst_loose) if all_inst_loose else 0.0
        )

        total_instructions = len(all_inst_strict)

        # Average response length
        response_lengths = [s.get("response_length", 0) for s in samples]
        avg_response_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        # Thinking format compliance
        thinking_format_compliant = sum(
            1 for s in samples if s.get("thinking_format_valid", True)
        )
        thinking_format_compliance_rate = (
            thinking_format_compliant / len(samples) if samples else 0.0
        )

        # Thinking utilization
        thinking_utilization = 0
        if self.config.thinking_mode:
            thinking_utilization = sum(1 for s in samples if s.get("thinking_content"))

        # Build metrics dictionary
        eval_metrics = {
            "eval/prompt_level_strict_acc": prompt_strict_acc,
            "eval/prompt_level_loose_acc": prompt_loose_acc,
            "eval/inst_level_strict_acc": inst_strict_acc,
            "eval/inst_level_loose_acc": inst_loose_acc,
            "eval/total_prompts": total_count,
            "eval/total_instructions": total_instructions,
            "eval/prompt_strict_count": prompt_strict_count,
            "eval/prompt_loose_count": prompt_loose_count,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/avg_response_length": avg_response_length,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }

        if self.config.thinking_mode:
            eval_metrics["eval/thinking_format_compliance_rate"] = (
                thinking_format_compliance_rate
            )
            thinking_utilization_rate = (
                thinking_utilization / len(samples) if samples else 0.0
            )
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization_rate

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print("IFEval Evaluation Results")
        print(f"{'='*60}")
        print(
            f"Prompt-Level Strict Accuracy: {prompt_strict_acc:.4f} ({prompt_strict_count}/{total_count})"
        )
        print(
            f"Prompt-Level Loose Accuracy:  {prompt_loose_acc:.4f} ({prompt_loose_count}/{total_count})"
        )
        print(f"Instruction-Level Strict Acc: {inst_strict_acc:.4f}")
        print(f"Instruction-Level Loose Acc:  {inst_loose_acc:.4f}")
        print(f"\nTotal Instructions Evaluated: {total_instructions}")
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Avg Response Length: {avg_response_length:.0f} chars")
        if self.config.thinking_mode:
            print(f"Thinking Format Compliance: {thinking_format_compliance_rate:.4f}")
            print(f"Thinking Utilization: {thinking_utilization}/{total_count}")
        print(f"{'='*60}\n")

        # Log evaluation results
        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "thinking_mode": self.config.thinking_mode,
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        wandb_metrics["config/thinking_mode"] = (
            1.0 if self.config.thinking_mode else 0.0
        )
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    IFEvalEnv.cli()
