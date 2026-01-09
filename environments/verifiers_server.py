#
# Atropos adapter for Prime Intellect Environment Hub (via Verifiers library)
#
# Installation:
#   1. pip install verifiers
#   2. uv tool install prime
#   3. prime login
#   4. prime env install owner/env_name  (e.g., will/wordle)
#
# Usage:
#   python environments/verifiers_server.py serve --env.vf_env_name wordle
#   python environments/verifiers_server.py evaluate --env.vf_env_name wordle
#
"""
Verifiers Environment Server for Atropos

This module provides an adapter between Prime Intellect's Environment Hub
(via the `verifiers` library) and Atropos' RL training infrastructure.

The adapter translates:
- Verifiers datasets → Atropos training/eval items
- Verifiers rubrics → Atropos scoring
- Verifiers reward functions → Atropos rewards

Example:
    >>> from environments.verifiers_server import VerifiersEnv, VfEnvConfig
    >>> config = VfEnvConfig(vf_env_name="wordle")
    >>> env = VerifiersEnv(config, server_configs)
    >>> await env.setup()
"""
import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Optional Dependency: verifiers (Prime Intellect Environment Hub)
# =============================================================================
# The verifiers library is intentionally optional. If not installed,
# we provide a clear error message when the user tries to use VerifiersEnv.
try:
    import verifiers as vf  # type: ignore
except ImportError as e:
    vf = None  # type: ignore
    _verifiers_import_error = e
else:
    _verifiers_import_error = None


# =============================================================================
# Type Definitions
# =============================================================================
class VfDataItem(TypedDict):
    """Standard item format from Verifiers datasets."""
    question: str
    answer: str


class RolloutData(TypedDict):
    """Internal format for tracking rollout data during scoring."""
    messages: tuple
    gold_answer: str
    finish_reason: str
    tokens: List[int]
    masks: List[int]
    logprobs: List[float]


# =============================================================================
# Configuration
# =============================================================================
class VfEnvConfig(BaseEnvConfig):
    """
    Configuration for Verifiers Environment adapter.
    
    Attributes:
        vf_env_name: Name of the Prime Hub environment (e.g., "wordle", "will/wordle")
        env_args: Additional arguments passed to vf.load_environment()
        normalize_rewards: Whether to normalize rewards to [-1, 1] range
        min_mask_tokens: Minimum number of non-masked tokens required
    """
    vf_env_name: str = ""
    env_args: dict = {}
    normalize_rewards: bool = False
    min_mask_tokens: int = 10


# =============================================================================
# Main Environment Class
# =============================================================================
class VerifiersEnv(BaseEnv):
    """
    Atropos environment adapter for Prime Intellect Environment Hub.
    
    This class bridges the gap between Verifiers environments and Atropos'
    RL training infrastructure. It handles:
    
    1. Loading environments from the Prime Hub
    2. Dataset management (training and evaluation)
    3. Rollout generation and scoring via Verifiers rubrics
    4. Integration with Atropos' trajectory collection and logging
    
    Example:
        >>> env_config, server_configs = VerifiersEnv.config_init()
        >>> env_config.vf_env_name = "wordle"
        >>> env = VerifiersEnv(config=env_config, server_configs=server_configs)
        >>> await env.setup()
        >>> item = await env.get_next_item()
        >>> trajectories, backlog = await env.collect_trajectories(item)
    """
    
    name = "verifiers"
    env_config_cls = VfEnvConfig
    
    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        """
        Initialize the Verifiers environment adapter.
        
        Args:
            config: Environment configuration including vf_env_name
            server_configs: API server configurations for inference
            slurm: Whether running on SLURM cluster
            testing: Whether in testing mode
            
        Raises:
            ImportError: If verifiers library is not installed
            ValueError: If vf_env_name is empty or environment cannot be loaded
        """
        super().__init__(config, server_configs, slurm, testing)
        
        # Tracking for metrics
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List[Dict] = []
        
        # Validate verifiers is available
        if vf is None:
            raise ImportError(
                "Missing optional dependency 'verifiers'. "
                "Install it to use VerifiersEnv:\n"
                "  pip install verifiers\n\n"
                "For Prime Hub environments, also:\n"
                "  uv tool install prime\n"
                "  prime login\n"
                "  prime env install <owner>/<env_name>"
            ) from _verifiers_import_error
        
        # Validate environment name
        if not config.vf_env_name:
            raise ValueError(
                "vf_env_name is required. Specify the Prime Hub environment name, "
                "e.g., 'wordle' or 'will/wordle'"
            )
        
        # Load the Verifiers environment
        try:
            self.vf_env = vf.load_environment(
                config.vf_env_name, 
                **config.env_args
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load Verifiers environment '{config.vf_env_name}'. "
                f"Make sure it is installed via: prime env install {config.vf_env_name}\n"
                f"Original error: {e}"
            ) from e
        
        # Extract rubric components with validation
        self._setup_rubric()
        
        # Get system prompt (may be None for some environments)
        self.system_prompt = getattr(self.vf_env, 'system_prompt', None) or ""
        
        logger.info(
            f"Loaded Verifiers environment '{config.vf_env_name}' with "
            f"{len(self.reward_funcs)} reward functions"
        )
    
    def _setup_rubric(self) -> None:
        """
        Extract and validate rubric components from the Verifiers environment.
        
        Sets up:
        - self.rubric: The Verifiers Rubric object
        - self.parser: Answer parser (if available)
        - self.reward_funcs: List of reward functions
        - self.reward_weights: Raw weights for each reward function
        - self.reward_scales: Normalized weights (sum to 1.0)
        """
        # Get rubric (may be on Environment or accessible via methods)
        self.rubric = getattr(self.vf_env, 'rubric', None)
        
        # Get parser (optional - not all environments have one)
        if self.rubric and hasattr(self.rubric, 'parser'):
            self.parser = self.rubric.parser
        else:
            self.parser = None
        
        # Get reward functions - try multiple locations
        if hasattr(self.vf_env, 'get_reward_funcs'):
            self.reward_funcs = self.vf_env.get_reward_funcs()
        elif self.rubric and hasattr(self.rubric, 'get_reward_funcs'):
            self.reward_funcs = self.rubric.get_reward_funcs()
        else:
            self.reward_funcs = []
            logger.warning("No reward functions found in environment or rubric")
        
        # Get reward weights
        if hasattr(self.vf_env, 'get_reward_weights'):
            self.reward_weights = self.vf_env.get_reward_weights()
        elif self.rubric and hasattr(self.rubric, 'get_reward_weights'):
            self.reward_weights = self.rubric.get_reward_weights()
        else:
            # Default to equal weights
            self.reward_weights = [1.0] * len(self.reward_funcs)
        
        # Calculate normalized scales with division-by-zero protection
        weight_sum = sum(self.reward_weights)
        if weight_sum > 0:
            self.reward_scales = [w / weight_sum for w in self.reward_weights]
        else:
            # Fallback to equal weights if sum is zero
            n = len(self.reward_weights) or 1
            self.reward_scales = [1.0 / n] * n
            logger.warning("Reward weights sum to zero, using equal weights")
    
    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, List[APIServerConfig]]:
        """
        Initialize default configuration.
        
        Returns:
            Tuple of (VfEnvConfig, List[APIServerConfig]) with sensible defaults.
            
        Note:
            Override vf_env_name before using:
            >>> config, servers = VerifiersEnv.config_init()
            >>> config.vf_env_name = "your_env_name"
        """
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
            # Verifiers-specific defaults
            vf_env_name="",  # Must be set by user
            normalize_rewards=False,
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,  # Uses OpenAI API
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=64,
            ),
        ]
        return env_config, server_configs
    
    async def setup(self) -> None:
        """
        Load datasets from the Verifiers environment.
        
        Populates:
        - self.train: Training dataset (list of VfDataItem)
        - self.test: Evaluation dataset (list of VfDataItem)
        - self.iter: Current training iteration counter
        """
        # Load training data
        raw_train = self.vf_env.get_dataset()
        self.train = self._normalize_dataset(raw_train, "train")
        
        # Load evaluation data
        raw_test = self.vf_env.get_eval_dataset()
        self.test = self._normalize_dataset(raw_test, "eval")
        
        # Initialize iteration counter
        self.iter = 0
        
        logger.info(
            f"Loaded {len(self.train)} training items and "
            f"{len(self.test)} evaluation items"
        )
    
    def _normalize_dataset(
        self, 
        raw_data: Any, 
        dataset_name: str
    ) -> List[VfDataItem]:
        """
        Normalize dataset items to a consistent format.
        
        Handles various input formats from Verifiers environments:
        - Dict with 'question' and 'answer' keys
        - Dict with 'prompt' and 'answer' keys
        - Dict with 'input' and 'output' keys
        
        Args:
            raw_data: Raw dataset from Verifiers
            dataset_name: Name for logging ("train" or "eval")
            
        Returns:
            List of normalized VfDataItem dictionaries
        """
        normalized: List[VfDataItem] = []
        
        for i, item in enumerate(raw_data):
            try:
                # Try different field names that environments might use
                question = (
                    item.get("question") or 
                    item.get("prompt") or 
                    item.get("input") or
                    ""
                )
                answer = (
                    item.get("answer") or 
                    item.get("response") or 
                    item.get("output") or
                    item.get("target") or
                    ""
                )
                
                if question:  # Only include items with questions
                    normalized.append({
                        "question": str(question),
                        "answer": str(answer),
                    })
                else:
                    logger.debug(f"Skipping {dataset_name} item {i}: no question field")
                    
            except (KeyError, TypeError, AttributeError) as e:
                logger.warning(f"Error normalizing {dataset_name} item {i}: {e}")
                continue
        
        if not normalized:
            logger.warning(f"No valid items found in {dataset_name} dataset")
            
        return normalized
    
    async def get_next_item(self) -> VfDataItem:
        """
        Get the next training item.
        
        Cycles through the training dataset infinitely.
        
        Returns:
            Next training item as VfDataItem
        """
        if not self.train:
            raise RuntimeError("Training dataset is empty. Call setup() first.")
        
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item
    
    async def collect_trajectories(
        self, 
        item: VfDataItem
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Generate and score multiple completions for a training item.
        
        This is the core method for RL training. It:
        1. Generates group_size completions for the item
        2. Collects tokens, masks, and logprobs
        3. Scores each completion using Verifiers rubrics
        4. Returns a ScoredDataGroup for the trainer
        
        Args:
            item: Training item with 'question' and 'answer' fields
            
        Returns:
            Tuple of (ScoredDataGroup or None, backlog items)
        """
        # Build the prompt
        user_message = {"role": "user", "content": item["question"]}
        messages = [user_message]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        # Generate completions with managed server (tracks tokens/logprobs)
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )
            
            # Get tracked state (tokens, masks, logprobs)
            state = managed.get_state()
            nodes = state.get("nodes", [])
        
        # Prepare data for scoring
        to_score: List[RolloutData] = []
        for i, choice in enumerate(chat_completions.choices):
            # Build full message history
            full_messages = tuple(messages) + (
                {"role": "assistant", "content": choice.message.content},
            )
            
            # Get tokens and masks from tracked state
            if i < len(nodes):
                tokens = nodes[i].tokens
                masks = nodes[i].masked_tokens
                logprobs = nodes[i].logprobs
            else:
                # Fallback if node tracking failed
                logger.warning(f"No node data for completion {i}")
                continue
            
            to_score.append({
                "messages": full_messages,
                "gold_answer": item["answer"],
                "finish_reason": choice.finish_reason,
                "tokens": tokens,
                "masks": masks,
                "logprobs": logprobs,
            })
        
        # Score all rollouts
        scored = await self.score(to_score)
        
        return scored, []  # No backlog items
    
    async def score(
        self, 
        rollout_group_data: List[RolloutData]
    ) -> Optional[ScoredDataGroup]:
        """
        Score a batch of rollouts using Verifiers rubrics.
        
        Args:
            rollout_group_data: List of rollout data to score
            
        Returns:
            ScoredDataGroup with tokens, masks, and scores, or None if invalid
        """
        if not rollout_group_data:
            return None
        
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []
        scores["messages"] = []
        
        # Shuffle to prevent order bias
        random.shuffle(rollout_group_data)
        
        for item in rollout_group_data:
            # Skip items with too few tokens
            non_masked = len([m for m in item["masks"] if m != -100])
            if non_masked < self.config.min_mask_tokens:
                logger.debug(f"Skipping item with only {non_masked} non-masked tokens")
                continue
            
            # Calculate reward using rubric
            reward = await self._calculate_reward(item)
            
            # Normalize if configured
            if self.config.normalize_rewards:
                # Clamp to [0, 1] then scale to [-1, 1]
                reward = max(0.0, min(1.0, reward))
                reward = reward * 2.0 - 1.0
            
            scores["tokens"].append(item["tokens"])
            scores["masks"].append(item["masks"])
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(reward)
            scores["messages"].append(list(item["messages"]))
            
            # Track for metrics
            self.percent_correct_buffer.append(max(reward, 0))
            
            # Stop when we have enough
            if len(scores["tokens"]) >= self.config.group_size:
                break
        
        # Validate we have enough data
        if not scores["tokens"]:
            logger.debug("No valid rollouts to score")
            return None
        
        # Check if all scores are the same (uninformative for training)
        if self.config.ensure_scores_are_not_same:
            if len(set(scores["scores"])) == 1:
                logger.debug("All scores identical, returning None")
                return None
        
        return scores
    
    async def _calculate_reward(self, item: RolloutData) -> float:
        """
        Calculate the weighted reward for a single rollout.
        
        Uses Verifiers rubric reward functions if available,
        otherwise falls back to simple correctness checking.
        
        Args:
            item: Rollout data including messages and gold_answer
            
        Returns:
            Weighted reward score (typically 0.0 to 1.0)
        """
        if not self.reward_funcs or not self.rubric:
            # Fallback: simple string matching
            response = item["messages"][-1].get("content", "") if item["messages"] else ""
            return 1.0 if item["gold_answer"] in response else 0.0
        
        # Calculate rewards from each reward function
        rewards: List[float] = []
        for func in self.reward_funcs:
            try:
                # Call the Verifiers reward function
                # The API may vary, so we try common signatures
                if asyncio.iscoroutinefunction(self.rubric.call_reward_func):
                    reward = await self.rubric.call_reward_func(
                        func=func,
                        prompt=item["messages"][0].get("content", "") if item["messages"] else "",
                        completion=list(item["messages"]),
                        answer=item["gold_answer"],
                    )
                else:
                    reward = self.rubric.call_reward_func(
                        func=func,
                        prompt=item["messages"][0].get("content", "") if item["messages"] else "",
                        completion=list(item["messages"]),
                        answer=item["gold_answer"],
                    )
                rewards.append(float(reward) if reward is not None else 0.0)
            except Exception as e:
                logger.warning(f"Reward function failed: {e}")
                rewards.append(0.0)
        
        # Apply weighted sum
        if not rewards:
            return 0.0
        
        weighted_sum = sum(r * s for r, s in zip(rewards, self.reward_scales))
        return weighted_sum
    
    async def rollout_and_score_eval(
        self, 
        question: str, 
        answer: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a single evaluation rollout.
        
        Args:
            question: The question/prompt to evaluate
            answer: The gold/expected answer
            **kwargs: Additional arguments (system_prompt, state, info)
            
        Returns:
            Dict with 'score' and 'sample' keys
        """
        system_prompt = kwargs.get("system_prompt") or self.system_prompt
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        # Generate completion
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,  # Deterministic for eval
            )
        
        response_content = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_content})
        
        # Parse answer if parser available
        answer_parsed = None
        if self.parser and hasattr(self.parser, 'parse_answer'):
            try:
                answer_parsed = self.parser.parse_answer(completion=response_content)
            except Exception as e:
                logger.debug(f"Parser failed: {e}")
        
        # Calculate score using reward functions
        rollout_item: RolloutData = {
            "messages": tuple(messages),
            "gold_answer": answer,
            "finish_reason": completion.choices[0].finish_reason or "",
            "tokens": [],
            "masks": [],
            "logprobs": [],
        }
        score = await self._calculate_reward(rollout_item)
        
        # Build sample for logging
        sample = {
            "messages": messages,
            "question": question,
            "gold_answer": answer,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": score,
            "correct": score > 0.5,  # Threshold at 0.5
            "finish_reason": completion.choices[0].finish_reason,
        }
        
        return {"score": score, "sample": sample}
    
    async def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Run evaluation on the test dataset.
        
        Evaluates all test items and logs metrics.
        
        Returns:
            Dict of evaluation metrics
        """
        if not self.test:
            logger.warning("No test data available for evaluation")
            return {"eval/avg_score": 0.0}
        
        start_time = time.time()
        
        # Run all evaluations
        eval_tasks = [
            self.rollout_and_score_eval(
                item["question"],
                item["answer"],
                system_prompt=self.system_prompt,
            )
            for item in self.test
        ]
        
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")
        
        # Extract scores and samples
        scores = [r["score"] for r in results]
        samples = [r["sample"] for r in results]
        
        # Calculate metrics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        percent_correct = sum(1 for s in scores if s > 0.5) / len(scores) if scores else 0.0
        
        end_time = time.time()
        
        # Store metrics for wandb
        self.eval_metrics.append(("eval/avg_score", avg_score))
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        
        eval_metrics = {
            "eval/avg_score": avg_score,
            "eval/percent_correct": percent_correct,
            "eval/num_samples": len(scores),
        }
        
        # Log to Atropos evaluation system
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
    
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None) -> None:
        """
        Log metrics to Weights & Biases.
        
        Includes percent correct and any buffered evaluation metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = {}
        
        # Calculate and log percent correct from buffer
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
            self.percent_correct_buffer = []
        
        # Add any stored eval metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []
        
        # Call parent to handle server metrics
        await super().wandb_log(wandb_metrics)
    
    def save_checkpoint(self, step: int, data: Optional[Dict] = None) -> None:
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)
    
    def load_checkpoint(self) -> None:
        """Load checkpoint and restore iteration state."""
        super().load_checkpoint()


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    VerifiersEnv.cli()
