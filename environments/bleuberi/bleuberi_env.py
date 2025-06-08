"""
BLEUBERI Environment for Atropos.

This environment implements the BLEUBERI approach for instruction-following
using BLEU scores as rewards. Based on the paper:
"BLEUBERI: BLEU is a surprisingly effective reward for instruction following"
https://arxiv.org/abs/2505.11080
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field

import wandb
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, Item

from .dataset_utils import (
    aggregate_references,
    load_tulu_dataset,
    select_examples,
)
from .reward_functions import REWARD_FUNCTIONS

logger = logging.getLogger(__name__)


class BLEUBERIEnvConfig(BaseEnvConfig):
    """Configuration for the BLEUBERI environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="allenai/tulu-3-sft-mixture",
        description="Name of the dataset on Hugging Face",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use",
    )

    # Reference model configuration
    ref_models: List[str] = Field(
        default=["gold"],
        description="List of reference models to use (or 'gold' for ground truth)",
    )

    # Reward configuration
    reward_funcs: List[str] = Field(
        default=["bleu"],
        description="List of reward functions to use",
    )

    # Selection configuration
    selection_mode: str = Field(
        default="random",
        description="Mode for selecting examples (random, easy, medium, hard)",
    )
    num_examples: Optional[int] = Field(
        default=None,
        description="Number of examples to select",
    )

    # System prompt
    system_prompt: str = Field(
        default=(
            "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider "
            "the problem and deliberate with yourself via systematic reasoning processes to help come to "
            "a correct solution prior to answering. You should enclose your thoughts and internal monologue "
            "inside <think> </think> tags, and then provide your solution or response to the problem."
        ),
        description="System prompt for the model",
    )

    # Random seed
    seed: int = Field(
        default=42,
        description="Random seed for dataset shuffling and example selection",
    )


class BLEUBERIEnv(BaseEnv):
    """
    BLEUBERI Environment for Atropos.

    This environment uses BLEU scores as rewards for training models
    to follow instructions. Based on the paper:
    "BLEUBERI: BLEU is a surprisingly effective reward for instruction following"
    """

    name = "bleuberi"
    env_config_cls = BLEUBERIEnvConfig

    def __init__(
        self,
        config: BLEUBERIEnvConfig,
        server_configs,
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.test_dataset = None
        self.aggregated_data = None
        self.train_examples = None
        self.test_examples = None
        self.train_index = 0

        # Track correct responses
        self.percent_correct_buffer = []

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)

    async def setup(self):
        """Set up the environment, loading datasets and preparing examples."""
        logger.info("Setting up BLEUBERI environment")

        # Load dataset
        self.dataset = load_tulu_dataset(
            dataset_name=self.config.dataset_name,
            dataset_split=self.config.dataset_split,
            shuffle=True,
            seed=self.config.seed,
        )

        # Split into train and test (98% train, 2% test)
        train_size = int(0.98 * len(self.dataset))

        train_dataset = self.dataset.select(range(train_size))
        test_dataset = self.dataset.select(range(train_size, len(self.dataset)))

        logger.info(
            f"Split dataset into {len(train_dataset)} train and {len(test_dataset)} test examples"
        )

        # Aggregate references
        self.train_aggregated = aggregate_references(
            train_dataset, self.config.ref_models
        )
        self.test_aggregated = aggregate_references(
            test_dataset, self.config.ref_models
        )

        # Select examples based on selection mode
        self.train_examples = select_examples(
            self.train_aggregated,
            selection_mode=self.config.selection_mode,
            num_examples=self.config.num_examples,
            seed=self.config.seed,
        )

        self.test_examples = select_examples(
            self.test_aggregated,
            selection_mode="random",
            num_examples=min(len(self.test_aggregated), 100),  # Limit test set size
            seed=self.config.seed,
        )

        logger.info(
            f"Selected {len(self.train_examples)} train and {len(self.test_examples)} test examples"
        )

        # Shuffle train examples
        random.seed(self.config.seed)
        random.shuffle(self.train_examples)

        self.train_index = 0

    async def get_next_item(self) -> Item:
        """Get the next example from the dataset."""
        if not self.train_examples:
            logger.warning("No train examples available")
            return None

        # Cycle through the dataset
        example = self.train_examples[self.train_index]
        self.train_index = (self.train_index + 1) % len(self.train_examples)

        # Format the prompt as a conversation
        messages = []

        # Add system message if provided
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Add user prompt
        user_prompt = example.get("prompt")
        if not user_prompt:
            user_prompt = "Please respond to this message."

        messages.append({"role": "user", "content": user_prompt})

        # Create item
        item = Item(
            messages=messages,
            id=str(example.get("id", f"item_{self.train_index}")),
            metadata={
                "references": example.get("references", []),
                "source": example.get("source", "unknown"),
                "prompt": user_prompt,
            },
        )

        return item

    async def collect_trajectory(self, item: Item) -> Tuple[Dict, List[Item]]:
        """Generate a response and score it against references."""
        backlog = []

        try:
            # Generate response using the server
            response = await self.server.generate_chat_completion(item.messages)

            # Extract response content
            response_content = response.get("content", "")

            # Get references from item metadata
            references = item.metadata.get("references", [])

            # Calculate scores using the specified reward functions
            scores = []
            for reward_func_name in self.config.reward_funcs:
                if reward_func_name in REWARD_FUNCTIONS:
                    reward_func = REWARD_FUNCTIONS[reward_func_name]
                    score = reward_func([response_content], [references])[0]
                    scores.append(score)
                else:
                    logger.warning(f"Unknown reward function: {reward_func_name}")

            # Take the average of all scores
            final_score = sum(scores) / len(scores) if scores else 0.0

            # Track whether the response was deemed correct (score > 0.5)
            self.percent_correct_buffer.append(1.0 if final_score > 0.5 else 0.0)
            if len(self.percent_correct_buffer) > 100:
                self.percent_correct_buffer.pop(0)

            # Tokenize the response
            tokens = self.tokenizer.encode(response_content)
            mask = [1] * len(tokens)

            # Create scored data item
            scored_data = {
                "tokens": tokens,
                "masks": mask,
                "scores": final_score,
                "messages": item.messages
                + [{"role": "assistant", "content": response_content}],
            }

            return scored_data, backlog

        except Exception as e:
            logger.error(f"Error in collect_trajectory: {e}")
            return None, backlog

    async def evaluate(self):
        """Evaluate the model on the test set."""
        logger.info("Starting evaluation")

        if not self.test_examples:
            logger.warning("No test examples available for evaluation")
            return

        # Track evaluation metrics
        correct_count = 0
        total_count = 0
        all_scores = []

        # Create a wandb Table for evaluation examples
        eval_table = wandb.Table(columns=["prompt", "response", "references", "score"])

        # Process each test example
        for example in self.test_examples:
            # Create messages
            messages = []
            if self.config.system_prompt:
                messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            user_prompt = example.get("prompt")
            if not user_prompt:
                user_prompt = "Please respond to this message."

            messages.append({"role": "user", "content": user_prompt})

            # Create item
            item = Item(
                messages=messages,
                id=str(example.get("id", f"eval_{total_count}")),
                metadata={
                    "references": example.get("references", []),
                    "source": example.get("source", "unknown"),
                    "prompt": user_prompt,
                },
            )

            # Generate response
            try:
                response = await self.server.generate_chat_completion(item.messages)
                response_content = response.get("content", "")

                # Get references
                references = example.get("references", [])

                # Calculate scores
                scores = []
                for reward_func_name in self.config.reward_funcs:
                    if reward_func_name in REWARD_FUNCTIONS:
                        reward_func = REWARD_FUNCTIONS[reward_func_name]
                        score = reward_func([response_content], [references])[0]
                        scores.append(score)

                # Take the average of all scores
                final_score = sum(scores) / len(scores) if scores else 0.0
                all_scores.append(final_score)

                # Count as correct if score > 0.5
                is_correct = final_score > 0.5
                if is_correct:
                    correct_count += 1

                # Add to table
                eval_table.add_data(
                    user_prompt,
                    response_content,
                    str(references),
                    final_score,
                )

                total_count += 1

            except Exception as e:
                logger.error(f"Error in evaluation: {e}")

        # Calculate evaluation metrics
        accuracy = correct_count / total_count if total_count > 0 else 0

        # Log evaluation metrics
        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/average_score": (
                sum(all_scores) / len(all_scores) if all_scores else 0
            ),
            "eval/examples": eval_table,
        }

        await self.wandb_log(eval_metrics)
        logger.info(f"Evaluation completed: Accuracy = {accuracy:.4f}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add percent correct metric
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        # Call parent method to handle standard logging
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    BLEUBERIEnv.cli()
