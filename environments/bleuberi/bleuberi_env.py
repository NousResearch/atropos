"""
BLEUBERI Environment for Atropos.

This environment implements the BLEUBERI approach for instruction-following
using BLEU scores as rewards. Based on the paper:
"BLEUBERI: BLEU is a surprisingly effective reward for instruction following"
https://arxiv.org/abs/2505.11080
"""

import asyncio
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import TypedDict

import wandb
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup, ScoredDataItem
from atroposlib.envs.server_handling.openai_server import APIServerConfig

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load environment variables from .env file if available
load_dotenv()


# Define our own Item class for the environment
class BLEUBERIItem(TypedDict):
    """Item for BLEUBERI environment"""

    id: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Add the BLEUBERI repository to the Python path
_SUBMODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "bleuberi-repo")
)
if _SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, _SUBMODULE_DIR)

# Import components directly from the BLEUBERI repository
from training.dataset import KeywordDataset  # noqa: E402


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
    cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for datasets and models",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream the dataset",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the dataset",
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

    # Dataset limiting (for testing or development)
    max_train_examples: Optional[int] = Field(
        default=None,
        description="Maximum number of training examples to use (for testing purposes)",
    )
    max_test_examples: Optional[int] = Field(
        default=None,
        description="Maximum number of test examples to use (for testing purposes)",
    )

    # System prompt
    system_prompt: str = Field(
        default=(
            "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider "
            "the problem and deliberate with yourself via systematic reasoning processes to help come to "
            "a correct solution prior to answering. You should enclose your thoughts and internal monologue "
            "inside <think> </think> tags, and then provide your solution or response to the problem. "
            "After your thinking, make sure to clearly provide your final answer inside <answer></answer> tags."
        ),
        description="System prompt for the model",
    )

    # Reasoning
    reasoning: bool = Field(
        default=True,
        description="Whether to enable reasoning in the system prompt",
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

    @classmethod
    def config_init(cls) -> Tuple[BLEUBERIEnvConfig, List[APIServerConfig]]:
        """Initialize configuration with OpenAI API settings."""
        # Load API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")

        # Create environment config with all necessary settings
        env_config = BLEUBERIEnvConfig(
            tokenizer_name="gpt2",
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2,
            batch_size=-1,
            steps_per_eval=1,
            max_token_length=2048,
            wandb_name="bleuberi",
            dataset_name="allenai/tulu-3-sft-mixture",
            dataset_split="train",
            reward_funcs=["bleu"],
            ref_models=["gold"],
            max_train_examples=2,
            max_test_examples=1,
            max_num_workers=2,
            max_eval_workers=1,
            data_path_to_save_groups="bleuberi_openai_test.jsonl",
        )

        # Create OpenAI server config
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                timeout=60,
                num_max_requests_at_once=4,
                num_requests_for_eval=4,
            ),
        ]

        return env_config, server_configs

    def __init__(
        self,
        config: BLEUBERIEnvConfig,
        server_configs,
        slurm=False,
        testing=False,
    ):
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Check for OpenAI API key if using OpenAI server
        if any(
            getattr(server, "server_type", "") == "openai" for server in server_configs
        ):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY environment variable not found!")
            else:
                # Update server configs with API key if needed
                for server in server_configs:
                    if getattr(server, "server_type", "") == "openai" and not getattr(
                        server, "api_key", None
                    ):
                        server.api_key = api_key

        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.test_dataset = None
        self.aggregated_data = None
        self.train_examples = None
        self.test_examples = None
        self.train_index = 0

        # Track training metrics
        self.percent_correct_buffer = []
        self.token_lengths_buffer = []
        self.bleu_scores_buffer = []
        self.category_performance = {}  # Track performance by category

        # Store rollouts for wandb visualization
        self.rollouts_for_wandb = []
        self.num_rollouts_to_keep = 50  # Keep last 50 rollouts

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Initialize reward functions and metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize BLEUBERI dataset for reward calculation."""
        # We'll initialize a KeywordDataset instance that will be used for reward calculation
        # The path parameter is not important as we're only using the reward functions
        self.bleuberi_dataset = KeywordDataset("", self.tokenizer)

        # Initialize the metrics from BLEUBERI
        self.bleu = self.bleuberi_dataset.bleu
        self.rouge = self.bleuberi_dataset.rouge
        self.bertscore = self.bleuberi_dataset.bertscore

    async def setup(self):
        """Set up the environment, loading datasets and preparing examples."""
        self.logger.info("Setting up BLEUBERI environment")

        # Load dataset
        from datasets import load_dataset

        self.dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=self.config.cache_dir,
            streaming=self.config.streaming,
        )

        if self.config.shuffle and not self.config.streaming:
            self.dataset = self.dataset.shuffle(seed=self.config.seed)

        self.logger.info(f"Loaded dataset with {len(self.dataset)} examples")

        # Split into train and test (98% train, 2% test)
        train_size = int(0.98 * len(self.dataset))

        train_dataset = self.dataset.select(range(train_size))
        test_dataset = self.dataset.select(range(train_size, len(self.dataset)))

        self.logger.info(
            f"Split dataset into {len(train_dataset)} train and {len(test_dataset)} test examples"
        )

        # Apply example limits if specified (for testing purposes)
        if (
            self.config.max_train_examples is not None
            and len(train_dataset) > self.config.max_train_examples
        ):
            self.logger.info(
                f"Limiting train dataset to {self.config.max_train_examples} examples (from {len(train_dataset)})"
            )
            train_dataset = train_dataset.select(range(self.config.max_train_examples))

        if (
            self.config.max_test_examples is not None
            and len(test_dataset) > self.config.max_test_examples
        ):
            self.logger.info(
                f"Limiting test dataset to {self.config.max_test_examples} examples (from {len(test_dataset)})"
            )
            test_dataset = test_dataset.select(range(self.config.max_test_examples))

        # Aggregate references
        self.train_examples = await self._aggregate_references(train_dataset)
        self.test_examples = await self._aggregate_references(test_dataset)

        self.logger.info(
            f"Prepared {len(self.train_examples)} train and {len(self.test_examples)} test examples"
        )

        # Shuffle train examples
        random.seed(self.config.seed)
        random.shuffle(self.train_examples)

        self.train_index = 0

    async def _aggregate_references(self, dataset):
        """
        Aggregate references from the dataset based on specified reference models.

        This is an async wrapper around the BLEUBERI aggregation logic.
        """
        # Process examples to extract prompts and references
        examples = []

        for example in dataset:
            example_id = example.get("id", "unknown_id")

            # Get prompt from messages
            prompt = None
            if "messages" in example and example["messages"]:
                for msg in example["messages"]:
                    if msg.get("role") == "user":
                        prompt = msg.get("content")
                        break

            # Get ground truth from messages
            ground_truth = None
            if "messages" in example and example["messages"]:
                for msg in example["messages"]:
                    if msg.get("role") == "assistant":
                        ground_truth = msg.get("content")
                        break

            # Skip examples without prompt or ground truth
            if not prompt or not ground_truth:
                continue

            # Create example
            aggregated_example = {
                "id": example_id,
                "source": example.get("source", "unknown"),
                "messages": example.get("messages", []),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "references": [ground_truth],  # Using ground truth as reference
            }

            examples.append(aggregated_example)

        return examples

    async def get_next_item(self) -> BLEUBERIItem:
        """Get the next example from the dataset."""
        if not self.train_examples:
            self.logger.warning("No train examples available")
            return None

        # Cycle through the dataset
        example = self.train_examples[self.train_index]
        self.train_index = (self.train_index + 1) % len(self.train_examples)

        # Format the prompt as a conversation
        messages = []

        # Add system message if provided and reasoning is enabled
        if self.config.system_prompt and self.config.reasoning:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Add user prompt
        user_prompt = example.get("prompt")
        if not user_prompt:
            user_prompt = "Please respond to this message."

        messages.append({"role": "user", "content": user_prompt})

        # Create item
        item: BLEUBERIItem = {
            "messages": messages,
            "id": str(example.get("id", f"item_{self.train_index}")),
            "metadata": {
                "references": example.get("references", []),
                "source": example.get("source", "unknown"),
                "prompt": user_prompt,
            },
        }

        return item

    def _extract_answer(self, completion: str) -> str:
        """Extract the answer from a completion with potential thinking tags."""
        # Use the extract_answer method from BLEUBERI's KeywordDataset
        return self.bleuberi_dataset.extract_answer(completion)

    async def _calculate_bleu_score(
        self, response_content: str, references: List[str]
    ) -> float:
        """Calculate BLEU score for a response against references using BLEUBERI implementation."""
        kwargs = {"references": references}
        scores = self.bleuberi_dataset.bleu_reward_func([response_content], **kwargs)
        return scores[0] if scores else 0.0

    async def _calculate_reward(
        self, response_content: str, references: List[str]
    ) -> float:
        """
        Calculate the reward for a response based on the configured reward functions.
        Uses BLEUBERI's reward functions directly.
        """
        # Get the appropriate reward functions from BLEUBERI's implementation
        reward_funcs = self.bleuberi_dataset.get_reward_funcs(self.config.reward_funcs)

        if not reward_funcs:
            self.logger.warning("No valid reward functions found")
            return 0.0

        # Calculate scores using each reward function
        all_scores = []
        for reward_func in reward_funcs:
            # Apply the reward function
            kwargs = {"references": references}
            if (
                hasattr(reward_func, "__name__")
                and reward_func.__name__ == "rm_reward_func"
            ):
                # RM reward function requires prompts
                kwargs["prompts"] = [
                    [{"role": "user", "content": "prompt"}]
                ]  # dummy prompt

            scores = reward_func([response_content], **kwargs)

            if scores and len(scores) > 0:
                all_scores.append(scores[0])

        # Take the average of all scores
        final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        return final_score

    async def cleanup(self):
        """
        Cleanup the environment
        """
        # Let the parent class handle cleanup
        await super().cleanup()

    async def collect_trajectory(
        self, item: BLEUBERIItem
    ) -> Tuple[Optional[ScoredDataItem], List[BLEUBERIItem]]:
        """Generate a response and score it against references."""
        backlog = []

        try:
            # Generate response using the server
            response = await self.server.chat_completion(messages=item["messages"])

            # Extract response content
            response_content = response.choices[0].message.content

            # Get references and prompt from item metadata
            references = item["metadata"].get("references", [])
            prompt = item["metadata"].get("prompt", "")
            source_category = item["metadata"].get("source", "unknown")

            # Calculate reward metrics
            bleu_score = await self._calculate_bleu_score(response_content, references)

            # Calculate final score using the specified reward functions
            final_score = await self._calculate_reward(response_content, references)

            # Track metrics for wandb
            self.percent_correct_buffer.append(1.0 if final_score > 0.5 else 0.0)
            self.token_lengths_buffer.append(
                len(self.tokenizer.encode(response_content))
            )
            self.bleu_scores_buffer.append(bleu_score)

            # Maintain buffer size
            if len(self.percent_correct_buffer) > 100:
                self.percent_correct_buffer.pop(0)
                self.token_lengths_buffer.pop(0)
                self.bleu_scores_buffer.pop(0)

            # Track performance by category
            if source_category not in self.category_performance:
                self.category_performance[source_category] = {"scores": [], "count": 0}

            self.category_performance[source_category]["scores"].append(final_score)
            self.category_performance[source_category]["count"] += 1

            # Keep only the last 100 scores per category
            if len(self.category_performance[source_category]["scores"]) > 100:
                self.category_performance[source_category]["scores"].pop(0)

            # Store rollout for wandb visualization
            rollout_data = {
                "prompt": prompt,
                "response": response_content,
                "references": references,
                "bleu_score": bleu_score,
                "final_score": final_score,
                "category": source_category,
                "is_correct": final_score > 0.5,
            }

            self.rollouts_for_wandb.append(rollout_data)
            if len(self.rollouts_for_wandb) > self.num_rollouts_to_keep:
                self.rollouts_for_wandb.pop(0)

            # Tokenize the response
            tokens = self.tokenizer.encode(response_content)
            mask = [1] * len(tokens)

            # Create scored data item as ScoredDataItem
            scored_data: ScoredDataItem = {
                "tokens": tokens,
                "masks": mask,
                "scores": final_score,
                "messages": item["messages"]
                + [{"role": "assistant", "content": response_content}],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }

            return scored_data, backlog

        except Exception as e:
            self.logger.error(f"Error in collect_trajectory: {e}")
            return None, backlog

    async def collect_trajectories(self, item: BLEUBERIItem) -> Tuple[
        Union[
            Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]
        ],
        List[BLEUBERIItem],
    ]:
        """
        Override the default collect_trajectories method to properly format data for jsonl2html.
        This implementation collects multiple trajectories and formats them correctly for HTML generation.
        """
        # Call the parent class implementation to get the original ScoredDataGroup
        tasks = []
        for _ in range(self.config.group_size):
            tasks.append(self.collect_trajectory(item))
        results = await asyncio.gather(*tasks)

        if any(not isinstance(result[0], dict) for result in results):
            logging.error("something wasn't a ScoredDataItem")
            raise ValueError(
                "collect_trajectory must return a ScoredDataItem or None to use the default "
                "collect_trajectories method"
            )

        backlog = []
        to_postprocess = ScoredDataGroup()
        to_postprocess["tokens"] = []
        to_postprocess["masks"] = []
        to_postprocess["scores"] = []
        to_postprocess["advantages"] = []
        to_postprocess["ref_logprobs"] = []
        to_postprocess["messages"] = []
        to_postprocess["group_overrides"] = {}
        to_postprocess["overrides"] = []
        to_postprocess["images"] = []

        self.logger.info("Processing results for BLEUBERI trajectories")
        for result in results:
            to_postprocess["tokens"].append(result[0]["tokens"])
            to_postprocess["masks"].append(result[0]["masks"])
            to_postprocess["scores"].append(result[0]["scores"])

            if result[0].get("advantages", None) is not None:
                to_postprocess["advantages"].append(result[0]["advantages"])
            if result[0].get("ref_logprobs", None) is not None:
                to_postprocess["ref_logprobs"].append(result[0]["ref_logprobs"])
            if result[0].get("messages", None) is not None:
                to_postprocess["messages"].append(result[0]["messages"])
            if result[0].get("group_overrides", None) is not None:
                to_postprocess["group_overrides"].update(result[0]["group_overrides"])
            if result[0].get("overrides", None) is not None:
                to_postprocess["overrides"].append(result[0]["overrides"])
            if result[0].get("images", None) is not None:
                to_postprocess["images"].append(result[0]["images"])

            backlog.extend(result[1])

        # Process the data for HTML compatibility before sending to the API
        # Convert nested message structure to flat strings for HTML rendering
        if "messages" in to_postprocess and to_postprocess["messages"]:
            # Extract the assistant message content from each result
            html_compatible_messages = []

            for result in results:
                if "messages" in result[0] and result[0]["messages"]:
                    # Find the LAST assistant message (most recent response)
                    assistant_messages = [
                        msg
                        for msg in result[0]["messages"]
                        if msg.get("role") == "assistant"
                    ]

                    if assistant_messages:
                        # Get just the content of the last assistant message
                        last_assistant_msg = assistant_messages[-1]
                        html_compatible_messages.append(
                            last_assistant_msg.get("content", "")
                        )

            # Replace the nested messages with flat strings
            if html_compatible_messages:
                to_postprocess["messages"] = html_compatible_messages
                self.logger.info(
                    f"Prepared HTML-compatible format with {len(html_compatible_messages)} messages"
                )

        # The parent's handle_send_to_api method will write this to JSONL

        return to_postprocess, backlog

    async def evaluate(self):
        """Evaluate the model on the test set."""
        self.logger.info("Starting evaluation")

        if not self.test_examples:
            self.logger.warning("No test examples available for evaluation")
            return

        # Track evaluation metrics
        correct_count = 0
        total_count = 0
        all_bleu_scores = []
        all_final_scores = []
        category_results = {}
        token_lengths = []

        # Create detailed wandb tables for evaluation
        eval_table = wandb.Table(
            columns=[
                "prompt",
                "response",
                "reference",
                "bleu_score",
                "final_score",
                "category",
                "is_correct",
                "token_length",
            ]
        )

        # Process each test example
        for example in self.test_examples:
            # Create messages
            messages = []
            if self.config.system_prompt and self.config.reasoning:
                messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            user_prompt = example.get("prompt")
            if not user_prompt:
                user_prompt = "Please respond to this message."

            messages.append({"role": "user", "content": user_prompt})

            source_category = example.get("source", "unknown")

            # Track category performance
            if source_category not in category_results:
                category_results[source_category] = {
                    "correct": 0,
                    "total": 0,
                    "scores": [],
                }

            # Create item
            item: BLEUBERIItem = {
                "messages": messages,
                "id": str(example.get("id", f"eval_{total_count}")),
                "metadata": {
                    "references": example.get("references", []),
                    "source": source_category,
                    "prompt": user_prompt,
                },
            }

            # Generate response
            try:
                response = await self.server.chat_completion(
                    messages=item["messages"],
                    split="eval",  # Use eval split to track eval separately
                )
                response_content = response.choices[0].message.content

                # Get references
                references = example.get("references", [])
                reference_text = references[0] if references else "No reference"

                # Calculate metrics
                bleu_score = await self._calculate_bleu_score(
                    response_content, references
                )

                # Calculate final score
                final_score = await self._calculate_reward(response_content, references)

                # Get token length
                token_length = len(self.tokenizer.encode(response_content))
                token_lengths.append(token_length)

                # Track scores
                all_bleu_scores.append(bleu_score)
                all_final_scores.append(final_score)

                # Count as correct if score > 0.5
                is_correct = final_score > 0.5
                if is_correct:
                    correct_count += 1

                # Update category stats
                category_results[source_category]["total"] += 1
                category_results[source_category]["scores"].append(final_score)
                if is_correct:
                    category_results[source_category]["correct"] += 1

                # Add to detailed evaluation table
                eval_table.add_data(
                    user_prompt,
                    response_content,
                    reference_text,
                    bleu_score,
                    final_score,
                    source_category,
                    is_correct,
                    token_length,
                )

                total_count += 1

            except Exception as e:
                self.logger.error(f"Error in evaluation: {e}")

        # Calculate evaluation metrics
        accuracy = correct_count / total_count if total_count > 0 else 0

        # Create category performance table
        category_table = wandb.Table(
            columns=["category", "accuracy", "avg_score", "sample_count"]
        )
        for category, results in category_results.items():
            if results["total"] > 0:
                cat_accuracy = results["correct"] / results["total"]
                cat_avg_score = (
                    sum(results["scores"]) / len(results["scores"])
                    if results["scores"]
                    else 0
                )

                category_table.add_data(
                    category, cat_accuracy, cat_avg_score, results["total"]
                )

        # Create comprehensive eval metrics
        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/avg_bleu": (
                sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0
            ),
            "eval/avg_final_score": (
                sum(all_final_scores) / len(all_final_scores) if all_final_scores else 0
            ),
            "eval/avg_token_length": (
                sum(token_lengths) / len(token_lengths) if token_lengths else 0
            ),
            "eval/max_token_length": max(token_lengths) if token_lengths else 0,
            "eval/examples": eval_table,
            "eval/category_performance": category_table,
        }

        # Add histograms for evaluation metrics
        if len(all_final_scores) > 10:
            eval_metrics["eval/score_distribution"] = wandb.Histogram(all_final_scores)

        if len(all_bleu_scores) > 10:
            eval_metrics["eval/bleu_distribution"] = wandb.Histogram(all_bleu_scores)

        if len(token_lengths) > 10:
            eval_metrics["eval/token_length_distribution"] = wandb.Histogram(
                token_lengths
            )

        await self.wandb_log(eval_metrics)
        self.logger.info(f"Evaluation completed: Accuracy = {accuracy:.4f}")

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create a table of rollouts for wandb visualization."""
        if not self.rollouts_for_wandb:
            return wandb_metrics

        # Create rollout table with detailed information
        table = wandb.Table(
            columns=[
                "prompt",
                "response",
                "reference",
                "bleu_score",
                "rouge_score",
                "bertscore",
                "final_score",
                "category",
                "is_correct",
            ]
        )

        # Add data to the table (limit to most recent 20 for display)
        for rollout in self.rollouts_for_wandb[-20:]:
            # Skip any non-dictionary items
            if not isinstance(rollout, dict):
                continue

            # Format references as a single string for display
            reference_text = "No reference"
            references = rollout.get("references", [])
            if references and isinstance(references, list):
                reference_text = references[0]

            table.add_data(
                rollout.get("prompt", ""),
                rollout.get("response", ""),
                reference_text,
                rollout.get("bleu_score", 0.0),
                rollout.get("rouge_score", 0.0),
                rollout.get("bertscore", 0.0),
                rollout.get("final_score", 0.0),
                rollout.get("category", "unknown"),
                rollout.get("is_correct", False),
            )

        wandb_metrics["train/rollouts"] = table
        return wandb_metrics

    async def create_category_performance_table(self, wandb_metrics: Dict) -> Dict:
        """Create a table of performance by category for wandb."""
        if not self.category_performance:
            return wandb_metrics

        # Create category performance table
        table = wandb.Table(
            columns=["category", "avg_score", "correct_rate", "sample_count"]
        )

        # Calculate metrics for each category
        for category, data in self.category_performance.items():
            if data["scores"]:
                avg_score = sum(data["scores"]) / len(data["scores"])
                correct_rate = sum(1 for s in data["scores"] if s > 0.5) / len(
                    data["scores"]
                )

                table.add_data(category, avg_score, correct_rate, data["count"])

        wandb_metrics["train/category_performance"] = table
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb with enhanced visualizations."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add basic training metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        # Add token length statistics
        if self.token_lengths_buffer:
            wandb_metrics["train/avg_token_length"] = sum(
                self.token_lengths_buffer
            ) / len(self.token_lengths_buffer)
            wandb_metrics["train/max_token_length"] = max(self.token_lengths_buffer)

        # Add score distributions
        if self.bleu_scores_buffer:
            wandb_metrics["train/avg_bleu"] = sum(self.bleu_scores_buffer) / len(
                self.bleu_scores_buffer
            )

        # Create histograms for score distributions
        if len(self.bleu_scores_buffer) > 10:
            wandb_metrics["train/bleu_distribution"] = wandb.Histogram(
                self.bleu_scores_buffer
            )

        # Add rollout table and category performance
        wandb_metrics = await self.create_rollout_table(wandb_metrics)
        wandb_metrics = await self.create_category_performance_table(wandb_metrics)

        # Call parent method to handle standard logging
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    BLEUBERIEnv.cli()
