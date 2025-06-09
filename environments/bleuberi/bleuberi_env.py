"""
BLEUBERI Environment for Atropos.

This environment implements the BLEUBERI approach for instruction-following
using BLEU scores as rewards. Based on the paper:
"BLEUBERI: BLEU is a surprisingly effective reward for instruction following"
https://arxiv.org/abs/2505.11080
"""

import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import TypedDict

import wandb
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.envs.server_handling.openai_server import APIServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load environment variables from .env file if available
load_dotenv()

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    print(
        "WARNING: OPENAI_API_KEY environment variable not found. Make sure to set it!"
    )
else:
    print(
        f"Found OPENAI_API_KEY environment variable ({os.environ.get('OPENAI_API_KEY')[:5]}...)"
    )


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
        if not api_key:
            print("WARNING: OPENAI_API_KEY environment variable not found!")
            print(
                "Please set the OPENAI_API_KEY environment variable or add it to a .env file"
            )

        # Create environment config with all necessary settings
        env_config = BLEUBERIEnvConfig(
            tokenizer_name="gpt2",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=-1,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="bleuberi",
            dataset_name="allenai/tulu-3-sft-mixture",
            dataset_split="train",
            reward_funcs=["bleu"],
            ref_models=["gold"],
            # Optional: Add a place to save the data
            data_path_to_save_groups="bleuberi_openai_results.jsonl",
        )

        # Create OpenAI server config
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",  # Or your preferred model
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                timeout=60,
                num_max_requests_at_once=8,
                num_requests_for_eval=16,
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
                self.logger.warning(
                    "Please set the OPENAI_API_KEY environment variable or add it to a .env file"
                )
            else:
                self.logger.info(
                    f"Found OPENAI_API_KEY in environment variables: {api_key[:5]}..."
                )

                # Update server configs with API key if needed
                for server in server_configs:
                    if getattr(server, "server_type", "") == "openai" and not getattr(
                        server, "api_key", None
                    ):
                        server.api_key = api_key
                        self.logger.info(
                            f"Updated server config with API key: {api_key[:5]}..."
                        )

        # Print out server configurations for debugging
        for i, server in enumerate(server_configs):
            self.logger.info(f"Server config {i+1}:")
            for key in dir(server):
                if not key.startswith("_") and not callable(getattr(server, key)):
                    value = getattr(server, key)
                    if key == "api_key" and value:
                        self.logger.info(f"  {key}: {value[:5]}...")
                    else:
                        self.logger.info(f"  {key}: {value}")

        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.test_dataset = None
        self.aggregated_data = None
        self.train_examples = None
        self.test_examples = None
        self.train_index = 0

        # Debug server setup
        if hasattr(self, "server"):
            self.logger.info("Server has been initialized:")
            self.logger.info(f"Server type: {type(self.server).__name__}")
            if hasattr(self.server, "servers"):
                self.logger.info(
                    f"Number of server instances: {len(self.server.servers)}"
                )
                for i, server in enumerate(self.server.servers):
                    self.logger.info(
                        f"Server instance {i+1} type: {type(server).__name__}"
                    )
        else:
            self.logger.warning("No 'server' attribute found after initialization!")

        # Track correct responses
        self.percent_correct_buffer = []

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Initialize reward functions and metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize BLEUBERI dataset for reward calculation."""
        # Logger is already initialized in __init__

        # We'll initialize a KeywordDataset instance that will be used for reward calculation
        # The path parameter is not important as we're only using the reward functions
        self.bleuberi_dataset = KeywordDataset("", self.tokenizer)

        # Initialize the metrics from BLEUBERI
        self.bleu = self.bleuberi_dataset.bleu
        self.rouge = self.bleuberi_dataset.rouge
        self.bertscore = self.bleuberi_dataset.bertscore

        self.logger.info("BLEUBERI reward metrics initialized")

    async def setup(self):
        """Set up the environment, loading datasets and preparing examples."""
        self.logger.info("Setting up BLEUBERI environment")

        # Load dataset
        try:
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
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            # Create a small dummy dataset for testing
            from datasets import Dataset

            dummy_data = []
            for i in range(10):
                dummy_data.append(
                    {
                        "id": i,
                        "messages": [
                            {"role": "user", "content": f"Sample prompt {i}"},
                            {"role": "assistant", "content": f"Sample response {i}"},
                        ],
                        "source": "dummy",
                    }
                )
            self.dataset = Dataset.from_list(dummy_data)
            self.logger.info(f"Created dummy dataset with {len(self.dataset)} examples")

        # Split into train and test (98% train, 2% test)
        train_size = int(0.98 * len(self.dataset))

        train_dataset = self.dataset.select(range(train_size))
        test_dataset = self.dataset.select(range(train_size, len(self.dataset)))

        self.logger.info(
            f"Split dataset into {len(train_dataset)} train and {len(test_dataset)} test examples"
        )

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
        return KeywordDataset.extract_answer(self, completion)

    async def _calculate_bleu_score(
        self, response_content: str, references: List[str]
    ) -> float:
        """Calculate BLEU score for a response against references using BLEUBERI implementation."""
        # Create a mock dataset instance to access the reward functions
        dataset = KeywordDataset("", self.tokenizer)

        # Prepare the inputs in the format expected by BLEUBERI
        completion = response_content
        kwargs = {"references": references}

        # Use BLEUBERI's bleu_reward_func method
        scores = dataset.bleu_reward_func([completion], **kwargs)
        return scores[0] if scores else 0.0

    async def _calculate_rouge_score(
        self, response_content: str, references: List[str]
    ) -> float:
        """Calculate ROUGE score for a response against references using BLEUBERI implementation."""
        # Create a mock dataset instance to access the reward functions
        dataset = KeywordDataset("", self.tokenizer)

        # Prepare the inputs in the format expected by BLEUBERI
        completion = response_content
        kwargs = {"references": references}

        # Use BLEUBERI's rouge_reward_func method
        scores = dataset.rouge_reward_func([completion], **kwargs)
        return scores[0] if scores else 0.0

    async def _calculate_bertscore(
        self, response_content: str, references: List[str]
    ) -> float:
        """Calculate BERTScore for a response against references using BLEUBERI implementation."""
        # Create a mock dataset instance to access the reward functions
        dataset = KeywordDataset("", self.tokenizer)

        # Prepare the inputs in the format expected by BLEUBERI
        completion = response_content
        kwargs = {"references": references}

        # Use BLEUBERI's bertscore_reward_func method
        scores = dataset.bertscore_reward_func([completion], **kwargs)
        return scores[0] if scores else 0.0

    async def _calculate_bleu_rouge_f1(
        self, response_content: str, references: List[str]
    ) -> float:
        """Calculate F1 of BLEU and ROUGE scores using BLEUBERI implementation."""
        # Create a mock dataset instance to access the reward functions
        dataset = KeywordDataset("", self.tokenizer)

        # Prepare the inputs in the format expected by BLEUBERI
        completion = response_content
        kwargs = {"references": references}

        # Use BLEUBERI's bleu_rouge_f1_reward_func method
        scores = dataset.bleu_rouge_f1_reward_func([completion], **kwargs)
        return scores[0] if scores else 0.0

    async def _calculate_reward(
        self, response_content: str, references: List[str]
    ) -> float:
        """
        Calculate the reward for a response based on the configured reward functions.
        Uses BLEUBERI's reward functions directly.
        """
        # Get the appropriate reward functions from BLEUBERI's implementation
        self.logger.info(f"Getting reward functions for: {self.config.reward_funcs}")
        reward_funcs = self.bleuberi_dataset.get_reward_funcs(self.config.reward_funcs)
        self.logger.info(f"Found {len(reward_funcs)} reward functions")

        if not reward_funcs:
            self.logger.warning("No valid reward functions found")
            return 0.0

        # Calculate scores using each reward function
        all_scores = []
        for i, reward_func in enumerate(reward_funcs):
            # Apply the reward function
            self.logger.info(
                f"Applying reward function {i+1}/{len(reward_funcs)}: {getattr(reward_func, '__name__', 'unnamed')}"
            )
            kwargs = {"references": references}
            if (
                hasattr(reward_func, "__name__")
                and reward_func.__name__ == "rm_reward_func"
            ):
                # RM reward function requires prompts
                kwargs["prompts"] = [
                    [{"role": "user", "content": "prompt"}]
                ]  # dummy prompt
                self.logger.info("Added prompts for RM reward function")

            self.logger.info(
                f"Calling reward function with response length {len(response_content)} and {len(references)} references"
            )
            scores = reward_func([response_content], **kwargs)
            self.logger.debug(f"Reward function returned scores: {scores}")

            if scores and len(scores) > 0:
                all_scores.append(scores[0])
                self.logger.info(f"Added score: {scores[0]}")

        # Take the average of all scores
        final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self.logger.info(
            f"Calculated final score: {final_score} from {len(all_scores)} individual scores"
        )
        return final_score

    async def collect_trajectory(
        self, item: BLEUBERIItem
    ) -> Tuple[Optional[ScoredDataItem], List[BLEUBERIItem]]:
        """Generate a response and score it against references."""
        self.logger.info(f"Starting collect_trajectory for item {item['id']}")
        backlog = []

        try:
            self.logger.info("Making API call to server.chat_completion...")
            # Generate response using the server
            response = await self.server.chat_completion(messages=item["messages"])
            self.logger.debug(f"API response received: {response}")

            # Extract response content
            response_content = response.choices[0].message.content
            self.logger.info(f"Extracted content: {response_content[:50]}...")

            # Get references from item metadata
            references = item["metadata"].get("references", [])
            self.logger.info(f"References count: {len(references)}")

            # Calculate score using the specified reward functions
            self.logger.info("Calculating reward score...")
            final_score = await self._calculate_reward(response_content, references)
            self.logger.info(f"Final reward score: {final_score}")

            # Track whether the response was deemed correct (score > 0.5)
            self.percent_correct_buffer.append(1.0 if final_score > 0.5 else 0.0)
            if len(self.percent_correct_buffer) > 100:
                self.percent_correct_buffer.pop(0)
            self.logger.info(
                f"Updated percent_correct_buffer, current size: {len(self.percent_correct_buffer)}"
            )

            # Tokenize the response
            self.logger.info("Tokenizing response...")
            tokens = self.tokenizer.encode(response_content)
            mask = [1] * len(tokens)
            self.logger.info(f"Tokenized length: {len(tokens)}")

            # Create scored data item as ScoredDataItem
            self.logger.info("Creating ScoredDataItem...")
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
            self.logger.info("ScoredDataItem created successfully")

            self.logger.info(f"Finished processing item {item['id']}, returning data")
            return scored_data, backlog

        except Exception as e:
            self.logger.error(f"Error in collect_trajectory: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, backlog

    async def evaluate(self):
        """Evaluate the model on the test set."""
        self.logger.info("Starting evaluation")

        if not self.test_examples:
            self.logger.warning("No test examples available for evaluation")
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
            if self.config.system_prompt and self.config.reasoning:
                messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            user_prompt = example.get("prompt")
            if not user_prompt:
                user_prompt = "Please respond to this message."

            messages.append({"role": "user", "content": user_prompt})

            # Create item
            item: BLEUBERIItem = {
                "messages": messages,
                "id": str(example.get("id", f"eval_{total_count}")),
                "metadata": {
                    "references": example.get("references", []),
                    "source": example.get("source", "unknown"),
                    "prompt": user_prompt,
                },
            }

            # Generate response
            try:
                response = await self.server.chat_completion(messages=item["messages"])
                response_content = response.choices[0].message.content

                # Get references
                references = example.get("references", [])

                # Calculate score
                final_score = await self._calculate_reward(response_content, references)
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
                self.logger.error(f"Error in evaluation: {e}")

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
        self.logger.info(f"Evaluation completed: Accuracy = {accuracy:.4f}")

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
