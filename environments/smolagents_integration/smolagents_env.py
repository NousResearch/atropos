"""
SmolagentsEnv - Environment for creating high-quality agent trajectories
for training language models using the SmolaGents agent framework.
"""

import json
import logging
import multiprocessing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .agent_process_runner import run_agent_process
from .server_proxy import ServerProxyManager


@dataclass
class Item:
    prompt: str
    metadata: Dict[str, Any]
    id: Optional[str] = None


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Prevent propagation to root logger to avoid duplicate logging
logger.propagate = False

# Only add handler if not already present
if not logger.handlers:
    # Add a console handler to make sure logs are visible
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class SmolagentsEnvConfig(BaseEnvConfig):
    """Configuration for SmolagentsEnv."""

    dataset_path: str = Field(default="data/gaia", description="Path to GAIA dataset")
    split: str = Field(
        default="validation", description="Dataset split to use (validation, test)"
    )
    use_chat_completion: bool = Field(
        default=True, description="Use chat completion API"
    )
    max_steps: int = Field(default=12, description="Maximum number of agent steps")
    agent_verbosity: int = Field(default=2, description="Agent verbosity level (0-3)")
    scoring_strategy: str = Field(
        default="combined",
        description="Scoring strategy: basic, correctness, or combined",
    )
    length_penalty_weight: float = Field(
        default=0.1, description="Weight for length penalty in scoring (0.0 to disable)"
    )
    save_full_traces: bool = Field(
        default=True, description="Save full agent execution traces in the output"
    )
    # Output path configured in __init__
    data_path_to_save_groups: Optional[str] = Field(
        default=None,
        description="Path to save JSONL output (defaults to timestamped file if None)",
    )
    # Process-based settings
    max_concurrent_processes: int = Field(
        default=5,
        description="Maximum number of concurrent processes for agent execution",
    )
    process_timeout: int = Field(
        default=240,  # 4 minutes by default
        description="Timeout for agent processes in seconds",
    )
    # Debugging options
    debug_scoring: bool = Field(
        default=False, description="Enable detailed score calculation logging"
    )


class SmolagentsEnv(BaseEnv):
    """
    Environment for generating high-quality agent trajectories using the SmolaGents framework.

    This environment:
    1. Loads tasks from the GAIA benchmark dataset
    2. Uses SmolaGents CodeAgent with appropriate tools
    3. Scores trajectories based on correctness and reasoning quality
    4. Integrates with Atropos SFT generation pipeline
    """

    name = "smolagents"
    env_config_cls = SmolagentsEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        """Initialize the config for CLI use."""
        env_config = SmolagentsEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=32,
            steps_per_eval=100,
            max_token_length=4096,
            wandb_name="smolagents",
            include_messages=True,
            # Process-based settings
            max_concurrent_processes=8,
            process_timeout=240,
            # Common settings
            dataset_path="data/gaia",
            split="validation",  # GAIA only supports 'validation' and 'test' splits
            use_chat_completion=True,
            # Debugging options
            debug_scoring=False,  # Set to True to enable detailed score logging
            # Using default timestamped output path from the config definition
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=32,
            ),
        ]
        return env_config, server_configs

    def __init__(
        self,
        config: SmolagentsEnvConfig,
        server_configs: Union[List[APIServerConfig], APIServerConfig],
        slurm=False,
        testing=False,
    ):
        # Set a timestamped output file path if not provided
        if config.data_path_to_save_groups is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.data_path_to_save_groups = f"smolagents_output_{timestamp}.jsonl"
            print(
                f"Using auto-generated output path: {config.data_path_to_save_groups}"
            )

        # Initialize the base class
        super().__init__(config, server_configs, slurm, testing)

        # Initialize dataset variables
        self.examples = []
        self.current_index = 0
        self.iter = 0  # Add iter for checkpoint tracking

        # Initialize the server proxy manager for process-based execution
        self.server_proxy_manager = None  # Will be initialized in setup()

        # Save config for easier access
        self.max_steps = config.max_steps
        self.verbosity = config.agent_verbosity
        self.scoring_strategy = config.scoring_strategy
        self.debug_scoring = config.debug_scoring

        # Track agent execution times and metrics
        self.agent_execution_times = []
        self.percent_correct_buffer = []
        self.eval_metrics = []

    async def setup(self):
        """Set up the environment, load dataset, and initialize server components."""
        logger.info("Setting up SmolagentsEnv...")
        logger.info(f"Using dataset split: {self.config.split}")

        # Initialize the server proxy manager
        logger.info("Setting up process-based isolation for agent execution")
        self.server_proxy_manager = ServerProxyManager(
            server=self.server, max_workers=self.config.max_concurrent_processes
        )
        self.server_proxy_manager.start()
        logger.info(
            f"Started server proxy manager with max_workers={self.config.max_concurrent_processes}"
        )

        # Load the GAIA dataset
        try:
            import os

            # Check if dataset exists
            dataset_path = self.config.dataset_path
            validation_path = os.path.join(
                dataset_path, "2023", "validation", "metadata.jsonl"
            )
            gaia_py_path = os.path.join(dataset_path, "GAIA.py")

            # If dataset files are missing, try to download them
            if not os.path.exists(validation_path) or not os.path.exists(gaia_py_path):
                logger.info(
                    f"GAIA dataset not found at {dataset_path}, attempting to download..."
                )
                from .download_gaia import download_gaia_dataset

                download_success = download_gaia_dataset(dataset_path)
                if not download_success:
                    logger.error("Failed to download GAIA dataset automatically.")
                    logger.error(
                        "Please run: python -m environments.smolagents_integration.download_gaia"
                    )
                    self.examples = []
                    return
                else:
                    logger.info(
                        f"GAIA dataset downloaded successfully to {dataset_path}"
                    )

            logger.info(
                f"Loading GAIA dataset directly from {self.config.dataset_path}"
            )

            # Load the metadata.jsonl file directly instead of using the datasets library
            import json

            metadata_path = os.path.join(
                self.config.dataset_path, "2023", self.config.split, "metadata.jsonl"
            )

            logger.info(f"Reading metadata from: {metadata_path}")

            # Check if the file exists
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                self.examples = []
                return

            # Read the metadata file directly
            self.examples = []
            with open(metadata_path, "r") as f:
                for i, line in enumerate(f):
                    try:
                        example = json.loads(line)
                        self.examples.append(
                            {
                                "question": example["Question"],
                                "true_answer": example["Final answer"],
                                "task": example["Level"],
                                "task_id": (
                                    example["task_id"]
                                    if "task_id" in example
                                    else f"task_{i}"
                                ),
                                "file_name": (
                                    os.path.join(
                                        self.config.dataset_path,
                                        "2023",
                                        self.config.split,
                                        example["file_name"],
                                    )
                                    if example.get("file_name")
                                    else ""
                                ),
                            }
                        )
                    except Exception as parse_error:
                        logger.error(
                            f"Error parsing line {i} of metadata file: {parse_error}"
                        )
                        continue

            logger.info(
                f"Loaded {len(self.examples)} examples from GAIA {self.config.split} set"
            )
        except Exception as e:
            import traceback

            logger.error(f"Error loading GAIA dataset: {type(e).__name__}: {e}")
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
            logger.error(
                "Please run: python -m environments.smolagents_integration.download_gaia"
            )
            # Create empty list if dataset loading fails
            self.examples = []

        logger.info("SmolagentsEnv setup complete")

    async def get_next_item(self) -> Item:
        """Get the next item from the GAIA dataset."""
        if not self.examples:
            logger.warning("No examples loaded in dataset")
            return None

        # Use iter to track position and support checkpointing
        example = self.examples[self.iter % len(self.examples)]
        self.iter += 1
        self.current_index = self.iter % len(self.examples)

        # Construct the prompt
        prompt = example["question"]

        # Add file information if available
        if example.get("file_name"):
            prompt += f"\n\nTo solve this task, you can use the file at: {example['file_name']}"

        # Create an Item object
        item = Item(
            prompt=prompt,
            metadata={
                "task_id": example["task_id"],
                "task": example["task"],
                "true_answer": example["true_answer"],
                "file_name": example.get("file_name", ""),
                "dataset_idx": self.current_index,
            },
        )

        return item

    async def collect_trajectories(self, items: Union[Item, List[Item]]) -> Tuple[
        Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any]],
        List[Item],
    ]:
        """
        Collect trajectories for multiple items using process-based parallelism.
        """
        # Handle both single item and list of items
        if not isinstance(items, list):
            items = [items] * self.config.group_size

        return await self._collect_trajectories_process_based(items)

    async def _collect_trajectories_process_based(
        self, items: List[Item]
    ) -> Tuple[List[Any], List[Item]]:
        """
        Collect trajectories for multiple items using process-based parallelism.
        """
        logger.info(
            f"Collecting trajectories for {len(items)} items using process-based parallelism"
        )

        # Create a manager for shared objects
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Create agent config dictionary
        agent_config = {
            "max_steps": self.max_steps,
            "verbosity": self.verbosity,
            "use_chat_completion": self.config.use_chat_completion,
            "model_name": getattr(self.server, "model_name", "unknown-model"),
        }

        # Start processes for each item
        processes = []
        proxies = []

        for item in items:
            # Create a server proxy for this process
            server_proxy, proxy_id = self.server_proxy_manager.create_server_proxy(
                model_name=agent_config["model_name"],
                timeout=self.config.process_timeout,
            )
            proxies.append(proxy_id)

            # Start a process for this item
            process = multiprocessing.Process(
                target=run_agent_process,
                args=(
                    item.prompt,
                    item.metadata,
                    server_proxy,
                    agent_config,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)

        logger.info(f"Started {len(processes)} agent processes")

        # Wait for all processes to complete or timeout
        for process in processes:
            process.join(timeout=self.config.process_timeout)

            # Check if process is still alive (timeout)
            if process.is_alive():
                logger.warning(f"Process {process.pid} timed out, terminating")
                process.terminate()
                process.join()

        # Clean up proxies
        for proxy_id in proxies:
            self.server_proxy_manager.remove_proxy(proxy_id)

        # Get all results from the queue
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get(block=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error getting result from queue: {e}")
                break

        logger.info(f"Collected {len(results)} results from processes")

        # Process results
        backlog = []
        to_postprocess = []

        for result in results:
            if result["status"] == "success":
                # Create scored data from successful result
                scored_data = {
                    "prompt": result["task_metadata"].get("prompt", ""),
                    "response": result["response"],
                    "task_id": result["task_id"],
                    "task": result["task_metadata"].get("task", ""),
                    "true_answer": result["task_metadata"].get("true_answer", ""),
                    "execution_time": result["execution_time"],
                }

                # Add agent memory if configured
                if self.config.save_full_traces and "agent_memory" in result:
                    scored_data["agent_memory"] = result["agent_memory"]

                # Score the trajectory
                score = self._score_trajectory(
                    scored_data["prompt"],
                    scored_data["response"],
                    scored_data["true_answer"],
                    scored_data.get("agent_memory"),
                    scored_data["execution_time"],
                )

                scored_data["score"] = score

                # Create ScoredDataGroup
                item_for_scoring = next(
                    (
                        i
                        for i in items
                        if i.metadata.get("task_id") == result["task_id"]
                    ),
                    None,
                )
                if item_for_scoring:
                    scored_group = self._create_scored_data_group(
                        item_for_scoring, scored_data
                    )
                    to_postprocess.append(scored_group)
                else:
                    logger.warning(
                        f"Could not find original item for task_id {result['task_id']}"
                    )
            else:
                # Just log the error and omit this example from training
                error_message = result.get("error_message", "Unknown error")
                task_id = result.get("task_id", "Unknown task")

                logger.warning(
                    f"Omitting failed task {task_id} from training batch: {error_message}"
                )

        # Return processed results
        logger.info(f"Final to_postprocess: len={len(to_postprocess)}")
        return to_postprocess, backlog

    async def postprocess_histories(
        self, histories: Union[ScoredDataGroup, List[ScoredDataGroup]]
    ) -> ScoredDataGroup:
        """
        Post-process the agent histories.

        We need to merge multiple ScoredDataGroups into a single ScoredDataGroup.
        """
        logger.info(
            f"postprocess_histories called with: type={type(histories)}, is_none={histories is None}"
        )
        if isinstance(histories, list):
            logger.info(f"  List length: {len(histories)}")

        if not isinstance(histories, list):
            # If it's already a single ScoredDataGroup, return it with group_overrides
            logger.info(f"  Single history, returning directly: {type(histories)}")
            if (
                "group_overrides" not in histories
                or histories["group_overrides"] is None
            ):
                histories["group_overrides"] = {}
            return histories

        # If we have multiple ScoredDataGroups, merge them
        logger.info(f"  Merging {len(histories)} histories")
        merged = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            advantages=None,
            ref_logprobs=None,
            messages=[] if self.config.include_messages else None,
            group_overrides={},
            overrides=None,
        )

        # Merge all the fields
        for i, history in enumerate(histories):
            logger.info(
                f"  Processing history {i}: type={type(history)}, is_none={history is None}"
            )
            if history is not None:
                logger.info(f"    History {i} tokens: {len(history['tokens'])}")
                merged["tokens"].extend(history["tokens"])
                merged["masks"].extend(history["masks"])
                merged["scores"].extend(history["scores"])

                if merged["messages"] is not None and "messages" in history:
                    logger.info(f"    History {i} messages: {len(history['messages'])}")
                    merged["messages"].extend(history["messages"])

        logger.info(
            f"  Final merged data: tokens={len(merged['tokens'])}, scores={len(merged['scores'])}"
        )
        return merged

    def _score_trajectory(
        self,
        prompt: str,
        agent_response: str,
        true_answer: str,
        agent_memory: List[Dict] = None,
        execution_time: float = 0,  # Parameter kept for backward compatibility but not used in scoring
    ) -> float:
        """
        Score the agent trajectory based on multiple criteria:
        - Answer correctness using GAIA scoring
        - Message format adherence
        - Final answer tool usage
        - Execution success (detection of errors)
        - Efficiency (steps only)

        Args:
            prompt: The original task prompt
            agent_response: The final response from the agent
            true_answer: The ground truth answer
            agent_memory: The memory trace of the agent's steps
            execution_time: Time taken for execution (not used in scoring)

        Returns:
            float: A score between 0.0 and 1.0
        """
        # Import all scoring functions upfront
        from .evaluations.smolagent_integrations.rubrics.gaia_scorer import (
            check_close_call,
            question_scorer,
        )
        from .evaluations.smolagent_integrations.smolagents_scorer import (
            calculate_efficiency_score,
            calculate_execution_score,
            check_final_answer_usage,
            check_format_adherence,
        )

        # Initialize component scores
        correctness_score = 0.0
        format_score = 0.0
        final_answer_score = 0.0
        execution_score = 0.0
        efficiency_score = 0.0

        # 1. Calculate correctness score using GAIA scorer
        # Ensure agent_response is a string before scoring
        if not isinstance(agent_response, str):
            logger.warning(
                f"agent_response is not a string, it's a {type(agent_response)}: {agent_response}"
            )
            try:
                if isinstance(agent_response, set):
                    # Convert sets to comma-separated strings
                    agent_response = ", ".join(str(item) for item in agent_response)
                else:
                    # Try to convert other types to string
                    agent_response = str(agent_response)
            except Exception as e:
                logger.error(f"Failed to convert agent_response to string: {e}")
                agent_response = ""

        is_correct = question_scorer(agent_response, true_answer)
        is_near_correct = check_close_call(agent_response, true_answer, is_correct)

        if is_correct:
            correctness_score = 1.0
        elif is_near_correct:
            correctness_score = 0.5

        # 2. Check format adherence
        if agent_memory:
            format_scores = []
            for step in agent_memory:
                if "content" in step and isinstance(step["content"], str):
                    format_scores.append(check_format_adherence(step["content"]))
                elif "model_output" in step and isinstance(step["model_output"], str):
                    format_scores.append(check_format_adherence(step["model_output"]))

            # Average the format scores across all steps
            format_score = (
                sum(format_scores) / len(format_scores) if format_scores else 0.0
            )

            # 3. Check for final_answer tool usage
            final_answer_used = False
            for step in agent_memory:
                if "content" in step and isinstance(step["content"], str):
                    if check_final_answer_usage(step["content"]):
                        final_answer_used = True
                        break
                elif "model_output" in step and isinstance(step["model_output"], str):
                    if check_final_answer_usage(step["model_output"]):
                        final_answer_used = True
                        break

            final_answer_score = 1.0 if final_answer_used else 0.0

            # 4. Check for execution errors and calculate execution score
            execution_score = calculate_execution_score(agent_memory)

            # 5. Calculate efficiency score
            steps_count = len(agent_memory)
            efficiency_score = calculate_efficiency_score(
                steps_count=steps_count, max_steps=self.max_steps
            )

        # Component weights - can be adjusted to emphasize different aspects
        correctness_weight = 0.50  # 50% of score - correctness matters most
        format_weight = 0.20  # 20% - format adherence is important
        final_answer_weight = 0.10  # 10% - using final_answer tool properly
        execution_weight = 0.10  # 10% - avoiding errors
        efficiency_weight = 0.10  # 10% - being efficient

        # Calculate combined score with weights
        combined_score = (
            correctness_score * correctness_weight
            + format_score * format_weight
            + final_answer_score * final_answer_weight
            + execution_score * execution_weight
            + efficiency_score * efficiency_weight
        )

        # Apply length penalty if configured
        length_penalty = 0.0
        if self.config.length_penalty_weight > 0:
            # agent_response should already be a string from the previous conversion
            # but double-check just to be sure
            response_to_measure = (
                agent_response
                if isinstance(agent_response, str)
                else str(agent_response)
            )
            response_length = len(response_to_measure)
            # Penalize very long responses
            if response_length > 2000:
                length_penalty = min(
                    0.3,
                    self.config.length_penalty_weight * (response_length - 2000) / 1000,
                )
                combined_score = max(0.0, combined_score - length_penalty)

        # Debug logging for score calculation
        if self.debug_scoring:
            logger.info("=== SCORE CALCULATION (detailed) ===")
            logger.info("1. Correctness component:")
            logger.info(f"   - True answer: '{true_answer}'")
            logger.info(f"   - Agent answer: '{agent_response}'")
            logger.info(f"   - Is correct: {is_correct}")
            logger.info(f"   - Is near correct: {is_near_correct}")
            logger.info(f"   - Raw score: {correctness_score:.3f}")
            logger.info(f"   - Weight: {correctness_weight}")
            logger.info(
                f"   - Weighted score: {correctness_score * correctness_weight:.3f}"
            )

            logger.info("2. Format adherence component:")
            logger.info(
                f"   - Format scores: {format_scores if 'format_scores' in locals() else []}"
            )
            logger.info(f"   - Raw score: {format_score:.3f}")
            logger.info(f"   - Weight: {format_weight}")
            logger.info(f"   - Weighted score: {format_score * format_weight:.3f}")

            logger.info("3. Final answer tool usage:")
            logger.info(
                f"   - Final answer tool used: {final_answer_used if 'final_answer_used' in locals() else False}"
            )
            logger.info(f"   - Raw score: {final_answer_score:.3f}")
            logger.info(f"   - Weight: {final_answer_weight}")
            logger.info(
                f"   - Weighted score: {final_answer_score * final_answer_weight:.3f}"
            )

            logger.info("4. Execution component:")
            logger.info(f"   - Steps count: {len(agent_memory) if agent_memory else 0}")
            logger.info(f"   - Raw score: {execution_score:.3f}")
            logger.info(f"   - Weight: {execution_weight}")
            logger.info(
                f"   - Weighted score: {execution_score * execution_weight:.3f}"
            )

            logger.info("5. Efficiency component:")
            logger.info(f"   - Steps count: {len(agent_memory) if agent_memory else 0}")
            logger.info(f"   - Max steps: {self.max_steps}")
            logger.info(f"   - Raw score: {efficiency_score:.3f}")
            logger.info(f"   - Weight: {efficiency_weight}")
            logger.info(
                f"   - Weighted score: {efficiency_score * efficiency_weight:.3f}"
            )

            logger.info("6. Length penalty:")
            logger.info(
                f"   - Response length: {len(agent_response) if isinstance(agent_response, str) else 'N/A'}"
            )
            logger.info(f"   - Penalty: {length_penalty:.3f}")

            logger.info("7. Final score calculation:")
            logger.info(
                f"   - Correctness: {correctness_score * correctness_weight:.3f}"
            )
            logger.info(f"   - Format adherence: {format_score * format_weight:.3f}")
            logger.info(
                f"   - Final answer tool: {final_answer_score * final_answer_weight:.3f}"
            )
            logger.info(f"   - Execution: {execution_score * execution_weight:.3f}")
            logger.info(f"   - Efficiency: {efficiency_score * efficiency_weight:.3f}")
            logger.info(f"   - Length penalty: -{length_penalty:.3f}")
            logger.info(f"   - FINAL SCORE: {combined_score:.3f}")

        return combined_score

    def _create_scored_data_group(
        self, item: Item, scored_data: Dict
    ) -> ScoredDataGroup:
        """
        Create a ScoredDataGroup for the trainer API.

        Converts the agent trajectory into tokenized format for the trainer.
        """
        # Prepare the data in message format or token format
        if self.config.include_messages:
            # Create message format with agent memory if available
            messages = []

            # Add system message with task description
            messages.append(
                {
                    "role": "system",
                    "content": "You are an AI assistant solving a task with reasoning and problem-solving skills.",
                }
            )

            # Add user message with the prompt
            messages.append({"role": "user", "content": item.prompt})

            # For message format, extract agent memory if available
            if self.config.save_full_traces and "agent_memory" in scored_data:
                # Add intermediate reasoning steps
                for message in scored_data["agent_memory"]:
                    messages.append(message)
            else:
                # Just add the final response
                messages.append(
                    {"role": "assistant", "content": scored_data["response"]}
                )

            # Create a comprehensive markdown document for HTML compatibility
            # Format similar to the Wikipedia environment
            complete_conversation = []

            # Add task information at the top
            task_type = item.metadata.get("task", "Unknown task")
            task_id = item.metadata.get("task_id", "Unknown ID")
            complete_conversation.append(f"# GAIA Task: {task_type} (ID: {task_id})")

            # Process each message in the conversation
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Skip empty messages
                if not content:
                    continue

                # Add a header for each role
                complete_conversation.append(f"## {role.upper()}")

                # Handle different content formats
                if isinstance(content, list):
                    content_text = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            content_text.append(item["text"])
                        else:
                            content_text.append(str(item))
                    content = "\n".join(content_text)
                elif not isinstance(content, str):
                    # Handle non-string content
                    content = str(content)

                # Add the message content
                complete_conversation.append(content)

                # If this message has agent memory, show it
                if role == "assistant" and i > 1 and self.config.save_full_traces:
                    # Agent memory with thinking is often present in assistant messages
                    if "<thinking>" in content:
                        complete_conversation.append("### Thinking Process")
                        # The thinking is already in the content

                    # Add tool usage information if present
                    if "tool_usage" in msg:
                        tool_name = msg.get("tool_usage", {}).get(
                            "name", "unknown tool"
                        )
                        complete_conversation.append(f"### 🛠️ Tool Used: {tool_name}")
                        tool_args = msg.get("tool_usage", {}).get("args", {})
                        if tool_args:
                            complete_conversation.append("```json")
                            complete_conversation.append(
                                json.dumps(tool_args, indent=2)
                            )
                            complete_conversation.append("```")

                        tool_result = msg.get("tool_usage", {}).get("result", None)
                        if tool_result:
                            complete_conversation.append("### Tool Result")
                            complete_conversation.append("```")
                            complete_conversation.append(str(tool_result))
                            complete_conversation.append("```")

            # Add score information at the end
            complete_conversation.append(f"\n## Score: {scored_data['score']:.4f}")

            # Join everything into a single string with double newlines between sections
            full_conversation_markdown = "\n\n".join(complete_conversation)

            # Create the ScoredDataGroup with a single comprehensive markdown document
            scored_group = ScoredDataGroup(
                tokens=[self.tokenizer.encode(json.dumps(messages))],
                masks=[[1] * len(self.tokenizer.encode(json.dumps(messages)))],
                scores=[scored_data["score"]],
                messages=[
                    full_conversation_markdown
                ],  # Use single markdown document for HTML
                _original_messages=[messages],  # Keep original for trainer API
            )

        else:
            # Create a proper conversation with role-based messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant solving a task with reasoning and problem-solving skills.",
                },
                {"role": "user", "content": item.prompt},
                {"role": "assistant", "content": scored_data["response"]},
            ]

            # Use the standard tokenize_for_trainer utility

            # Tokenize using the standard utility (only trains on assistant messages)
            tokenized = tokenize_for_trainer(
                self.tokenizer,
                messages,
                train_on_all_assistant_turns=True,  # Train on all assistant turns if present
            )

            # Create the ScoredDataGroup
            scored_group = ScoredDataGroup(
                tokens=[tokenized["tokens"]],
                masks=[tokenized["masks"]],
                scores=[scored_data["score"]],
                messages=None,
            )

        return scored_group

    async def evaluate(self, **kwargs):
        """
        Evaluate the agent on a subset of the GAIA benchmark.

        Provides metrics on:
        - Success rate
        - Average score
        - Execution time
        - Step efficiency
        """
        logger.info("Starting evaluation on GAIA benchmark")

        # Use a fixed subset of examples for evaluation
        # Start from a different point than training to avoid overlap
        eval_start = len(self.examples) // 2
        eval_count = min(
            10, len(self.examples) // 10
        )  # 10% of dataset or 10 examples max

        eval_examples = self.examples[eval_start : eval_start + eval_count]

        results = []
        correct_count = 0

        # Create items for evaluation
        eval_items = []
        for example in eval_examples:
            # Create an Item for this example
            item = Item(
                prompt=example["question"],
                metadata={
                    "task_id": example["task_id"],
                    "task": example["task"],
                    "true_answer": example["true_answer"],
                    "file_name": example.get("file_name", ""),
                },
            )
            eval_items.append(item)

        # Use the existing process-based trajectory collection
        scored_groups, _ = await self.collect_trajectories(eval_items)

        # Process the scored groups
        for scored_group in scored_groups:
            if isinstance(scored_group, ScoredDataGroup):
                score = (
                    scored_group["scores"][0]
                    if "scores" in scored_group and scored_group["scores"]
                    else 0
                )

                # Try to extract the task_id from metadata
                task_id = None
                if (
                    "group_overrides" in scored_group
                    and scored_group["group_overrides"]
                ):
                    task_id = scored_group["group_overrides"].get("task_id")

                results.append(
                    {
                        "task_id": task_id or "unknown",
                        "score": score,
                    }
                )

                if score > 0.5:  # Consider it correct if score > 0.5
                    correct_count += 1

                # Since we're using the process-based approach, the execution time
                # is stored in the server metrics which are already tracked

        # Calculate metrics
        if results:
            success_rate = correct_count / len(results)
            avg_score = sum(r["score"] for r in results) / len(results)

            # Calculate average time from agent_execution_times if available
            avg_time = 0
            if self.agent_execution_times:
                avg_time = sum(self.agent_execution_times) / len(
                    self.agent_execution_times
                )

            logger.info(f"Evaluation complete on {len(results)} examples:")
            logger.info(f"  Success rate: {success_rate:.2f}")
            logger.info(f"  Average score: {avg_score:.2f}")
            logger.info(f"  Average execution time: {avg_time:.2f}s")

            # Update wandb metrics
            if self.config.use_wandb:
                metrics = {
                    "eval/success_rate": success_rate,
                    "eval/avg_score": avg_score,
                    "eval/num_examples": len(results),
                    "eval/avg_execution_time": avg_time,
                }

                await self.wandb_log(metrics)

    def save_checkpoint(self, step, data=None):
        """Save environment state for checkpointing."""
        if data is None:
            data = {}
        # Save the iteration counter
        data["iter"] = self.iter
        # Save the current index in the dataset
        data["current_index"] = self.current_index
        # Call the parent class save_checkpoint
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load environment state from checkpoint."""
        # Call parent method first
        super().load_checkpoint()
        # Check if we loaded iter and current_index
        if hasattr(self, "checkpoint_data"):
            if "iter" in self.checkpoint_data:
                self.iter = self.checkpoint_data["iter"]
            if "current_index" in self.checkpoint_data:
                self.current_index = self.checkpoint_data["current_index"]

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb with comprehensive metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = dict()

        # Try to calculate percent_correct, skip if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        # Log agent performance metrics
        if self.agent_execution_times and len(self.agent_execution_times) > 0:
            wandb_metrics["agent/avg_execution_time"] = sum(
                self.agent_execution_times
            ) / len(self.agent_execution_times)
            wandb_metrics["agent/max_execution_time"] = max(self.agent_execution_times)
            wandb_metrics["agent/min_execution_time"] = min(self.agent_execution_times)
            # Reset the buffer
            self.agent_execution_times = []

        # Add dataset iteration tracking
        wandb_metrics["train/dataset_iterations"] = self.iter
        wandb_metrics["train/current_dataset_index"] = self.current_index

        # Add custom evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]

        # Clear buffers after logging
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def cleanup(self):
        """Clean up resources when environment is closed."""
        logger.info("Cleaning up SmolagentsEnv resources")

        # Clean up the server proxy manager
        if self.server_proxy_manager:
            self.server_proxy_manager.stop()
            logger.info("Stopped server proxy manager")

        # Let the parent class do its cleanup
        await super().cleanup()


if __name__ == "__main__":
    SmolagentsEnv.cli()
