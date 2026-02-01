"""
Anagram Word Puzzle Environment

This environment trains models to unscramble anagrams - words with their letters
randomly rearranged. The model must identify the original English word from the
scrambled version.

Example:
- Scrambled: "elppa" -> Answer: "apple"
- Scrambled: "nragle" -> Answer: "learng" or "glaner" (must be valid word)

This tests pattern recognition, vocabulary knowledge, and reasoning skills.
"""

import asyncio
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import wandb
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


# Built-in word list for training (common English words of various lengths)
DEFAULT_WORD_LIST = [
    # 3-4 letter words
    "cat", "dog", "run", "sun", "hat", "bat", "map", "cup", "pen", "bed",
    "red", "big", "hot", "top", "box", "fox", "mix", "six", "day", "way",
    "say", "may", "pay", "lay", "bay", "ray", "hay", "key", "boy", "toy",
    "joy", "cow", "how", "now", "row", "low", "new", "few", "dew", "sew",
    "air", "ear", "far", "car", "bar", "jar", "tar", "war", "arm", "art",
    "ant", "act", "add", "age", "ago", "aid", "aim", "all", "and", "any",
    # 5 letter words
    "apple", "beach", "chair", "dance", "eagle", "flame", "grape", "house",
    "inner", "joker", "knife", "lemon", "mango", "nurse", "ocean", "piano",
    "queen", "river", "snake", "table", "uncle", "video", "water", "xenon",
    "yacht", "zebra", "brain", "climb", "dream", "earth", "fresh", "giant",
    "happy", "image", "judge", "knock", "laugh", "magic", "night", "olive",
    "peace", "quick", "robot", "stone", "tiger", "unity", "voice", "world",
    # 6 letter words
    "basket", "candle", "desert", "engine", "flower", "garden", "helmet",
    "insect", "jacket", "kettle", "laptop", "marble", "needle", "orange",
    "pepper", "quartz", "rabbit", "silver", "temple", "unique", "velvet",
    "wallet", "yellow", "zipper", "anchor", "bridge", "castle", "donkey",
    "escape", "frozen", "ginger", "honest", "island", "jingle", "kitten",
    # 7 letter words
    "amazing", "balance", "captain", "diamond", "elegant", "fantasy", "general",
    "healthy", "imagine", "jealous", "kingdom", "library", "machine", "natural",
    "obvious", "perfect", "quality", "rainbow", "science", "teacher", "unusual",
    "village", "weather", "example", "younger", "zealous", "ancient", "brother",
    "chicken", "dolphin", "emperor", "fiction", "glacier", "harvest", "iceberg",
    # 8+ letter words
    "absolute", "baseball", "children", "daughter", "elephant", "familiar",
    "generous", "handsome", "innocent", "kangaroo", "language", "minister",
    "notebook", "opposite", "pleasure", "question", "romantic", "shoulder",
    "thousand", "universe", "valuable", "wonderful", "airplane", "birthday",
    "computer", "dinosaur", "exercise", "function", "grateful", "hospital",
]


def scramble_word(word: str) -> str:
    """
    Scramble a word's letters randomly, ensuring it's different from original.

    Args:
        word: The original word to scramble

    Returns:
        A scrambled version of the word (different from original if len > 1)
    """
    if len(word) <= 1:
        return word

    letters = list(word.lower())
    original = word.lower()

    # Try to get a different arrangement
    max_attempts = 100
    for _ in range(max_attempts):
        random.shuffle(letters)
        scrambled = "".join(letters)
        if scrambled != original:
            return scrambled

    # If word has all same letters (like "aaa"), just return it
    return "".join(letters)


class AnagramConfig(BaseEnvConfig):
    """Configuration for AnagramEnv with customizable options."""

    thinking_mode: bool = Field(
        default=True,
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for evaluation completions.",
    )

    rollout_temperature: float = Field(
        default=1.0,
        description="Temperature for training rollout completions.",
    )

    eval_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for evaluation completions.",
    )

    train_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for training completions.",
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed API calls.",
    )

    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay in seconds between retry attempts.",
    )

    min_response_length: int = Field(
        default=3,
        ge=1,
        description="Minimum response length to consider valid.",
    )

    min_word_length: int = Field(
        default=4,
        ge=3,
        description="Minimum word length to use for anagram puzzles.",
    )

    max_word_length: int = Field(
        default=10,
        ge=4,
        description="Maximum word length to use for anagram puzzles.",
    )


class AnagramEnv(BaseEnv):
    """
    Anagram Word Puzzle Environment.

    This environment presents scrambled words to the model and rewards it for
    correctly identifying the original word. It tests vocabulary knowledge,
    pattern recognition, and reasoning abilities.
    """

    name = "anagram"
    env_config_cls = AnagramConfig

    def __init__(
        self,
        config: AnagramConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: AnagramConfig = config
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Tracking metrics
        self.successful_solves = 0
        self.failed_solves = 0
        self.format_errors = 0
        self.total_attempts = 0
        self.rollouts_for_wandb = []

        # Pre-compile regex patterns
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

        # System prompts
        self.thinking_system_prompt = self._get_thinking_prompt()
        self.base_system_prompt = (
            "You are an expert at solving anagram puzzles. "
            "Given a scrambled word, identify the original English word. "
            "Wrap your final answer in <answer></answer> tags."
        )

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt."""
        return (
            self.config.custom_thinking_prompt
            if self.config.custom_thinking_prompt
            else "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
            "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
            "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
            "</think> tags, and then provide your solution or response to the problem."
        )

    def _reset_metrics(self) -> None:
        """Reset training metrics."""
        self.percent_correct_buffer = []
        self.successful_solves = 0
        self.failed_solves = 0
        self.format_errors = 0
        self.total_attempts = 0

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        if self.config.thinking_mode:
            return f"{self.thinking_system_prompt}\n\n{self.base_system_prompt}"
        return self.base_system_prompt

    @classmethod
    def config_init(cls) -> Tuple[AnagramConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = AnagramConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=512,
            steps_per_eval=25,
            train_max_tokens=2048,
            eval_max_tokens=2048,
            thinking_mode=True,
            wandb_name="anagram",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        """Set up the environment by preparing word lists."""
        # Filter words by length configuration
        self.word_list = [
            word for word in DEFAULT_WORD_LIST
            if self.config.min_word_length <= len(word) <= self.config.max_word_length
        ]

        # Create separate eval set (last 20% of words)
        split_idx = int(len(self.word_list) * 0.8)
        self.train_words = self.word_list[:split_idx]
        self.eval_words = self.word_list[split_idx:]

        # Shuffle training words
        random.seed(42)
        random.shuffle(self.train_words)

        print(f"\nAnagram Environment Configuration:")
        print(f"  - Word length range: {self.config.min_word_length}-{self.config.max_word_length}")
        print(f"  - Training words: {len(self.train_words)}")
        print(f"  - Evaluation words: {len(self.eval_words)}")
        print(f"  - Thinking mode: {self.config.thinking_mode}")
        print(f"  - Sample words: {self.train_words[:5]}")

        self.iter = 0

    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer from within <answer> tags.

        Args:
            response: Model response text

        Returns:
            Extracted answer or None if not found/invalid format
        """
        if self.config.thinking_mode:
            # Check for exactly one pair of think tags
            think_open_count = len(self._think_pattern.findall(response))
            think_close_count = len(self._think_close_pattern.findall(response))

            if think_open_count != 1 or think_close_count != 1:
                return None

            # Parse only content after </think> tags
            match = self._think_content_pattern.search(response)
            if match:
                response = match.group(1)
            else:
                return None

        # Find answer between tags
        matches = self._answer_pattern.findall(response)

        # Must have exactly one answer block
        if len(matches) != 1:
            return None

        return matches[0].strip().lower()

    def _create_anagram_prompt(self, scrambled: str, original: str) -> str:
        """Create the user prompt for anagram solving task."""
        return (
            f"Unscramble the following letters to form a valid English word.\n\n"
            f"Scrambled letters: {scrambled}\n\n"
            f"Hint: The word has {len(original)} letters.\n\n"
            f"Provide your answer wrapped in <answer></answer> tags."
        )

    async def get_next_item(self) -> Item:
        """Generate next training item with anagram puzzle."""
        self.iter += 1

        # Get next word
        original_word = self.train_words[self.iter % len(self.train_words)]
        scrambled_word = scramble_word(original_word)

        # Create system message
        system_content = self._create_system_content()

        # Create user prompt
        user_content = self._create_anagram_prompt(scrambled_word, original_word)

        prompt = tuple(
            [
                frozenset({"role": "system", "content": system_content}.items()),
                frozenset({"role": "user", "content": user_content}.items()),
            ]
        )

        return (prompt, original_word.lower())

    def _convert_messages_to_list(self, prompt_tuple: Tuple) -> List[Dict]:
        """Convert frozenset message format to list format."""
        messages = []
        for role_dict in prompt_tuple:
            messages.append(dict(role_dict))
        return messages

    def _get_train_completion_params(self) -> Dict:
        """Get completion parameters for training rollouts."""
        return {
            "n": self.config.group_size,
            "max_tokens": self.config.train_max_tokens,
            "temperature": self.config.rollout_temperature,
        }

    def _get_eval_completion_params(self) -> Dict:
        """Get completion parameters for evaluation."""
        return {
            "n": 1,
            "max_tokens": self.config.eval_max_tokens,
            "temperature": self.config.eval_temperature,
            "split": "eval",
        }

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        """Collect and score model trajectories."""
        messages = self._convert_messages_to_list(item[0])
        completion_params = self._get_train_completion_params()

        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay

        for attempt in range(max_retries):
            try:
                completions = await self.server.chat_completion(
                    messages=messages, **completion_params
                )

                if not completions.choices:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return None, []

                # Filter valid completions
                valid_completions = []
                for completion_choice in completions.choices:
                    if (
                        completion_choice.message.content is not None
                        and isinstance(completion_choice.message.content, str)
                        and len(completion_choice.message.content.strip())
                        >= self.config.min_response_length
                    ):
                        valid_completions.append(completion_choice)

                if len(valid_completions) < len(completions.choices) // 2:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue

                # Build trajectories
                to_score = []
                for completion_choice in valid_completions:
                    trajectory_messages = messages + [
                        {
                            "role": "assistant",
                            "content": completion_choice.message.content,
                        }
                    ]
                    to_score.append((tuple(trajectory_messages), item[1]))

                break

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return None, []

        scored_data = await self.score(to_score)

        if scored_data is not None:
            await self.add_rollouts_for_wandb(scored_data, item)

        return scored_data, []

    async def score(self, rollout_group_data: List[Tuple]) -> Optional[ScoredDataGroup]:
        """Score a group of rollout data."""
        if not rollout_group_data:
            return None

        try:
            scores = ScoredDataGroup()
            scores["tokens"] = []
            scores["masks"] = []
            scores["scores"] = []

            random.shuffle(rollout_group_data)

            for item in rollout_group_data:
                if not item or len(item) < 2 or not item[0]:
                    continue

                model_response = item[0][-1]["content"]
                expected_answer = item[1]

                # Extract answer from model response
                extracted_answer = self._extract_answer(model_response)

                # Score 1.0 if exact match, 0.0 otherwise
                reward = 1.0 if extracted_answer == expected_answer else 0.0

                # Track metrics
                self.total_attempts += 1
                if reward == 1.0:
                    self.successful_solves += 1
                else:
                    self.failed_solves += 1
                    if extracted_answer is None:
                        self.format_errors += 1

                # Tokenize the conversation
                out_dict = tokenize_for_trainer(self.tokenizer, item[0])
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

                # Skip obviously bad examples
                if len([1 for mask in masks if mask != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(reward)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            if not scores["tokens"]:
                return None

            # Log group results
            group_successes = sum(1 for score in scores["scores"] if score == 1.0)
            group_size = len(scores["scores"])
            success_indicator = "✅" if group_successes > 0 else "❌"

            total_success_rate = (
                (self.successful_solves / self.total_attempts * 100)
                if self.total_attempts > 0
                else 0.0
            )

            print(
                f"{success_indicator} Group scored: {group_successes}/{group_size} solved | "
                f"Total success rate: {self.successful_solves}/{self.total_attempts} ({total_success_rate:.1f}%)"
            )

            # Update buffer
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))

            # Return None if all scores are the same (no learning signal)
            if len(set(scores["scores"])) == 1:
                return None

            return scores

        except Exception as e:
            print(f"Error in score method: {e}")
            return None

    async def rollout_and_score_eval(self, word: str) -> dict:
        """Rollout and score evaluation for a single word."""
        try:
            scrambled = scramble_word(word)
            expected_answer = word.lower()

            system_content = self._create_system_content()
            user_content = self._create_anagram_prompt(scrambled, word)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            completion_params = self._get_eval_completion_params()

            for attempt in range(self.config.max_retries):
                try:
                    completion = await self.server.chat_completion(
                        messages=messages, **completion_params
                    )

                    if not completion.choices:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        return {"score": 0.0, "sample": None}

                    model_response = completion.choices[0].message.content

                    if model_response is None or len(model_response.strip()) < self.config.min_response_length:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        return {"score": 0.0, "sample": None}

                    break

                except Exception:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    return {"score": 0.0, "sample": None}

            extracted_answer = self._extract_answer(model_response)
            score = 1.0 if extracted_answer == expected_answer else 0.0

            full_messages = messages + [{"role": "assistant", "content": model_response}]

            sample = {
                "messages": full_messages,
                "scrambled_word": scrambled,
                "original_word": word,
                "expected_answer": expected_answer,
                "extracted_answer": extracted_answer,
                "score": int(score),
                "correct": bool(score),
                "format_compliant": extracted_answer is not None,
            }

            return {"score": score, "sample": sample}

        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {"score": 0.0, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Evaluate the model on the evaluation word set."""
        start_time = time.time()

        try:
            eval_tasks = [
                self.rollout_and_score_eval(word) for word in self.eval_words
            ]
            results = await tqdm_asyncio.gather(*eval_tasks)

            valid_results = [
                result
                for result in results
                if not isinstance(result, Exception)
                and result
                and result.get("sample") is not None
            ]

            if not valid_results:
                print("Warning: No valid evaluation results obtained")
                return

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return

        scores = [result["score"] for result in valid_results]
        samples = [result["sample"] for result in valid_results]
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            print("Warning: No valid scores found during evaluation")
            return

        percent_correct = sum(valid_scores) / len(valid_scores)
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        format_compliant = sum(
            1 for sample in samples if sample.get("format_compliant", False)
        )

        end_time = time.time()

        eval_metrics = {
            "eval/percent_correct": percent_correct,
            "eval/total_samples": len(samples),
            "eval/correct_samples": sum(valid_scores),
            "eval/format_compliance_rate": (
                format_compliant / len(samples) if samples else 0.0
            ),
        }

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

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ) -> None:
        """Add rollouts to wandb for visualization."""
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        expected_answer = item[1]

        # Extract scrambled word from the item prompt
        scrambled_word = "unknown"
        try:
            for role_dict in item[0]:
                role_dict_converted = dict(role_dict)
                if role_dict_converted.get("role") == "user":
                    user_content = role_dict_converted.get("content", "")
                    if "Scrambled letters:" in user_content:
                        start = user_content.find("Scrambled letters:") + len("Scrambled letters:")
                        end = user_content.find("\n", start)
                        scrambled_word = user_content[start:end].strip()
                    break
        except Exception:
            scrambled_word = "extraction_failed"

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        num_keep = min(num_keep, len(scored_data["tokens"]))

        current_rollouts = []

        for i in range(num_keep):
            full_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )
            score_val = scored_data["scores"][i]

            current_rollouts.append(
                (full_text, score_val, expected_answer, scrambled_word)
            )

        self.rollouts_for_wandb.append(current_rollouts)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if not self.rollouts_for_wandb:
            return wandb_metrics

        table = wandb.Table(
            columns=["full_text", "score", "expected_answer", "scrambled_word"]
        )

        for group_rollouts in self.rollouts_for_wandb:
            for rollout_tuple in group_rollouts:
                if len(rollout_tuple) == 4:
                    table.add_data(*rollout_tuple)

        wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        if self.total_attempts > 0:
            wandb_metrics["train/success_rate"] = (
                self.successful_solves / self.total_attempts
            )
            wandb_metrics["train/failure_rate"] = (
                self.failed_solves / self.total_attempts
            )
            wandb_metrics["train/format_error_rate"] = (
                self.format_errors / self.total_attempts
            )

        wandb_metrics.update(
            {
                "train/thinking_mode_enabled": (
                    1.0 if self.config.thinking_mode else 0.0
                ),
                "train/total_attempts": self.total_attempts,
                "train/successful_solves": self.successful_solves,
                "train/failed_solves": self.failed_solves,
                "train/format_errors": self.format_errors,
            }
        )

        self._reset_metrics()

        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    AnagramEnv.cli()
