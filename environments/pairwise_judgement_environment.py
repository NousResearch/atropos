import json
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class PairwiseJudgementConfig(BaseEnvConfig):
    """Configuration for PairwiseJudgementEnv with thinking mode option."""

    thinking_mode: bool = Field(
        default=False,
        description="Whether to enable thinking mode with <think></think> tags.",
    )


class PairwiseJudgementEnv(BaseEnv):
    name = "pairwise_judgement"
    env_config_cls = PairwiseJudgementConfig

    def __init__(
        self,
        config: PairwiseJudgementConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: PairwiseJudgementConfig = config
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Detailed metrics tracking
        self.judgment_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.judgment_letter_correct = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.error_count = 0  # Failed to follow format
        self.total_judgments = 0
        self.rollouts_for_wandb = []

        # System prompts
        self.thinking_system_prompt = (
            "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
            "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
            "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
            "</think> tags, and then provide your solution or response to the problem."
        )

        self.judgment_system_prompt = (
            "Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. "
            "You should choose the assistant that follows the user's instructions and answers the user's question best. "
            "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "
            "Begin your evaluation by comparing the four responses and provide a short explanation. "
            "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
            "Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
            '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is best.'
        )

    @classmethod
    def config_init(cls) -> Tuple[PairwiseJudgementConfig, List[APIServerConfig]]:
        env_config = PairwiseJudgementConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers_per_node=16,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="pairwise_judgment",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            thinking_mode=False,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        # Load placeholder train dataset (not actually used since we generate synthetic examples)
        try:
            self.train = load_dataset("example/train", split="train")
            print(f"Loaded placeholder train dataset with {len(self.train)} examples")
        except:
            # Create minimal placeholder data if dataset doesn't exist
            # Note: This isn't actually used since get_next_item() generates synthetic examples
            self.train = [{"question": "What is 2+2?", "answer": "4"}] * 100
            print("Using synthetic placeholder training data")

        # Load evaluation dataset
        self.test = load_dataset("allenai/reward-bench-2", split="test")
        print(f"Loaded reward-bench-2 eval dataset with {len(self.test)} examples")

        # Debug: Show sample evaluation item structure
        if len(self.test) > 0:
            sample_item = self.test[0]
            print(f"\nSample eval item structure:")
            print(f"- Prompt: {sample_item['prompt'][:100]}...")
            print(f"- Chosen responses: {len(sample_item['chosen'])}")
            print(f"- Rejected responses: {len(sample_item['rejected'])}")
            print(f"- First chosen (truncated): {sample_item['chosen'][0][:200]}...")
            print(
                f"- First rejected (truncated): {sample_item['rejected'][0][:200]}..."
            )

        self.iter = 0

    def process_judgement(self, judgment: str, track_metrics: bool = True) -> str:
        """Extract judgment from model response."""
        if self.config.thinking_mode:
            # Check for exactly one pair of think tags
            think_open_count = len(re.findall(r"<think>", judgment))
            think_close_count = len(re.findall(r"</think>", judgment))

            if think_open_count != 1 or think_close_count != 1:
                if track_metrics:
                    self.error_count += 1
                    self.total_judgments += 1
                return "error"

            # Parse only content after </think> tags
            match = re.search(r"</think>\s*(.*)", judgment, re.DOTALL)
            if match:
                judgment = match.group(1)
            else:
                if track_metrics:
                    self.error_count += 1
                    self.total_judgments += 1
                return "error"

        if track_metrics:
            self.total_judgments += 1

        if "[[A]]" in judgment:
            if track_metrics:
                self.judgment_letter_counts["A"] += 1
            return "A"
        elif "[[B]]" in judgment:
            if track_metrics:
                self.judgment_letter_counts["B"] += 1
            return "B"
        elif "[[C]]" in judgment:
            if track_metrics:
                self.judgment_letter_counts["C"] += 1
            return "C"
        elif "[[D]]" in judgment:
            if track_metrics:
                self.judgment_letter_counts["D"] += 1
            return "D"
        else:
            if track_metrics:
                self.error_count += 1
            return "error"

    def create_judgment_prompt(self, question: str, answers: List[str]) -> str:
        """Create the user prompt for judgment task."""
        if len(answers) != 4:
            raise ValueError("Need exactly 4 answers for judgment")

        prompt = f"[User Question]\n{question}\n\n"

        for i, answer in enumerate(answers):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"[The Start of Assistant {letter}'s Answer]\n{answer}\n[The End of Assistant {letter}'s Answer]\n\n"

        return prompt.strip()

    async def get_next_item(self) -> Item:
        # Simple placeholder for training data
        self.iter += 1

        # Create system message
        if self.config.thinking_mode:
            system_content = (
                f"{self.thinking_system_prompt}\n\n{self.judgment_system_prompt}"
            )
        else:
            system_content = self.judgment_system_prompt

        # Create varied placeholder judgment tasks
        examples = [
            {
                "question": "What is the capital of France?",
                "correct": "The capital of France is Paris, which has been the capital since 987 AD and serves as the political, economic, and cultural center of the country.",
                "incorrect": [
                    "The capital of France is London.",
                    "France's capital is Berlin, located in central Europe.",
                    "I don't know the answer to this question.",
                ],
            },
            {
                "question": "How do you fix a memory leak in Python?",
                "correct": "To fix memory leaks in Python: 1) Use memory profilers like tracemalloc or memory_profiler to identify leaks, 2) Ensure proper cleanup of resources with context managers, 3) Break circular references, 4) Close files and database connections explicitly, and 5) Use weak references when appropriate.",
                "incorrect": [
                    "Just restart your computer and the memory leak will be fixed.",
                    "Python automatically handles all memory management, so memory leaks are impossible.",
                    "You need to reinstall Python to fix memory leaks.",
                ],
            },
            {
                "question": "Explain the difference between machine learning and artificial intelligence.",
                "correct": "Artificial Intelligence (AI) is the broader field focused on creating systems that can perform tasks typically requiring human intelligence. Machine Learning (ML) is a subset of AI that uses algorithms to learn patterns from data without being explicitly programmed for each task. So ML is one approach to achieving AI.",
                "incorrect": [
                    "Machine learning and artificial intelligence are exactly the same thing with different names.",
                    "Machine learning is much broader than AI and includes all computer science.",
                    "AI is only about robots, while machine learning is only about statistics.",
                ],
            },
        ]

        # Select random example
        example = random.choice(examples)

        # Create list with correct and incorrect answers
        all_answers = [example["correct"]] + example["incorrect"]
        random.shuffle(all_answers)

        # Find where correct answer ended up
        correct_index = all_answers.index(example["correct"])
        correct_answer = chr(65 + correct_index)  # Convert to A, B, C, D

        user_content = self.create_judgment_prompt(example["question"], all_answers)

        prompt = tuple(
            [
                frozenset({"role": "system", "content": system_content}.items()),
                frozenset({"role": "user", "content": user_content}.items()),
            ]
        )

        return (prompt, correct_answer)

    def prepare_eval_item(self, item: dict) -> Tuple[Tuple, str]:
        """
        Prepare an evaluation item from the reward-bench-2 dataset.

        Dataset structure:
        - chosen: list with 1 element (the best response)
        - rejected: list with 3+ elements (worse responses)
        - We take chosen[0] + rejected[:3] to create exactly 4 responses for judgment
        """
        question = item["prompt"]
        chosen_responses = item["chosen"]
        rejected_responses = item["rejected"]

        # Take one chosen response and three rejected responses to create 4-way judgment
        if len(chosen_responses) == 0 or len(rejected_responses) < 3:
            return None, None

        chosen = chosen_responses[0]
        rejected = rejected_responses[:3]

        # Create list with answer and whether it's correct
        data = [(chosen, True)] + [(r, False) for r in rejected]
        random.shuffle(data)

        # Extract shuffled answers and find correct position
        shuffled_answers = [item[0] for item in data]
        correct_index = next(i for i, (_, is_correct) in enumerate(data) if is_correct)
        correct_answer = chr(65 + correct_index)  # Convert to A, B, C, D

        # Create system message
        if self.config.thinking_mode:
            system_content = (
                f"{self.thinking_system_prompt}\n\n{self.judgment_system_prompt}"
            )
        else:
            system_content = self.judgment_system_prompt

        # Create user prompt
        user_content = self.create_judgment_prompt(question, shuffled_answers)

        prompt = tuple(
            [
                frozenset({"role": "system", "content": system_content}.items()),
                frozenset({"role": "user", "content": user_content}.items()),
            ]
        )

        return prompt, correct_answer

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=1024 * 4,
            temperature=0.8,
        )

        to_score = []
        for completion_choice in completions.choices:
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            to_score.append((tuple(trajectory_messages), item[1]))

        scored_data = await self.score(to_score)

        # Add rollouts for wandb visualization
        if scored_data is not None:
            await self.add_rollouts_for_wandb(scored_data, item)

        return scored_data, []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            model_response = item[0][-1]["content"]
            ground_truth = item[1]

            predicted_answer = self.process_judgement(
                model_response, track_metrics=True
            )
            reward = 1.0 if predicted_answer == ground_truth else 0.0

            # Track correct judgments per letter
            if predicted_answer == ground_truth and predicted_answer != "error":
                self.judgment_letter_correct[predicted_answer] += 1

            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward if reward else -1.0)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        if all([score == 1.0 for score in scores["scores"]]) or all(
            [score == -1.0 for score in scores["scores"]]
        ):
            return None

        return scores

    async def rollout_and_score_eval(self, test_item) -> dict:
        """Rollout and score evaluation with detailed sample data collection."""
        prompt, ground_truth = self.prepare_eval_item(test_item)
        if prompt is None:
            return {"score": 0.0, "sample": None}

        messages = []
        for role_dict in prompt:
            messages.append(dict(role_dict))

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(
            prompt=prompt_text,
            n=1,
            max_tokens=1024 * 4,
            temperature=0.0,
            split="eval",
        )

        model_response = completion.choices[0].text
        predicted_answer = self.process_judgement(model_response, track_metrics=False)

        score = 1.0 if predicted_answer == ground_truth else 0.0

        # Extract question and answer choices from the user message
        user_content = messages[1]["content"]
        question_match = re.search(
            r"\[User Question\]\s*(.*?)\s*\[The Start of Assistant A",
            user_content,
            re.DOTALL,
        )
        question = (
            question_match.group(1).strip() if question_match else "Unknown question"
        )

        # Extract individual answer choices
        answer_choices = {}
        for letter in ["A", "B", "C", "D"]:
            pattern = rf"\[The Start of Assistant {letter}\'s Answer\]\s*(.*?)\s*\[The End of Assistant {letter}\'s Answer\]"
            match = re.search(pattern, user_content, re.DOTALL)
            if match:
                answer_choices[letter] = match.group(1).strip()

        # Add full conversation including model response
        full_messages = messages + [{"role": "assistant", "content": model_response}]

        sample = {
            "messages": full_messages,
            "question": question,
            "answer_choices": answer_choices,
            "ground_truth": ground_truth,
            "predicted_judgment": predicted_answer,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
            "thinking_mode": self.config.thinking_mode,
            "format_compliant": predicted_answer != "error",
            "dataset_item_id": test_item.get("id", "unknown"),
            "dataset_subset": test_item.get("subset", "unknown"),
        }

        # Add thinking-specific parsing info
        if self.config.thinking_mode:
            if "</think>" in model_response:
                sample["response_after_think"] = model_response.split("</think>")[
                    -1
                ].strip()
                sample["thinking_content"] = re.search(
                    r"<think>(.*?)</think>", model_response, re.DOTALL
                )
                if sample["thinking_content"]:
                    sample["thinking_content"] = (
                        sample["thinking_content"].group(1).strip()
                    )
            else:
                sample["response_after_think"] = model_response
                sample["thinking_content"] = None

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        results = await tqdm_asyncio.gather(*eval_tasks)

        # Extract scores and samples
        scores = [result["score"] for result in results if result["sample"] is not None]
        samples = [
            result["sample"] for result in results if result["sample"] is not None
        ]
        valid_scores = [s for s in scores if s is not None]

        if valid_scores:
            percent_correct = sum(valid_scores) / len(valid_scores)
            self.eval_metrics.append(("eval/percent_correct", percent_correct))

            # Track performance by subset if available
            subset_scores = {}
            subset_samples = {}
            for i, sample in enumerate(samples):
                if sample and i < len(scores):
                    subset = sample.get("dataset_subset", "unknown")
                    if subset not in subset_scores:
                        subset_scores[subset] = []
                        subset_samples[subset] = []
                    subset_scores[subset].append(scores[i])
                    subset_samples[subset].append(sample)

            # Log subset-specific metrics
            for subset, subset_score_list in subset_scores.items():
                if subset_score_list:
                    valid_subset_scores = [
                        s for s in subset_score_list if s is not None
                    ]
                    if valid_subset_scores:
                        avg_score = sum(valid_subset_scores) / len(valid_subset_scores)
                        self.eval_metrics.append(
                            (f"eval/percent_correct_{subset}", avg_score)
                        )

            # Calculate additional metrics
            format_compliant = sum(
                1 for sample in samples if sample.get("format_compliant", False)
            )
            thinking_mode_used = self.config.thinking_mode

            # Add overall dataset statistics
            self.eval_metrics.append(("eval/total_items", len(self.test)))
            self.eval_metrics.append(("eval/valid_scores", len(valid_scores)))
            self.eval_metrics.append(("eval/subset_count", len(subset_scores)))
            self.eval_metrics.append(
                (
                    "eval/format_compliance_rate",
                    format_compliant / len(samples) if samples else 0.0,
                )
            )

            end_time = time.time()

            # Log evaluation results with sample-level detail
            eval_metrics = {
                "eval/percent_correct": percent_correct,
                "eval/total_samples": len(samples),
                "eval/correct_samples": sum(valid_scores),
                "eval/format_compliance_rate": (
                    format_compliant / len(samples) if samples else 0.0
                ),
            }

            # Add subset metrics to eval_metrics dict
            for subset, subset_score_list in subset_scores.items():
                if subset_score_list:
                    valid_subset_scores = [
                        s for s in subset_score_list if s is not None
                    ]
                    if valid_subset_scores:
                        avg_score = sum(valid_subset_scores) / len(valid_subset_scores)
                        eval_metrics[f"eval/percent_correct_{subset}"] = avg_score

            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": 0.0,
                    "max_tokens": 1024 * 4,
                    "thinking_mode": thinking_mode_used,
                },
            )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        """Add rollouts to wandb for visualization."""
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        # Extract ground truth and question info
        ground_truth = item[1]
        question_info = "placeholder_question"  # This would be the actual question in real implementation

        # Keep a reasonable number of rollouts
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        num_keep = min(num_keep, len(scored_data["tokens"]))

        current_rollouts = []
        for i in range(num_keep):
            # Decode the full trajectory
            full_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )
            score_val = scored_data["scores"][i]

            # Extract the model's judgment
            if len(scored_data["tokens"]) > i:
                messages = []
                try:
                    # Try to reconstruct the conversation
                    if i < len(scored_data.get("messages", [])):
                        messages = scored_data["messages"][i]
                    else:
                        # Fallback to decoding tokens
                        messages = [{"role": "assistant", "content": full_text}]

                    if messages and isinstance(messages[-1], dict):
                        model_response = messages[-1].get("content", "")
                        predicted_judgment = self.process_judgement(
                            model_response, track_metrics=False
                        )
                    else:
                        predicted_judgment = "unknown"
                except Exception:
                    predicted_judgment = "parse_error"
            else:
                predicted_judgment = "unknown"

            current_rollouts.append(
                (
                    full_text,
                    score_val,
                    ground_truth,
                    predicted_judgment,
                    question_info,
                    "thinking" if self.config.thinking_mode else "direct",
                )
            )

        self.rollouts_for_wandb.append(current_rollouts)

        # Keep only recent rollouts
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=[
                    "full_text",
                    "score",
                    "ground_truth",
                    "predicted_judgment",
                    "question_info",
                    "mode",
                ]
            )
            for group_rollouts in self.rollouts_for_wandb:
                for rollout_tuple in group_rollouts:
                    if len(rollout_tuple) == 6:
                        table.add_data(*rollout_tuple)
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Basic accuracy metrics
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        # Judgment letter distribution and accuracy
        total_letters = sum(self.judgment_letter_counts.values())
        if total_letters > 0:
            for letter in ["A", "B", "C", "D"]:
                letter_count = self.judgment_letter_counts[letter]
                letter_correct = self.judgment_letter_correct[letter]

                # Letter frequency
                wandb_metrics[f"train/judgment_freq_{letter}"] = (
                    letter_count / total_letters
                )

                # Letter accuracy
                if letter_count > 0:
                    wandb_metrics[f"train/judgment_acc_{letter}"] = (
                        letter_correct / letter_count
                    )
                else:
                    wandb_metrics[f"train/judgment_acc_{letter}"] = 0.0

        # Error rate metrics
        if self.total_judgments > 0:
            wandb_metrics["train/error_rate"] = self.error_count / self.total_judgments
            wandb_metrics["train/format_compliance_rate"] = 1.0 - (
                self.error_count / self.total_judgments
            )

        # Thinking mode metrics
        if self.config.thinking_mode:
            wandb_metrics["train/thinking_mode_enabled"] = 1.0
        else:
            wandb_metrics["train/thinking_mode_enabled"] = 0.0

        # Dataset metrics
        wandb_metrics["train/total_judgments"] = self.total_judgments

        # Additional configuration metrics
        wandb_metrics["config/group_size"] = self.config.group_size
        wandb_metrics["config/max_token_length"] = self.config.max_token_length

        # Calculate judgment distribution entropy (measure of balance)
        if total_letters > 0:
            import math

            entropy = 0.0
            for letter in ["A", "B", "C", "D"]:
                freq = self.judgment_letter_counts[letter] / total_letters
                if freq > 0:
                    entropy -= freq * math.log(freq)
            wandb_metrics["train/judgment_entropy"] = entropy
            wandb_metrics["train/judgment_balance"] = entropy / math.log(
                4
            )  # Normalized entropy

        # Reset training metrics
        self.percent_correct_buffer = []
        self.judgment_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.judgment_letter_correct = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.error_count = 0
        self.total_judgments = 0

        # Add evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = []

        # Add rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    PairwiseJudgementEnv.cli()
