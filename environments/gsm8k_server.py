import random
import re # Added import
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI. To solve the problem, you must use a multi-thought process. "
    "Enclose your thoughts in <think> tags, with each thought in its own tag like so: "
    "<think><thought_1>first thought</thought_1><thought_2>second thought</thought_2>"
    "...<thought_n>nth thought</thought_n></think>"
    "After your thoughts, provide your answer to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

Your final answer must end with \\boxed{your answer here}.
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""


def extract_final_answer_segment(completion_text: str) -> str:
    """
    Extracts the segment of the completion text that likely contains the final answer.
    This is typically the portion after the last </think> tag.
    """
    last_think_tag_index = completion_text.rfind("</think>")
    if last_think_tag_index != -1:
        return completion_text[last_think_tag_index + len("</think>"):]
    return completion_text # Fallback to the whole string if </think> is not found


def check_multi_thought_format(completion_text: str) -> bool:
    """
    Checks if the completion text follows the multi-thought format.
    Format: <think><thought_1>...</thought_1><thought_2>...</thought_2>...</think>
    """
    think_match = re.search(r"<think>(.*)</think>", completion_text, re.DOTALL)
    if not think_match:
        return False
    
    thinking_block = think_match.group(1)
    
    # Check for at least one well-formed <thought_N>...</thought_N> tag within the <think> block
    # This ensures there's at least one thought, and it's properly closed.
    # Using a non-greedy match for the content of thought_N to handle multiple such tags correctly.
    if re.search(r"<thought_\d+>.*?</thought_\d+>", thinking_block, re.DOTALL):
        return True
        
    return False


class GSM8kRow(TypedDict):
    question: str
    answer: str


class GSM8kEnv(BaseEnv):

    name = "gsm8k"

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
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="gsm8k",
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
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"]
                    .split("#")[-1]
                    .strip()
                    .replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question: str, answer: str) -> number:
        prompt = system_prompt + "\n\nUser: " + question + "\nAssistant: "
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_segment = extract_final_answer_segment(completion.choices[0].text)
        answer_parsed = parse(
            answer_segment,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        return score

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["question"], item["gold_answer"])
            )
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: GSM8kRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        prompt = system_prompt + "\n\nUser: " + item["question"] + "\nAssistant: "
        gold_answer = (
            "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"
        )

        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = list()
        to_backlog = list()
        for i, completion_choice in enumerate(completions.choices):
            # Construct messages for tokenize_for_trainer
            # The system prompt is implicitly part of the prompt passed to the completion endpoint
            # For tokenization, we'll represent the interaction as:
            # 1. User's question (which was part of the input prompt)
            # 2. Assistant's full response (text from completion)
            messages = [
                {"role": "user", "content": item["question"]}, # Original question for context
                {"role": "assistant", "content": completion_choice.text},
            ]
            to_score.append(
                {
                    "messages": messages, # This will be used by tokenize_for_trainer
                    "raw_prompt": prompt, # Keep the full prompt for potential future use/debugging
                    "raw_completion": completion_choice.text, # Keep raw completion
                    "gold_answer": gold_answer,
                    "finish_reason": completion_choice.finish_reason,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["messages"] = list() # Initialize messages list
        final_correctness_scores = [] # Stores base correctness (1.0/-1.0) for non-skipped items
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            random.shuffle(rollout_group_data)
            for data_item in rollout_group_data: # Renamed item to avoid conflict
                # print(item[0][-1]["content"])
                answer_segment = extract_final_answer_segment(data_item["raw_completion"])
                answer_parsed = parse(
                    answer_segment, # Use extracted segment
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                correctness_verified = verify(answer_parsed, gold_parsed)
                # print(
                #     f"message: {data_item['raw_completion']}, ground_truth: {data_item['gold_answer']}, reward: {correctness_verified}"
                # )
                
                base_score = 1.0 if correctness_verified else -1.0
                
                # Manual tokenization and masking
                full_text = data_item["raw_prompt"] + data_item["raw_completion"]
                tokens = self.tokenizer.encode(full_text)
                
                prompt_tokens = self.tokenizer.encode(data_item["raw_prompt"])
                if prompt_tokens and prompt_tokens[-1] == self.tokenizer.eos_token_id:
                    prompt_tokens = prompt_tokens[:-1]
                
                # Basic assertion to ensure prompt tokens are a prefix of full tokens
                # This might need adjustment if the tokenizer adds special tokens differently
                # to concatenated strings vs. individual strings.
                # A more robust check might involve re-encoding raw_completion and comparing.
                assert tokens[:len(prompt_tokens)] == self.tokenizer.encode(data_item["raw_prompt"])[:len(prompt_tokens)], \
                       "Token mismatch between prompt and start of full text"

                masks = [-100] * len(prompt_tokens) + tokens[len(prompt_tokens):]

                # Construct messages for logging (similar to math_server_zero.py)
                log_messages = [
                    {"role": "user", "content": data_item["raw_prompt"]}, 
                    {"role": "assistant", "content": data_item["raw_completion"]}
                ]
                scores["messages"].append(log_messages)

                # remove obviously bad examples (completion too short)
                if sum(1 for m_val in masks if m_val != -100) < 10:
                    # Need to pop the message if we skip, or handle this before appending message
                    scores["messages"].pop() # Remove the last added message
                    continue
                
                # If not skipped:
                final_correctness_scores.append(base_score)

                # Apply format reward/penalty to get the final score for training
                format_bonus = 0.1 
                format_is_good = check_multi_thought_format(data_item["raw_completion"])
                final_score_for_training = base_score + (format_bonus if format_is_good else -format_bonus)
                
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(final_score_for_training)

                if len(scores["tokens"]) >= self.config.group_size: # or final_correctness_scores
                    break
            
            if not final_correctness_scores: # If all items were skipped
                return None

            # Update percent_correct_buffer with actual correctness scores
            for c_score in final_correctness_scores:
                self.percent_correct_buffer.append(max(c_score, 0.0))
            
            # Length penalty check using final_correctness_scores
            # print(scores['scores']) # This would print training scores
            # print(final_correctness_scores) # This would print base scores
            if final_correctness_scores and all([c_score == 1.0 for c_score in final_correctness_scores]):
                # Do length penalty :)
                token_lengths = [len(token) for token in scores["tokens"]]
                if max(token_lengths) == 0:
                    # What? But don't want to crash a run so just in case...
                    return None

                # Get max allowed token length from config
                max_allowed_length = self.config.max_token_length
                # Set threshold at 50% of max_token_length - no penalty below this
                length_threshold = max_allowed_length * 0.5

                # Apply modified length penalty with threshold
                scores["scores"] = []
                for length in token_lengths:
                    if length <= length_threshold:
                        # No penalty for responses under threshold
                        scores["scores"].append(1.0)
                    else:
                        # Calculate how far we are between threshold and max as a percentage
                        percentage_of_range = (length - length_threshold) / (
                            max_allowed_length - length_threshold
                        )
                        # Cap at 1.0 in case length exceeds max_allowed_length
                        percentage_of_range = min(percentage_of_range, 1.0)
                        # Apply linear penalty scaling from 1.0 down to 0.0
                        scores["scores"].append(1.0 - percentage_of_range)
            if all([scores["scores"][0] == score for score in scores["scores"]]):
                return None  # If all the same, we return None
            return scores
        else:
            # If the gold solution is not parseable, we return None
            return None

    async def get_next_item(self) -> GSM8kRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GSM8kEnv.cli()
