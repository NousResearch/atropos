import asyncio
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union
from pydantic import Field

import openai
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    OpenaiConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Static path to .env file - CHANGE THIS TO YOUR ABSOLUTE PATH
ENV_FILE_PATH = "/home/teknium/trajectory-handler/.env"

class LMJudgeConfig(BaseEnvConfig):
    use_openai_judge: bool = Field(True, description="If external OpenAI API should be used for judging")

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class LLMEquivalencyJudgeEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the LLM Equivalency Judge environment.

        Args:
            config: Configuration for the base environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()

        # Fix path construction
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        
        # Log whether OpenAI judging is enabled in config
        use_openai_judge = getattr(config, 'use_openai_judge', True)
        print(f"OpenAI judging configuration: {'ENABLED' if use_openai_judge else 'DISABLED'}")
        
        # Only check for OpenAI keys if use_openai_judge is enabled
        if use_openai_judge:
            # First check if the environment variables are already set
            self.openai_api_key = (
                os.environ.get("OAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            
            # Load from the specified .env file if API key not already present
            if self.openai_api_key is None and os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"Loaded environment variables from {env_path}")
                self.openai_api_key = (
                    os.environ.get("OAI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                )
            
            # If still None, notify the user
            if self.openai_api_key is None:
                if os.path.exists(env_path):
                    print(f"Warning: OpenAI API key not found in environment or in {env_path} - will use local model")
                else:
                    print(f"Warning: OpenAI API key not found in environment and .env file not found at {env_path} - will use local model")
                # Force use_openai_for_judging to False when key is missing
                self.use_openai_for_judging = False
            else:
                # Key exists, set to use OpenAI
                self.use_openai_for_judging = True
                print(f"Found OpenAI API key: {self.openai_api_key[:4]}...{self.openai_api_key[-4:] if self.openai_api_key and len(self.openai_api_key) > 8 else '****'}")
        else:
            # Config explicitly disabled OpenAI judging
            self.openai_api_key = None
            self.use_openai_for_judging = False
            print("OpenAI judging disabled by configuration - using local model for judgments")

        # Set the OpenAI model to use
        self.openai_model = getattr(config, 'openai_model', os.environ.get("OAI_MODEL") or "gpt-4.1-nano")
        print(f"Using OpenAI model: {self.openai_model}")

        # Initialize the OpenAI client as an instance attribute
        self.openai_client = None
        if self.use_openai_for_judging:
            try:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=30.0,
                )
                print(
                    f"OpenAI client successfully initialized for model {self.openai_model} with 3 retries"
                )
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.use_openai_for_judging = False
                print("Falling back to local model for equivalency judgments")
        else:
            print("Using local model for equivalency judgments (OpenAI not configured)")
            
        # Final status of judge
        print(f"FINAL JUDGE STATUS: {'Using OpenAI' if self.use_openai_for_judging else 'Using local model'}")

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[OpenaiConfig]]:
        env_config = LMJudgeConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="textbook_questions_judge",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            use_openai_judge=True,
            openai_model="gpt-4.1-nano"
        )
        server_configs = [
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading and preparing the dataset.
        """
        # Load the TextbooksToRLQuestions-100k dataset
        full_dataset = load_dataset(
            "TamasSimonds/TextbooksToRLQuestions-100k", split="train"
        )

        # full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(
            test_size=0.005, seed=42, shuffle=False
        )

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        # Print some dataset statistics
        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

        # Initialize iteration counter
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract question and solution from the dataset item
        question_text = next_item["question"]
        solution = next_item["solution"]

        # Create prompt tuple using frozensets as required
        prompt = []

        # Add system prompt as defined at the top of the script
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))

        # Add user message with the question
        prompt.append(frozenset({"role": "user", "content": question_text}.items()))

        return (tuple(prompt), solution, question_text)

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        """
        Generate and collect model responses for scoring.

        Args:
            item: Input item containing prompt and expected answer

        Returns:
            Tuple of lists containing items to score and backlog
        """
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model using completion() instead of chat_completion()
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=1024 * 16,
            temperature=1.0,  # Using temperature to get diverse responses
        )

        to_score = list()
        to_backlog = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer, original question, and stop reason
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # Solution (ground truth)
                    item[2],  # Original question
                    completion_choice.finish_reason,  # Add the stop reason
                )
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def is_equivalent_answer(
        self, question: str, user_answer: str, ground_truth: str, is_eval: bool = False
    ) -> bool:
        """
        Uses an LLM judge to determine if the user's answer is equivalent to the ground truth answer.
        Can use either OpenAI API or local model depending on configuration.

        Returns True if the human answer is equivalent, otherwise False.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI evaluator. Your task is to decide if a human-provided answer is equivalent "
                    "to the given ground truth answer. When evaluating, consider the following rules:\n\n"
                    "1. Semantic Equivalence: Accept if the human answer conveys the same meaning as the ground truth, "
                    "even if reworded.\n"
                    "2. Numerical Reasoning: Consider mathematically equivalent answers as the same "
                    "(e.g., 0.25 is equal to 1/4 or \\Boxed{1.5} is equal to 1.5).\n"
                    "3. Context Tolerance: Ignore extra context, conversational text, or fluff as long as the "
                    "correct answer is present.\n"
                    "4. Formatting and Speech Constraints: Also consider the original question's instructions. "
                    "If the question requires a specific format or restricts certain responses, "
                    "the human answer must adhere to these constraints.\n\n"
                    "Only respond with 'True' if the user's answer is equivalent to the ground truth and meets "
                    "the question constraints, or 'False' if it is not. Do not include any additional text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Ground truth answer: {ground_truth}\n\n"
                    f"User answer: {user_answer}\n\n"
                    "Is the user's answer equivalent to the ground truth "
                    "(and compliant with any instructions in the original question)? "
                    "Answer True or False."
                ),
            },
        ]

        # For debugging in eval, print judge status
        if is_eval:
            print(f"JUDGE STATUS CHECK - OpenAI enabled: {self.use_openai_for_judging}, OpenAI client initialized: {self.openai_client is not None}")

        # Use OpenAI API if configured and enabled
        if self.use_openai_for_judging and self.openai_client is not None:
            try:
                if is_eval:
                    print(f"USING OPENAI API ({self.openai_model}) FOR EQUIVALENCE JUDGE...")
                # Call the OpenAI API using the instance client
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent judgments
                    max_completion_tokens=5000,  # Only need a few tokens for True/False
                )

                # Extract result
                result_text = response.choices[0].message.content.strip()
                if is_eval:
                    print(f"OPENAI JUDGE RESULT: {result_text}")
                return result_text.lower() == "true"

            except Exception as e:
                if is_eval:
                    print(f"ERROR USING OPENAI API: {str(e)}")
                    print(f"API KEY: {self.openai_api_key[:4]}...{self.openai_api_key[-4:] if self.openai_api_key and len(self.openai_api_key) > 8 else '****'}")
                    print(f"MODEL: {self.openai_model}")
                else:
                    print(f"Error using OpenAI API: {e}")
                
                print("Falling back to local model for judgment")
                # Don't permanently disable OpenAI judging, just fall back for this request

        # Use local server if OpenAI is not configured, explicitly disabled, or on fallback
        if is_eval:
            if not self.use_openai_for_judging:
                print("USING LOCAL MODEL BECAUSE OPENAI JUDGING IS DISABLED BY CONFIGURATION")
            elif self.openai_client is None:
                print("USING LOCAL MODEL BECAUSE OPENAI CLIENT IS NOT CONFIGURED")
            else:
                print("USING LOCAL MODEL BECAUSE OPENAI API CALL FAILED")
            
        if is_eval:
            print("LOCAL MODEL MESSAGES:")
            for msg in messages:
                print(f"  {msg['role'].upper()}: {msg['content'][:100]}...")
            
        completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=16000,
            temperature=0.1,
        )
        
        if is_eval:
            if hasattr(completion, 'choices') and completion.choices:
                result = completion.choices[0].message.content.strip() if hasattr(completion.choices[0], 'message') else "UNKNOWN"
                print(f"LOCAL MODEL RESULT: {result}")
            else:
                print(f"LOCAL MODEL RETURNED UNEXPECTED FORMAT: {completion}")

        # Extract the result and convert it to a boolean
        result_text = completion.choices[0].message.content.strip()
        is_true = result_text.lower() == "true"
        
        if is_eval:
            print(f"FINAL JUDGE DECISION: {'TRUE' if is_true else 'FALSE'}")
            
        return is_true

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Score the generated model responses by comparing to ground truth using an LLM judge.

        Args:
            rollout_group_data: List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]
            ground_truth = item[1]  # Solution/ground truth
            question = item[2]  # Original question
            stop_reason = item[3]  # Get the stop reason

            # If the response was cut off due to length, give it a score of 0
            if stop_reason == "length":
                reward = 0
            else:
                # Check if there's exactly one proper set of think tags
                # Count complete <think></think> pairs
                think_tag_pairs = re.findall(
                    r"<think>.*?</think>",
                    model_response,
                    re.DOTALL | re.IGNORECASE
                )
                
                has_exactly_one_proper_think_tag = len(think_tag_pairs) == 1
                
                # Check for malformed tags (incomplete closing tag)
                has_malformed_tags = re.search(
                    r"<think>(?:(?!</think>).)*$",  # <think> without matching </think>
                    model_response,
                    re.DOTALL | re.IGNORECASE
                ) is not None
                
                # Extract thinking and answer sections
                think_match = re.search(
                    r"<think>(.*?)</think>",
                    model_response,
                    re.DOTALL | re.IGNORECASE,
                )

                if think_match and has_exactly_one_proper_think_tag and not has_malformed_tags:
                    # If there's a properly formatted thinking section, use the non-thinking part for evaluation
                    # Get everything after the properly closed </think> tag
                    answer_section = re.sub(r".*?</think>", "", model_response, flags=re.DOTALL | re.IGNORECASE).strip()
                    
                    # Use LLM judge to evaluate the answer
                    is_equivalent = await self.is_equivalent_answer(
                        question, answer_section, ground_truth
                    )
                    reward = 1.0 if is_equivalent else 0.0
                else:
                    # If missing or improper think tags, give a score of 0
                    reward = 0.0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics for wandb logging
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Return None if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Score (1 for correct, 0 for incorrect)
        """
        # Extract question and solution from the test item
        question_text = test_item["question"]
        ground_truth = test_item["solution"]

        # Create messages for model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ]

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        print("\n" + "="*80)
        print("BEGINNING EVALUATION ON SAMPLE QUESTION:")
        print(f"QUESTION: {question_text}")
        print(f"GROUND TRUTH: {ground_truth}")
        
        # Get model completion
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024 * 16,
            temperature=0.2,  # Lower for eval
            split="train",
        )
        
        # Extract the model's response from the completion
        model_response = completion.choices[0].text
        print(f"MODEL RESPONSE: {model_response}")

        # Check if there's exactly one proper set of think tags
        # Count complete <think></think> pairs
        think_tag_pairs = re.findall(
            r"<think>.*?</think>",
            model_response,
            re.DOTALL | re.IGNORECASE
        )
        
        has_exactly_one_proper_think_tag = len(think_tag_pairs) == 1
        
        # Check for malformed tags (incomplete closing tag)
        has_malformed_tags = re.search(
            r"<think>(?:(?!</think>).)*$",  # <think> without matching </think>
            model_response,
            re.DOTALL | re.IGNORECASE
        ) is not None
        
        # Extract thinking and answer sections
        think_match = re.search(
            r"<think>(.*?)</think>",
            model_response,
            re.DOTALL | re.IGNORECASE,
        )

        if think_match and has_exactly_one_proper_think_tag and not has_malformed_tags:
            # If there's a properly formatted thinking section, use the non-thinking part for evaluation
            # Get everything after the properly closed </think> tag
            answer_section = re.sub(r".*?</think>", "", model_response, flags=re.DOTALL | re.IGNORECASE).strip()
            print("THINK TAG DETECTED: Extracting answer after think tag")
        else:
            # If missing or improper think tags, we still want to evaluate the full response in eval mode
            # to see if the model got the right answer, but log a warning
            answer_section = model_response
            if has_malformed_tags:
                print("WARNING: Malformed think tags detected. Using full response for evaluation.")
            elif len(think_tag_pairs) > 1:
                print(f"WARNING: Multiple think tags detected ({len(think_tag_pairs)}). Using full response for evaluation.")
            else:
                print("WARNING: No think tags detected. Using full response for evaluation.")

        print(f"EXTRACTED ANSWER SECTION: {answer_section}")
        print("EQUIVALENCE JUDGE STARTING...")
        
        # Use LLM judge to evaluate the answer - passing is_eval=True to provide detailed logs
        is_equivalent = await self.is_equivalent_answer(
            question_text, answer_section, ground_truth, is_eval=True
        )
        
        print(f"EQUIVALENCE JUDGE RESULT: {'CORRECT' if is_equivalent else 'INCORRECT'}")
        print("="*80 + "\n")

        # Return 1 for equivalent answer, 0 otherwise
        return 1 if is_equivalent else 0

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        # Run evaluation
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # ground truth solution
                    item[2],  # original question
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "ground_truth", "question"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2], item[3])
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

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

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    LLMEquivalencyJudgeEnv.cli()
