"""
SimpleQA Evaluation Environment for Atropos (Generative with String Matching)

This environment evaluates models on the SimpleQA benchmark - a factuality
benchmark that measures the ability for language models to answer short,
fact-seeking questions.

Dataset: lighteval/SimpleQA (originally from OpenAI)
Paper: https://openai.com/index/introducing-simpleqa/

DEFAULT MODE (Nous Research style):
- Uses exact match and fuzzy match string comparisons
- No LLM judge needed - faster and cheaper to run
- Exact match: normalized gold answer == normalized prediction
- Fuzzy match: gold answer tokens contained in prediction

ALTERNATIVE MODE (Original OpenAI style):
- Uses GPT-4o as LLM judge
- Grades: CORRECT, INCORRECT, NOT_ATTEMPTED
- More nuanced but slower and requires API costs

Supports optional thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import openai
import wandb
from datasets import load_dataset
from eval_helpers import (
    ANSWER_TAG_PATTERN,
    create_system_content,
    extract_thinking_content,
    get_default_thinking_prompt,
    save_eval_results,
    validate_thinking_format,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)

# SimpleQA prompt template - Nous style with <answer> tags
SIMPLEQA_PROMPT_TEMPLATE = """Please provide your answer within <answer></answer> tags. Give a concise, accurate answer.

Example format:
<answer>John Doe</answer>

Question: {question}"""


# LLM Judge grading template - identical to lighteval's GRADER_TEMPLATE (for optional judge mode)
SIMPLEQA_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
- Predicted answers "120k", "124k", and 115k" are all CORRECT.
- Predicted answers "100k" and "113k" are INCORRECT.
- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
- For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
- For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
- Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
- For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
- For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""


class SimpleQAEvalConfig(BaseEnvConfig):
    """Configuration for SimpleQA evaluation environment."""

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
        default="lighteval/SimpleQA",
        description="HuggingFace dataset name for SimpleQA.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation.",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for model answer generation.",
    )

    eval_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for evaluation responses. Set to 0 for provider default.",
    )

    # Scoring mode configuration
    use_llm_judge: bool = Field(
        default=False,
        description="If True, use LLM judge (GPT-4o style). If False, use exact/fuzzy string matching (Nous style).",
    )

    # Judge model configuration (when use_llm_judge=True)
    judge_model_name: str = Field(
        default="gpt-4o-2024-08-06",
        description="Model name for the judge. Official SimpleQA uses 'gpt-4o-2024-08-06'.",
    )

    judge_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the judge model API. Official SimpleQA uses OpenAI.",
    )

    judge_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name containing the API key for the judge model.",
    )

    judge_temperature: float = Field(
        default=0.0,
        description="Temperature for judge model (should be 0 for consistency).",
    )

    judge_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for judge response (0 = use model default).",
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


class SimpleQAEvalEnv(BaseEnv):
    """
    SimpleQA Evaluation Environment for Atropos.

    Evaluates models on the SimpleQA factuality benchmark.

    Two scoring modes:
    1. String Matching (default, Nous style): Uses exact/fuzzy match - fast, no LLM needed
    2. LLM Judge (optional, OpenAI style): Uses GPT-4o judge - more nuanced but slower

    Key features:
    - Loads SimpleQA dataset from HuggingFace (lighteval/SimpleQA)
    - Open-ended question answering (not multiple choice)
    - Optional thinking mode with <think></think> tags
    - Tracks exact match, fuzzy match, and combined accuracy
    """

    name = "simpleqa_eval"
    env_config_cls = SimpleQAEvalConfig

    def __init__(
        self,
        config: SimpleQAEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: SimpleQAEvalConfig = config

        # Initialize metrics tracking
        self.eval_metrics = []

        # Pre-compile regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Pre-compile regex patterns for <answer></answer> tag extraction
        self._answer_tag_pattern = re.compile(
            r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
        )

        # Initialize judge client only if using LLM judge mode
        self.judge_client = None
        if self.config.use_llm_judge:
            judge_api_key = os.environ.get(self.config.judge_api_key_env)
            if not judge_api_key:
                raise ValueError(
                    f"Judge API key not found in environment variable: {self.config.judge_api_key_env}. "
                    f"Set this env var or use --env.use_llm_judge False"
                )
            self.judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=self.config.judge_base_url,
            )

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

    # ==================== String Matching Functions (Nous style) ====================

    def _exact_match(self, gold: str, prediction: str) -> bool:
        """
        Evaluate open-ended answer using exact match (case-insensitive).
        Matches: evaluate_open_ended_exact_match from Nous lighteval.
        """
        if not prediction:
            return False

        return prediction.lower().strip() == gold.lower().strip()

    def _fuzzy_match(self, gold: str, prediction: str) -> bool:
        """
        Evaluate open-ended answer using fuzzy matching.
        Returns True if prediction contains the ground truth or vice versa (case-insensitive).
        Matches: evaluate_open_ended_fuzzy_match from Nous lighteval.
        """
        if not prediction:
            return False

        pred_lower = prediction.lower().strip()
        truth_lower = gold.lower().strip()

        # Check if either string contains the other
        return truth_lower in pred_lower or pred_lower in truth_lower

    def _score_string_match(self, gold: str, prediction: str) -> Dict:
        """
        Score a prediction using Nous-style string matching methods.

        Returns dict with:
        - exact_match: bool (case-insensitive exact match)
        - fuzzy_match: bool (containment in either direction)
        """
        exact = self._exact_match(gold, prediction)
        fuzzy = self._fuzzy_match(gold, prediction)

        return {
            "exact_match": exact,
            "fuzzy_match": fuzzy,
        }

    # ==================== LLM Judge Functions (OpenAI style) ====================

    def _format_judge_prompt(
        self, question: str, gold_answer: str, predicted_answer: str
    ) -> str:
        """Format the judge prompt using the lighteval grading template."""
        return SIMPLEQA_GRADER_TEMPLATE.format(
            question=question,
            target=gold_answer,
            predicted_answer=predicted_answer,
        )

    def _parse_judge_grade(self, judge_response: str) -> Tuple[str, float]:
        """
        Parse the judge's grade from their response.

        Returns:
            Tuple of (grade_string, score)
            - CORRECT: 1.0
            - INCORRECT: 0.0
            - NOT_ATTEMPTED: 0.0 (but tracked separately)
        """
        response = judge_response.strip().upper()

        # Direct match
        if response == "A":
            return "CORRECT", 1.0
        elif response == "B":
            return "INCORRECT", 0.0
        elif response == "C":
            return "NOT_ATTEMPTED", 0.0

        # Try to find A, B, or C in the response
        if "A" in response and "B" not in response and "C" not in response:
            return "CORRECT", 1.0
        elif "B" in response and "A" not in response and "C" not in response:
            return "INCORRECT", 0.0
        elif "C" in response and "A" not in response and "B" not in response:
            return "NOT_ATTEMPTED", 0.0

        # Check for text matches
        if (
            "CORRECT" in response
            and "INCORRECT" not in response
            and "NOT" not in response
        ):
            return "CORRECT", 1.0
        elif "INCORRECT" in response:
            return "INCORRECT", 0.0
        elif "NOT_ATTEMPTED" in response or "NOT ATTEMPTED" in response:
            return "NOT_ATTEMPTED", 0.0

        # Unable to parse - treat as incorrect
        return "PARSE_ERROR", 0.0

    # ==================== Core Evaluation Functions ====================

    @classmethod
    def config_init(cls) -> Tuple[SimpleQAEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = SimpleQAEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="simpleqa_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            # SimpleQA specific defaults
            dataset_name="lighteval/SimpleQA",
            eval_temperature=0.6,
            eval_max_tokens=0,
            thinking_mode=True,
            # Default to string matching (Nous style)
            use_llm_judge=False,
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
        """Load the SimpleQA dataset and prepare for evaluation."""
        scoring_mode = (
            "LLM Judge (GPT-4o)"
            if self.config.use_llm_judge
            else "String Matching (Nous)"
        )

        print("\nSimpleQA Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Scoring mode: {scoring_mode}")
        print(f"  Max tokens for answer: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.use_llm_judge:
            print(
                f"  Judge model: {self.config.judge_model_name} @ {self.config.judge_base_url}"
            )
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")

        # Load SimpleQA dataset
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

        except Exception as e:
            print(f"Error loading SimpleQA dataset: {e}")
            raise

        self.all_eval_items = self.eval_data
        self.iter = 0

    def _validate_thinking_format(self, response: str) -> Tuple[bool, str]:
        """Validate thinking format. Returns (is_valid, content_after_think)."""
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

    def _extract_answer_tag(self, response: str) -> Optional[str]:
        """
        Extract the content inside <answer></answer> tags.
        This is the primary method for getting the model's answer.
        """
        match = self._answer_tag_pattern.search(response)
        if match:
            return match.group(1).strip()
        return None

    def _extract_answer_for_scoring(self, response: str) -> Tuple[str, bool, bool]:
        """
        Extract the answer to use for scoring from the model response.

        Handles both thinking mode and answer tags:
        - If thinking mode: extract content after </think>
        - Then extract content from <answer></answer> tags

        Returns:
            Tuple of (answer_text, thinking_format_valid, answer_tag_found)
        """
        # First, handle thinking mode
        thinking_format_valid = True
        content_after_think = response

        if self.config.thinking_mode:
            thinking_format_valid, content_after_think = self._validate_thinking_format(
                response
            )

        # Now extract from <answer> tags
        answer_content = self._extract_answer_tag(content_after_think)

        if answer_content is not None:
            return answer_content, thinking_format_valid, True

        # Fallback: if no answer tags, use content after think (or full response)
        return content_after_think, thinking_format_valid, False

    def _format_simpleqa_prompt(self, question: str) -> str:
        """Format a question using the SimpleQA template."""
        return SIMPLEQA_PROMPT_TEMPLATE.format(question=question)

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
        """Evaluate a single SimpleQA question."""
        try:
            # SimpleQA uses 'problem' for question and 'answer' for gold
            question = eval_item.get("problem", "")
            gold_answer = eval_item.get("answer", "")
            metadata = eval_item.get("metadata", {})
            topic = (
                metadata.get("topic", "unknown")
                if isinstance(metadata, dict)
                else "unknown"
            )

            if not question or not gold_answer:
                return {"score": None, "sample": None}

            # Format the prompt
            formatted_prompt = self._format_simpleqa_prompt(question)

            # Build messages for model
            messages = []
            system_content = self._create_system_content()
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": formatted_prompt})

            # Get model answer with retry logic
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

                    completion = await self.server.chat_completion(**completion_kwargs)

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
                        return {"score": None, "sample": None}

            if not model_response:
                return {"score": None, "sample": None}

            # Extract answer using the combined thinking + answer tag extraction
            answer_for_scoring, thinking_format_valid, answer_tag_found = (
                self._extract_answer_for_scoring(model_response)
            )

            # Extract thinking content for logging
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)

            # Score the response based on mode
            if self.config.use_llm_judge:
                # LLM Judge mode
                result = await self._score_with_judge(
                    question, gold_answer, answer_for_scoring
                )
                if result is None:
                    return {"score": None, "sample": None}

                sample = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_response": model_response,
                    "answer_for_scoring": answer_for_scoring,
                    "judge_response": result["judge_response"],
                    "grade": result["grade"],
                    "score": result["score"],
                    "topic": topic,
                    "finish_reason": finish_reason,
                    "response_length": len(model_response),
                    "thinking_mode": self.config.thinking_mode,
                    "thinking_format_valid": thinking_format_valid,
                    "answer_tag_found": answer_tag_found,
                    "scoring_mode": "llm_judge",
                }

                if self.config.thinking_mode:
                    sample["thinking_content"] = thinking_content

                if self.config.full_debug:
                    status = (
                        "✓"
                        if result["grade"] == "CORRECT"
                        else ("○" if result["grade"] == "NOT_ATTEMPTED" else "✗")
                    )
                    print(f"  [{status}] {topic[:20]}: {result['grade']}")

                return {"score": result["score"], "sample": sample}

            else:
                # String matching mode (Nous style)
                match_results = self._score_string_match(
                    gold_answer, answer_for_scoring
                )

                # Score is 1.0 if either exact or fuzzy match
                is_correct = (
                    match_results["exact_match"] or match_results["fuzzy_match"]
                )

                sample = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_response": model_response,
                    "answer_for_scoring": answer_for_scoring,
                    "exact_match": match_results["exact_match"],
                    "fuzzy_match": match_results["fuzzy_match"],
                    "is_correct": is_correct,
                    "score": 1.0 if is_correct else 0.0,
                    "topic": topic,
                    "finish_reason": finish_reason,
                    "response_length": len(model_response),
                    "thinking_mode": self.config.thinking_mode,
                    "thinking_format_valid": thinking_format_valid,
                    "answer_tag_found": answer_tag_found,
                    "scoring_mode": "string_match",
                }

                if self.config.thinking_mode:
                    sample["thinking_content"] = thinking_content

                if self.config.full_debug:
                    status = "✓" if is_correct else "✗"
                    print(
                        f"  [{status}] {topic[:20]}: exact={match_results['exact_match']}, fuzzy={match_results['fuzzy_match']}"
                    )

                return {"score": 1.0 if is_correct else 0.0, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"score": None, "sample": None}

    async def _score_with_judge(
        self, question: str, gold_answer: str, prediction: str
    ) -> Optional[Dict]:
        """Score using LLM judge."""
        judge_prompt = self._format_judge_prompt(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=prediction,
        )

        judge_messages = [{"role": "user", "content": judge_prompt}]

        for attempt in range(self.config.max_retries):
            try:
                kwargs = {
                    "model": self.config.judge_model_name,
                    "messages": judge_messages,
                    "temperature": self.config.judge_temperature,
                }
                if self.config.judge_max_tokens > 0:
                    kwargs["max_tokens"] = self.config.judge_max_tokens

                judge_completion = await self.judge_client.chat.completions.create(
                    **kwargs
                )

                if (
                    judge_completion.choices
                    and judge_completion.choices[0].message.content
                ):
                    judge_response = judge_completion.choices[0].message.content
                    grade, score = self._parse_judge_grade(judge_response)
                    return {
                        "judge_response": judge_response,
                        "grade": grade,
                        "score": score,
                    }

            except Exception as e:
                print(
                    f"  Judge Error (attempt {attempt + 1}/{self.config.max_retries}): {type(e).__name__}: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    print(f"  Judge failed after {self.config.max_retries} attempts")
                    return None

        return None

    async def evaluate(self, *args, **kwargs) -> None:
        """Run SimpleQA evaluation."""
        start_time = time.time()

        scoring_mode = (
            "LLM Judge (GPT-4o)"
            if self.config.use_llm_judge
            else "String Matching (Nous)"
        )

        print(f"\n{'='*60}")
        print("Starting SimpleQA Evaluation")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Scoring mode: {scoring_mode}")
        print(f"  Max tokens (for answer): {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating SimpleQA")

            valid_results = [
                r
                for r in results
                if r and r.get("sample") is not None and r.get("score") is not None
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

        # Build metrics based on scoring mode
        eval_metrics = {
            "eval/total_questions": total_count,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
            "eval/scoring_mode": (
                1.0 if self.config.use_llm_judge else 0.0
            ),  # 1=judge, 0=string
        }

        if self.config.use_llm_judge:
            # LLM Judge metrics
            correct_count = sum(1 for s in samples if s.get("grade") == "CORRECT")
            incorrect_count = sum(1 for s in samples if s.get("grade") == "INCORRECT")
            not_attempted_count = sum(
                1 for s in samples if s.get("grade") == "NOT_ATTEMPTED"
            )
            parse_error_count = sum(
                1 for s in samples if s.get("grade") == "PARSE_ERROR"
            )

            accuracy = correct_count / total_count if total_count > 0 else 0.0
            attempted_count = correct_count + incorrect_count
            accuracy_if_attempted = (
                correct_count / attempted_count if attempted_count > 0 else 0.0
            )
            not_attempted_rate = (
                not_attempted_count / total_count if total_count > 0 else 0.0
            )

            eval_metrics.update(
                {
                    "eval/accuracy": accuracy,
                    "eval/accuracy_if_attempted": accuracy_if_attempted,
                    "eval/not_attempted_rate": not_attempted_rate,
                    "eval/correct_count": correct_count,
                    "eval/incorrect_count": incorrect_count,
                    "eval/not_attempted_count": not_attempted_count,
                    "eval/parse_error_count": parse_error_count,
                }
            )
        else:
            # String matching metrics (Nous style)
            exact_match_count = sum(1 for s in samples if s.get("exact_match", False))
            fuzzy_match_count = sum(1 for s in samples if s.get("fuzzy_match", False))
            correct_count = sum(1 for s in samples if s.get("is_correct", False))

            exact_match_rate = (
                exact_match_count / total_count if total_count > 0 else 0.0
            )
            fuzzy_match_rate = (
                fuzzy_match_count / total_count if total_count > 0 else 0.0
            )
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            eval_metrics.update(
                {
                    "eval/accuracy": accuracy,
                    "eval/exact_match_accuracy": exact_match_rate,
                    "eval/fuzzy_match_accuracy": fuzzy_match_rate,
                    "eval/correct_count": correct_count,
                    "eval/exact_match_count": exact_match_count,
                    "eval/fuzzy_match_count": fuzzy_match_count,
                }
            )

        # Per-topic accuracy
        topic_results = {}
        for sample in samples:
            topic = sample.get("topic", "unknown")
            if topic not in topic_results:
                topic_results[topic] = {"correct": 0, "total": 0}
            topic_results[topic]["total"] += 1

            if self.config.use_llm_judge:
                if sample.get("grade") == "CORRECT":
                    topic_results[topic]["correct"] += 1
            else:
                if sample.get("is_correct", False):
                    topic_results[topic]["correct"] += 1

        # Average response length
        response_lengths = [s.get("response_length", 0) for s in samples]
        avg_response_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )
        eval_metrics["eval/avg_response_length"] = avg_response_length

        # Answer tag usage (primary format indicator for SimpleQA Nous)
        answer_tag_found_count = sum(
            1 for s in samples if s.get("answer_tag_found", False)
        )
        answer_tag_rate = answer_tag_found_count / len(samples) if samples else 0.0
        eval_metrics["eval/answer_tag_rate"] = answer_tag_rate
        eval_metrics["eval/answer_tag_found_count"] = answer_tag_found_count

        # Thinking format compliance (for thinking mode)
        if self.config.thinking_mode:
            thinking_format_compliant = sum(
                1 for s in samples if s.get("thinking_format_valid", True)
            )
            thinking_format_compliance_rate = (
                thinking_format_compliant / len(samples) if samples else 0.0
            )
            eval_metrics["eval/thinking_format_compliance_rate"] = (
                thinking_format_compliance_rate
            )

            thinking_utilization = sum(1 for s in samples if s.get("thinking_content"))
            thinking_utilization_rate = (
                thinking_utilization / len(samples) if samples else 0.0
            )
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization_rate

        # Add top topic metrics
        sorted_topics = sorted(topic_results.items(), key=lambda x: -x[1]["total"])[:20]
        for topic, stats in sorted_topics:
            if stats["total"] > 0:
                topic_accuracy = stats["correct"] / stats["total"]
                topic_key = topic.replace(" ", "_").replace("-", "_").lower()[:30]
                eval_metrics[f"eval/topic_{topic_key}_accuracy"] = topic_accuracy

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print(f"SimpleQA Evaluation Results ({scoring_mode})")
        print(f"{'='*60}")

        if self.config.use_llm_judge:
            print(
                f"Overall Accuracy: {eval_metrics['eval/accuracy']:.4f} ({correct_count}/{total_count})"
            )
            print(
                f"Accuracy (if attempted): {eval_metrics['eval/accuracy_if_attempted']:.4f}"
            )
            print(f"Not Attempted Rate: {eval_metrics['eval/not_attempted_rate']:.4f}")
            print("\nGrade Distribution:")
            print(f"  CORRECT: {correct_count} ({100*correct_count/total_count:.1f}%)")
            print(
                f"  INCORRECT: {incorrect_count} ({100*incorrect_count/total_count:.1f}%)"
            )
            print(
                f"  NOT_ATTEMPTED: {not_attempted_count} ({100*not_attempted_count/total_count:.1f}%)"
            )
        else:
            print(
                f"Overall Accuracy: {eval_metrics['eval/accuracy']:.4f} ({correct_count}/{total_count})"
            )
            print(
                f"Exact Match Accuracy: {eval_metrics['eval/exact_match_accuracy']:.4f} ({exact_match_count}/{total_count})"
            )
            print(
                f"Fuzzy Match Accuracy: {eval_metrics['eval/fuzzy_match_accuracy']:.4f} ({fuzzy_match_count}/{total_count})"
            )

        print(f"\nEvaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Avg Response Length: {avg_response_length:.0f} chars")
        print(
            f"Answer Tag Rate: {answer_tag_rate:.4f} ({answer_tag_found_count}/{total_count})"
        )
        if self.config.thinking_mode:
            print(f"Thinking Format Compliance: {thinking_format_compliance_rate:.4f}")
            print(f"Thinking Utilization: {thinking_utilization}/{total_count}")

        if len(sorted_topics) > 0:
            print("\nTop Topics (by count):")
            for topic, stats in sorted_topics[:10]:
                if stats["total"] > 0:
                    topic_acc = stats["correct"] / stats["total"]
                    print(
                        f"  {topic}: {topic_acc:.4f} ({stats['correct']}/{stats['total']})"
                    )

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
                    "scoring_mode": (
                        "llm_judge" if self.config.use_llm_judge else "string_match"
                    ),
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
        wandb_metrics["config/use_llm_judge"] = (
            1.0 if self.config.use_llm_judge else 0.0
        )

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    SimpleQAEvalEnv.cli()
