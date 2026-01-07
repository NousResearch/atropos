"""
NL2Bash Generation Environment for Atropos

Trains LLMs to translate natural language instructions into Bash commands.
Uses the NL2SH-ALFA dataset (NAACL 2025) with string-based verification.
"""

import random
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from bash_utils import commands_match, extract_boxed_bash
from nl2bash_loader import load_nl2bash_split
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

# System prompt following the established Atropos pattern
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are a Bash command expert. Given a natural language instruction,
generate the appropriate Bash command.

You are allocated a maximum of 1024 tokens, please strive to use less.

Provide your Bash command inside \\boxed{} like this: \\boxed{find . -name "*.txt"}

Important:
- Generate a single, complete Bash command
- Do not include explanatory text outside of <think> tags
- Ensure your command is valid Bash syntax

So please end your answer with \\boxed{your bash command here}"""


class NL2BashItem(TypedDict):
    """Type definition for a NL2Bash dataset item."""

    nl: str
    bash: str
    bash2: Optional[str]
    difficulty: Optional[int]


def format_instruction(nl: str) -> str:
    """Format the natural language instruction for the prompt."""
    return f"Instruction: {nl}"


class BashEnv(BaseEnv):
    """
    Environment for training LLMs to generate Bash commands.

    Uses the NL2SH-ALFA dataset and verifies correctness
    by string matching against gold commands.
    """

    name = "nl2bash"

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
        # Track accuracy by difficulty level (0=easy, 1=medium, 2=hard)
        self.difficulty_correct = {0: [], 1: [], 2: []}

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=1024,
            wandb_name="nl2bash",
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
        """Log custom metrics to WandB."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log percent correct
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = list()

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Load the NL2SH-ALFA dataset and prepare train/test splits."""
        # Load training data
        print("Loading NL2SH-ALFA training data...")
        self.train = load_nl2bash_split("train")
        print(f"Loaded {len(self.train)} training examples")

        # Load test data
        print("Loading NL2SH-ALFA test data...")
        self.test = load_nl2bash_split("test")
        print(f"Loaded {len(self.test)} test examples")

        random.shuffle(self.train)
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _score_bash(
        self,
        generated_bash: str,
        gold_bash: str,
        alt_bash: Optional[str] = None,
    ) -> float:
        """
        Score generated Bash command by string matching.

        Returns:
            1.0 if command matches gold or alternative
            -1.0 if incorrect or malformed
        """
        if not generated_bash:
            return -1.0

        if commands_match(generated_bash, gold_bash, alt_bash):
            return 1.0
        else:
            return -1.0

    async def rollout_and_score_eval(
        self,
        nl: str,
        gold_bash: str,
        alt_bash: Optional[str],
        difficulty: Optional[int],
    ) -> dict:
        """Rollout and score a single evaluation item."""
        user_content = format_instruction(nl)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.6,
            )
            response_content = completion.choices[0].message.content

        # Extract and score generated Bash
        generated_bash = extract_boxed_bash(response_content)
        score = self._score_bash(generated_bash, gold_bash, alt_bash)
        correct = score == 1.0

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_content},
            ],
            "instruction": nl,
            "gold_bash": gold_bash,
            "alt_bash": alt_bash,
            "generated_bash": generated_bash,
            "score": 1 if correct else 0,
            "correct": correct,
            "difficulty": difficulty,
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {
            "score": 1 if correct else 0,
            "sample": sample,
            "difficulty": difficulty,
        }

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set."""
        import time

        start_time = time.time()

        eval_tasks = []
        # Evaluate on all 300 test items (small enough to do full eval)
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(
                    item["nl"],
                    item["bash"],
                    item.get("bash2"),
                    item.get("difficulty"),
                )
            )
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores) if scores else 0

        # Calculate difficulty-stratified accuracy
        difficulty_scores = {0: [], 1: [], 2: []}
        for result in results:
            diff = result.get("difficulty")
            if diff is not None and diff in difficulty_scores:
                difficulty_scores[diff].append(result["score"])

        end_time = time.time()

        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

        # Add difficulty-stratified metrics
        difficulty_names = {0: "easy", 1: "medium", 2: "hard"}
        for diff, name in difficulty_names.items():
            if difficulty_scores[diff]:
                accuracy = sum(difficulty_scores[diff]) / len(difficulty_scores[diff])
                eval_metrics[f"eval/accuracy_{name}"] = accuracy
                self.eval_metrics.append((f"eval/accuracy_{name}", accuracy))

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.6,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(
        self, item: NL2BashItem
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        """Generate Bash commands for a given instruction."""
        user_content = format_instruction(item["nl"])
        user_message = {"role": "user", "content": user_content}

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[{"role": "system", "content": system_prompt}, user_message],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )

            try:
                state = managed.get_state()
                nodes = state["nodes"]
            except AttributeError:
                # Fallback for OpenAIServer which doesn't track state
                nodes = []
                for choice in chat_completions.choices:
                    content = choice.message.content
                    if self.tokenizer:
                        tokens = self.tokenizer.encode(content)

                        # Create dummy node-like object
                        class Node:
                            def __init__(self, t):
                                self.tokens = t
                                self.masked_tokens = t
                                self.logprobs = [0.0] * len(t)

                        nodes.append(Node(tokens))
                    else:
                        nodes.append(None)

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = [
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            ]
            to_score.append(
                {
                    "messages": messages,
                    "gold_bash": item["bash"],
                    "alt_bash": item.get("bash2"),
                    "finish_reason": chat_completion.finish_reason,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score generated Bash commands by string matching."""
        scores = ScoredDataGroup()

        # If all scores are the same, return None (no training signal)
        # if len(set(scores["scores"])) == 1:
        #     return None

        # Add messages to scores to avoid reconstruction from tokens
        scores["messages"] = [
            item["messages"]
            for item in rollout_group_data
            if len([1 for i in item["masks"] if i != -100]) >= 10
        ]
        # Align messages with the filtered tokens/scores
        # Note: The loop above filtered items < 10 masks.
        # We need to ensure messages list matches tokens list length and order

        # Redo the loop to be safe and cleaner
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()
        scores["messages"] = list()

        # Get gold info from first item (all items in group have same gold)
        gold_bash = rollout_group_data[0]["gold_bash"]
        alt_bash = rollout_group_data[0].get("alt_bash")

        for item in rollout_group_data:
            response_content = item["messages"][-1]["content"]
            generated_bash = extract_boxed_bash(response_content)
            reward = self._score_bash(generated_bash, gold_bash, alt_bash)

            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]

            # Remove obviously bad examples (very short)
            # if len([1 for i in masks if i != -100]) < 10:
            #     continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(reward)
            scores["messages"].append(item["messages"])

            if len(scores["tokens"]) >= self.config.group_size:
                break

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scores

    async def get_next_item(self) -> NL2BashItem:
        """Get the next training item."""
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    BashEnv.cli()
