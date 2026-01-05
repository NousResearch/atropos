"""
SQL Query Generation Environment for Atropos

Trains LLMs to generate correct SQL queries from natural language questions.
Uses the Salesforce/WikiSQL dataset with execution-based scoring.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from sql_executor import (
    create_table_from_wikisql,
    execute_sql,
    extract_boxed_sql,
    results_match,
)
from tqdm.asyncio import tqdm_asyncio
from wikisql_loader import load_wikisql_split

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

system_prompt += """You are a SQL expert. Given a table schema and a natural language question,
generate a SQL query that answers the question.

You are allocated a maximum of 1024 tokens, please strive to use less.

Provide your SQL query inside \\boxed{} like this: \\boxed{SELECT column FROM table WHERE condition}

Important:
- Use only the columns provided in the table schema
- The table is always named "data"
- Ensure your SQL syntax is valid SQLite

So please end your answer with \\boxed{your SQL query here}"""


class WikiSQLRow(TypedDict):
    """Type definition for a WikiSQL dataset row."""

    question: str
    header: List[str]
    rows: List[List[Any]]
    types: List[str]
    gold_sql: str


def format_table_schema(
    header: List[str], rows: List[List[Any]], max_rows: int = 3
) -> str:
    """Format table schema for the prompt."""
    schema = f"Table: data\nColumns: {', '.join(header)}\n"
    if rows:
        schema += "Sample data:\n"
        for row in rows[:max_rows]:
            row_str = " | ".join(str(v) for v in row)
            schema += f"  {row_str}\n"
    return schema


class SQLQueryEnv(BaseEnv):
    """
    Environment for training LLMs to generate SQL queries.

    Uses the Salesforce/WikiSQL dataset and verifies correctness
    by executing SQL against in-memory SQLite databases.
    """

    name = "sql_query"

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
        self.execution_success_buffer = list()

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
            wandb_name="sql_query",
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

        # Log execution success rate
        try:
            wandb_metrics["train/execution_success"] = sum(
                self.execution_success_buffer
            ) / len(self.execution_success_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = list()
        self.execution_success_buffer = list()

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Load the WikiSQL dataset and prepare train/test splits."""
        # Load WikiSQL dataset directly from GitHub source
        print("Loading WikiSQL training data...")
        self.train = load_wikisql_split("train")
        print(f"Loaded {len(self.train)} training examples")

        print("Loading WikiSQL test data...")
        self.test = load_wikisql_split("test")
        print(f"Loaded {len(self.test)} test examples")

        random.shuffle(self.train)
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _score_sql(
        self,
        generated_sql: str,
        gold_sql: str,
        header: List[str],
        rows: List[List[Any]],
    ) -> Tuple[float, bool]:
        """
        Score SQL by execution comparison.

        Returns:
            Tuple of (score, execution_success)
        """
        if not generated_sql:
            return -1.0, False

        # Create in-memory table
        try:
            conn = create_table_from_wikisql(header, rows)
        except Exception:
            return -1.0, False

        # Execute gold SQL
        gold_result = execute_sql(conn, gold_sql)
        if gold_result is None:
            conn.close()
            return -1.0, False

        # Execute generated SQL
        gen_result = execute_sql(conn, generated_sql)
        conn.close()

        if gen_result is None:
            return -1.0, False

        # Compare results
        if results_match(gen_result, gold_result):
            return 1.0, True
        else:
            return -1.0, True

    async def rollout_and_score_eval(
        self, question: str, gold_sql: str, header: List[str], rows: List[List[Any]]
    ) -> dict:
        """Rollout and score a single evaluation item."""
        table_schema = format_table_schema(header, rows)
        user_content = f"{table_schema}\nQuestion: {question}"

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

        # Extract and score generated SQL
        generated_sql = extract_boxed_sql(response_content)
        score, exec_success = self._score_sql(generated_sql, gold_sql, header, rows)
        correct = score == 1.0

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_content},
            ],
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "score": 1 if correct else 0,
            "correct": correct,
            "execution_success": exec_success,
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": 1 if correct else 0, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set."""
        import time

        start_time = time.time()

        eval_tasks = []
        # Limit eval to first 500 items for efficiency
        for item in self.test[:500]:
            eval_tasks.append(
                self.rollout_and_score_eval(
                    item["question"], item["gold_sql"], item["header"], item["rows"]
                )
            )
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores) if scores else 0

        end_time = time.time()

        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

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
        self, item: WikiSQLRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        """Generate SQL queries for a given question."""
        table_schema = format_table_schema(item["header"], item["rows"])
        user_content = f"{table_schema}\nQuestion: {item['question']}"
        user_message = {"role": "user", "content": user_content}

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[{"role": "system", "content": system_prompt}, user_message],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "gold_sql": item["gold_sql"],
                    "header": item["header"],
                    "rows": item["rows"],
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
        """Score generated SQL queries by execution comparison."""
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Get table info from first item
        gold_sql = rollout_group_data[0]["gold_sql"]
        header = rollout_group_data[0]["header"]
        rows = rollout_group_data[0]["rows"]

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            response_content = item["messages"][-1]["content"]

            # Extract SQL from response
            generated_sql = extract_boxed_sql(response_content)

            # Score by execution
            reward, exec_success = self._score_sql(
                generated_sql, gold_sql, header, rows
            )
            self.execution_success_buffer.append(1 if exec_success else 0)

            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]

            # Remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Check if all scores are the same
        if all([score == 1 for score in scores["scores"]]):
            # Apply length penalty when all are correct
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) == 0:
                return None

            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5

            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    scores["scores"].append(1.0)
                else:
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    percentage_of_range = min(percentage_of_range, 1.0)
                    scores["scores"].append(1.0 - percentage_of_range)

        if all([scores["scores"][0] == score for score in scores["scores"]]):
            return None  # If all the same, return None

        return scores

    async def get_next_item(self) -> WikiSQLRow:
        """Get the next training item."""
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    SQLQueryEnv.cli()
