import asyncio
import heapq
import json
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import aiohttp
import httpx
import modal
import numpy as np
import regex as re
from aiolimiter import AsyncLimiter
from datasets import load_dataset
from pydantic import Field
from rich import print as rprint

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin, solve the problem, and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs it reads the inputs runs the "
    "algorithm and writes output to STDOUT."
)

limiter = AsyncLimiter(1000, 5)


def get_prompt(question, problem_type, starter_code=None):
    prompt = ""
    prompt += f"Question: {question}\n\n"
    if problem_type == "func":
        prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


code_exec = modal.Function.from_name("code_exec", "execute_code")

run_test = modal.Function.from_name("test-lcb-code", "run_test")


async def submit_code(test_cases, code):
    """Submit code for execution with container semaphore"""
    async with containers_semaphore:
        return await run_test.remote.aio(test_cases, code)


httpx_timeout = httpx.Timeout(None, connect=None, read=None, write=None, pool=None)
httpx_limits = httpx.Limits(
    max_keepalive_connections=None,
    max_connections=None,
    keepalive_expiry=None,
)

aio_timeout = aiohttp.ClientTimeout(
    total=None, connect=None, sock_connect=None, sock_read=None
)


async def get_results(code, answer):
    task_args = []
    for i in range(len(answer)):
        task_args.append({"code": code, "input": answer[i]})

    results = []
    async for response_json in code_exec.map.aio(task_args):
        results.append(
            (
                response_json["output"],
                response_json["error"],
                response_json["returncode"],
            )
        )

    return results


class CodeConfig(BaseEnvConfig):
    dataset_name: str = Field(
        "normal", description="dataset name, should be normal or deepmind"
    )
    max_test_cases_to_log: Optional[int] = Field(
        None,
        description="Maximum number of test cases to include in logged output "
        "(for storage). All test cases are still evaluated.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt. If None, no system message is used.",
    )
    eval_max_samples: Optional[int] = Field(
        None,
        description="Maximum number of test problems to evaluate. If None, evaluates all test problems.",
    )
    max_running_containers: int = Field(
        50,
        description="Maximum number of concurrent container executions for code testing.",
    )
    eval_group_size: int = Field(
        16,
        description="Number of samples to generate for each problem during evaluation.",
    )


class CodingEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = 0
        self.complete = []
        self.total = []
        self.complement = []
        self.eval_metrics = []
        self.blacklist = []
        self.deq = deque()
        self.next_heap = []

    @classmethod
    def config_init(cls) -> Tuple[CodeConfig, List[APIServerConfig]]:
        env_config = CodeConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=200,
            batch_size=-1,
            steps_per_eval=10,
            max_batches_offpolicy=3,
            max_eval_workers=32,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.25,
            max_token_length=32768,
            min_items_sent_before_logging=1,
            wandb_name="coding_rl",
            dataset_name="normal",
            max_test_cases_to_log=5,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    # item looks like {problem, tests, problem_type, idx}
    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        print("COLLECTING TRAJECTORIES")
        train_or_valid = item["split"]
        user_msg = {
            "role": "user",
            "content": get_prompt(
                item["problem"], item["problem_type"], item["starter_code"]
            ),
        }

        # Build chat messages, only include system message if provided
        chat_messages = []
        if self.config.system_prompt is not None:
            chat_messages.append({"role": "system", "content": self.config.system_prompt})
        chat_messages.append(user_msg)

        prompt_tokens = tokenize_for_trainer(self.tokenizer, chat=chat_messages)
        buffer = 32

        async def generate_and_score(index: int) -> Tuple[List[int], List[int], float]:
            # Step 1: Generate single completion

            max_tokens = (
                self.config.max_token_length - len(prompt_tokens["tokens"]) - buffer
            )

            chat_completion = await self.server.chat_completion(
                messages=chat_messages,
                n=1,
                max_tokens=max_tokens,
            )
            content = chat_completion.choices[0].message.content
            assistant_msg = {"role": "assistant", "content": content}

            # Zip input and output together
            tests = list(zip(item["tests"]["input"], item["tests"]["output"]))

            # Sort by length of input string, descending
            tests.sort(key=lambda x: len(x[0]), reverse=True)

            # Unzip back into separate lists
            item["tests"]["input"], item["tests"]["output"] = zip(*tests)

            # Convert back to lists if needed (zip returns tuples)
            item["tests"]["input"] = list(item["tests"]["input"])
            item["tests"]["output"] = list(item["tests"]["output"])

            FIRST_BATCH = 3
            if "fn_name" not in item:
                fn_name = "none"
            else:
                fn_name = item["fn_name"]

            if len(item["tests"]["input"]) > FIRST_BATCH:
                first_tests = {
                    "input": item["tests"]["input"][:FIRST_BATCH],
                    "output": item["tests"]["output"][:FIRST_BATCH],
                }
                second_tests = {
                    "input": item["tests"]["input"][FIRST_BATCH:],
                    "output": item["tests"]["output"][FIRST_BATCH:],
                }
                first_tests["fn_name"] = fn_name
                second_tests["fn_name"] = fn_name
            else:
                first_tests = item["tests"]
                first_tests["fn_name"] = fn_name

            # Step 2: Prepare messages and score input
            messages = chat_messages + [assistant_msg]
            score_input = [
                (
                    messages,
                    first_tests,
                    item["idx"],
                    chat_completion.choices[0].finish_reason,
                )
            ]

            # Step 3: Call score() on just this one completion
            scored_group, tup = await self.score(
                score_input
            )  # tup = (code, error, length)

            # Extract the single result (we only scored one item)
            if (
                math.isclose(scored_group["scores"][0], -1.0)
                or len(item["tests"]["input"]) <= FIRST_BATCH
            ):
                return (
                    scored_group["tokens"][0],
                    scored_group["masks"][0],
                    scored_group["scores"][0],
                    scored_group["overrides"][0],
                    tup[0],
                    tup[1],
                    tup[2],
                )
            score_input = [
                (
                    messages,
                    second_tests,
                    item["idx"],
                    chat_completion.choices[0].finish_reason,
                )
            ]
            scored_group, tup = await self.score(
                score_input
            )  # tup = (code, error, length)
            return (
                scored_group["tokens"][0],
                scored_group["masks"][0],
                scored_group["scores"][0],
                scored_group["overrides"][0],
                tup[0],
                tup[1],
                tup[2],
            )

        start_time = time.time()
        if train_or_valid == "train":
            self.total.append(item["idx"])
            self.complement = sorted(list(set(self.total) - set(self.complete)))
            print("TOTAL: ", self.complement)
        if train_or_valid == "train":
            tasks = [generate_and_score(i) for i in range(self.config.group_size)]
        else:
            tasks = [generate_and_score(i) for i in range(self.config.eval_group_size)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        dt = end_time - start_time
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = [r[0] for r in results]
        scored_data["masks"] = [r[1] for r in results]
        scored_data["scores"] = [r[2] for r in results]
        scored_data["overrides"] = [r[3] for r in results]

        lengths = [r[6] for r in results]
        cur_id = item["idx"]
        if train_or_valid == "train":
            self.complete.append(cur_id)
            self.complement = sorted(list(set(self.total) - set(self.complete)))
        print(train_or_valid)
        print("CURRENT ID: ", cur_id, item["problem_type"])
        print("GEN LENGTHS: ", lengths)
        print("SCORES ", scored_data["scores"])
        print("MISSING ", self.complement)
        print(f"Elapsed time: {dt:.2f} seconds")

        async with self.heap_lock:
            heap_sum = 0
            for x in scored_data["scores"]:
                if math.isclose(1.0, x):
                    heap_sum += 1
            if 2 * heap_sum > self.config.group_size:
                self.blacklist.append(cur_id)
            else:
                heapq.heappush(self.next_heap, (-heap_sum, cur_id))
            with open("data_dump.txt", "a") as f:
                mp = {"cur_id": cur_id, "num_correct": heap_sum}
                f.write(json.dumps(mp) + "\n")

        return scored_data, []

    def _subset_test_cases_for_logging(self, test_cases):
        """Subset test cases for logging to reduce storage size, selecting shortest cases."""
        if self.config.max_test_cases_to_log is None:
            return test_cases

        if (
            isinstance(test_cases, dict)
            and "input" in test_cases
            and "output" in test_cases
        ):
            # Handle the case where test_cases has input/output lists
            if isinstance(test_cases["input"], list) and isinstance(
                test_cases["output"], list
            ):
                max_cases = min(
                    self.config.max_test_cases_to_log, len(test_cases["input"])
                )

                # Create pairs of (input, output) with their combined length
                case_pairs = list(zip(test_cases["input"], test_cases["output"]))

                # Sort by combined string length (input + output) to get shortest cases
                case_pairs_with_length = [
                    (inp, out, len(str(inp)) + len(str(out))) for inp, out in case_pairs
                ]
                case_pairs_with_length.sort(
                    key=lambda x: x[2]
                )  # Sort by combined length

                # Take the shortest cases
                shortest_cases = case_pairs_with_length[:max_cases]

                return {
                    "input": [inp for inp, _, _ in shortest_cases],
                    "output": [out for _, out, _ in shortest_cases],
                    "fn_name": test_cases.get("fn_name", "none"),
                    "_original_count": len(test_cases["input"]),
                    "_logged_count": max_cases,
                    "_selection_method": "shortest_by_length",
                }

        return test_cases

    def _truncate_text_for_logging(self, text, max_length=1000):
        """Truncate long text fields to reduce storage size."""
        if isinstance(text, str) and len(text) > max_length:
            return text[:max_length] + "...[TRUNCATED]"
        return text

    async def rollout_and_score_eval(self, test_item):
        """Rollout and score evaluation with detailed sample and group data collection."""
        # Store original collect_trajectories method to capture detailed sample data
        user_msg = {
            "role": "user",
            "content": get_prompt(
                test_item["problem"],
                test_item["problem_type"],
                test_item["starter_code"],
            ),
        }

        # Build chat messages, only include system message if provided
        chat_messages = []
        system_prompt_content = self.get_system_prompt()
        if system_prompt_content is not None:
            chat_messages.append({"role": "system", "content": system_prompt_content})
        chat_messages.append(user_msg)

        # Generate multiple samples for evaluation
        num_samples = self.config.eval_group_size
        sample_data = []
        response_texts = []

        from tqdm import tqdm

        for i in tqdm(
            range(num_samples),
            desc=f"Generating samples for problem {test_item['idx']}",
            leave=False,
        ):
            # Generate completion
            max_tokens = self.config.max_token_length - 1000  # Buffer
            chat_completion = await self.server.chat_completion(
                messages=chat_messages,
                n=1,
                max_tokens=max_tokens,
            )
            content = chat_completion.choices[0].message.content
            assistant_msg = {"role": "assistant", "content": content}
            messages = chat_messages + [assistant_msg]

            # Extract code and test
            code = self.extract_python_code_blocks(content)
            code_text = code[-1] if code else None
            response_texts.append(content)

            # Score the sample
            if code_text is None:
                passed = False
                error_trace = {"error_message": "No code found in response"}
                execution_output = ""
            else:
                try:
                    # Use the same test setup as collect_trajectories
                    if "fn_name" not in test_item:
                        fn_name = "none"
                    else:
                        fn_name = test_item["fn_name"]

                    test_cases = {
                        "tests": {
                            "input": test_item["tests"]["input"],
                            "output": test_item["tests"]["output"],
                            "fn_name": fn_name,
                        }
                    }

                    res, metadata = await submit_code(test_cases, code_text)
                    passed = set(res) == {True}
                    error_trace = metadata
                    execution_output = str(res)

                except Exception as e:
                    passed = False
                    error_trace = {"error": str(e)}
                    execution_output = ""

            # Create detailed sample entry
            sample_entry = {
                "problem_id": test_item["idx"],
                "messages": messages,
                "response_text": content,
                "code_text": code_text,
                "passed": passed,
                "error_trace": error_trace,
                "execution_output": execution_output,
                "finish_reason": chat_completion.choices[0].finish_reason,
                "problem_type": test_item["problem_type"],
                "difficulty": test_item.get("difficulty", "unknown"),
            }
            sample_data.append(sample_entry)

        # Calculate metrics
        num_correct = sum(1 for s in sample_data if s["passed"])
        total_attempts = len(sample_data)

        # Calculate pass@1 estimate
        def estimator(n: int, c: int, k: int) -> float:
            """Calculates 1 - comb(n - c, k) / comb(n, k)."""
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        pass_at_1 = estimator(total_attempts, num_correct, 1)

        # Create group data entry
        group_entry = {
            "problem_id": test_item["idx"],
            "problem": test_item["problem"],
            "difficulty": test_item.get("difficulty", "unknown"),
            "problem_type": test_item["problem_type"],
            "response_texts": response_texts,
            "scores": [
                1 if s["passed"] else 0 for s in sample_data
            ],  # Parallel list of 0/1 scores
            "metrics": {
                "pass_at_1": pass_at_1,
                "total_samples": total_attempts,
                "correct_samples": num_correct,
                "avg_completion_length": (
                    sum(len(s["response_text"]) for s in sample_data) / len(sample_data)
                    if sample_data
                    else 0
                ),
            },
            "starter_code": test_item.get("starter_code", ""),
            "tests": self._subset_test_cases_for_logging(test_item["tests"]),
        }

        return {"score": pass_at_1, "group": group_entry, "samples": sample_data}

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps
        """
        print("EVALUATION")
        start_time = time.time()
        test_data = (
            self.test
            if self.config.eval_max_samples is None
            else self.test[: self.config.eval_max_samples]
        )

        # Use rollout_and_score_eval for each test item with semaphore
        async def eval_with_semaphore(item):
            async with eval_semaphore:
                return await self.rollout_and_score_eval(item)

        eval_tasks = []
        for item in test_data:
            eval_tasks.append(eval_with_semaphore(item))

        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating test cases")

        # Extract groups and samples
        groups = [result["group"] for result in results]
        samples = []
        for result in results:
            samples.extend(result["samples"])

        # Aggregate metrics by difficulty
        all_scores, easy_scores, medium_scores, hard_scores = [], [], [], []
        for result in results:
            score = result["score"]
            all_scores.append(score)

            difficulty = result["group"]["difficulty"]
            if difficulty == "easy":
                easy_scores.append(score)
            elif difficulty == "medium":
                medium_scores.append(score)
            elif difficulty == "hard":
                hard_scores.append(score)

        # Calculate average pass@1 for each difficulty
        all_pass_1 = sum(all_scores) / len(all_scores) if all_scores else 0.0
        easy_pass_1 = sum(easy_scores) / len(easy_scores) if easy_scores else 0.0
        medium_pass_1 = (
            sum(medium_scores) / len(medium_scores) if medium_scores else 0.0
        )
        hard_pass_1 = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0

        # Calculate average completion length
        avg_comp_len = (
            sum(group["metrics"]["avg_completion_length"] for group in groups)
            / len(groups)
            if groups
            else 0.0
        )

        end_time = time.time()

        # Add to existing metrics for wandb
        self.eval_metrics.append(("eval/pass_1", all_pass_1))
        self.eval_metrics.append(("eval/easy_pass_1", easy_pass_1))
        self.eval_metrics.append(("eval/medium_pass_1", medium_pass_1))
        self.eval_metrics.append(("eval/hard_pass_1", hard_pass_1))
        self.eval_metrics.append(("eval/completion_length", avg_comp_len))
        print("STATS", self.eval_metrics)

        # Log evaluation results
        eval_metrics = {
            "eval/pass_1": all_pass_1,
            "eval/easy_pass_1": easy_pass_1,
            "eval/medium_pass_1": medium_pass_1,
            "eval/hard_pass_1": hard_pass_1,
            "eval/completion_length": avg_comp_len,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            groups=groups,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "max_tokens": self.config.max_token_length,
            },
        )

        return

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Setup the environment"""
        global eval_semaphore, containers_semaphore
        eval_semaphore = asyncio.Semaphore(self.config.max_eval_workers)
        containers_semaphore = asyncio.Semaphore(self.config.max_running_containers)

        if self.config.dataset_name == "deepmind":
            self.train = load_dataset("deepmind/code_contests", split="train")
        else:
            self.train = load_dataset("NousResearch/RL_Agentica_STDIN", split="train")
        test = load_dataset("NousResearch/lcb_test", split="test")
        self.test = []

        # Load all test cases for evaluation (no sampling)
        for problem in test:
            self.test.append(problem)
            self.test[-1]["idx"] = len(self.test) - 1
            self.test[-1]["split"] = "test"
        self.iter = 0
        self.queue_lock = asyncio.Lock()
        self.heap_lock = asyncio.Lock()

        self.good_indices = []
        if self.config.dataset_name != "deepmind":
            with open("indices.txt", "r") as f:
                for line in f:
                    self.deq.append(int(line.strip()))
                    self.good_indices.append(int(line.strip()))

            set_queue = set(self.deq)
            for i in range(len(self.train)):
                if i not in set_queue:
                    self.deq.append(i)
        else:
            self.good_indices = [i for i in range(10000)]
            for i in range(10000):
                self.deq.append(i)

        rprint("NUM FILTERED EXAMPLES:", len(self.good_indices))

    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out
        """
        async with self.queue_lock:
            if not self.deq:
                while len(self.next_heap) > 0:
                    self.deq.append(heapq.heappop(self.next_heap)[1])
            cur_idx = self.deq.popleft()
            next_item = self.train[cur_idx]
        next_item["idx"] = cur_idx
        next_item["split"] = "train"
        if self.config.dataset_name == "deepmind":
            next_item["problem"] = next_item["description"]
            next_item["tests"] = {
                "input": next_item["private_tests"]["input"]
                + next_item["generated_tests"]["input"],
                "output": next_item["private_tests"]["output"]
                + next_item["generated_tests"]["output"],
            }
            next_item["problem_type"] = "stdin_stdout"
        return next_item

    def extract_python_code_blocks(self, text):
        # Regex specifically looks for ```python\n...code...\n```
        pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
        result = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        python_blocks = [r for r in result]
        return python_blocks

    # V2
    async def score(self, rollout_group_data):

        assert len(rollout_group_data) == 1
        cur_id = rollout_group_data[0][2]

        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["overrides"] = list()

        codes = []
        lengths = []
        errors = []

        for item in rollout_group_data:
            scores["overrides"].append(dict())
            scores["overrides"][-1]["set_advantage_to_zero"] = False
            if item[3] == "length":
                scores["overrides"][-1]["set_advantage_to_zero"] = True
            else:
                if item[3] != "stop":
                    print("FINISH REASON", item[3])
                # assert item[3] == "stop"

            out_dict = tokenize_for_trainer(self.tokenizer, item[0], item[3])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            lengths.append(len(tokens))

            code = self.extract_python_code_blocks(item[0][-1]["content"])
            if len(code) > 0:
                code = code[-1]
            else:
                code = None
            test_cases = {"tests": item[1]}

            codes.append(code)

            if code is None:
                scores["scores"].append(-1.0)
                errors.append({"error_message": "No code"})
                continue
            else:
                try:
                    res, metadata = await submit_code(test_cases, code)

                except modal.exception.RemoteError:
                    res = [False]
                    print("index", cur_id)
                    print(test_cases["tests"]["fn_name"])
                    metadata = {"error": "segmentation fault"}
                    rprint("FAULT")
                    print("code:\n", code)
                    rprint(metadata)
                except Exception as f:
                    print("index", cur_id)
                    print(test_cases["tests"]["fn_name"])
                    print("except:", code)
                    res = [False]
                    metadata = {"error": "segmentation fault"}
                    rprint("NON SEGFAULT")
                    rprint("FAULT", f)
                for x in res:
                    if not isinstance(x, (int, bool)):
                        print("WARNING")
                        print(res)
                if set(res) == {True}:
                    scores["scores"].append(1.0)
                else:
                    scores["scores"].append(-1.0)

                errors.append(metadata)

        return scores, (codes[0], errors[0], lengths[0])


if __name__ == "__main__":
    CodingEnv.cli()
