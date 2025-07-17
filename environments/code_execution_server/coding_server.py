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

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)
system_prompt = ""

# system_prompt += """You are allocated a maximum of 16000 tokens, please strive to use less. You must submit \
# all code outputs in python, enclosed in triple backticks and specifying the language, like: \
# ```python\nYOUR CODE HERE\n```."""

system_prompt += (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)

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

lock = asyncio.Lock()
lock2 = asyncio.Lock()
async_semaphore = asyncio.Semaphore(50)
limiter = AsyncLimiter(1000, 5)

"""
def get_prompt(question, problem_type):
    prompt = f"### Question:\n{question}\n\n"
    if problem_type == "stdin_stdout":
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt
"""


def get_prompt(question, problem_type, starter_code=None):
    # prompt = ("You will be given a question (problem specification) and will generate a correct Python program "
    #          "that matches the specification and passes all tests.\n\n")
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


# system_prompt = """You must submit all code outputs in python, enclosed in triple backticks and \
# specifying the language, like: ```python\nYOUR CODE HERE\n```"""
async def submit_code(code, test_input, language="python"):
    payload = {"code": code, "input": test_input}
    # payload = Item(**payload)
    async with async_semaphore, limiter:
        # async with client.post(url, json=payload) as response:
        # response = await client.post(url, json=payload)
        response_json = await code_exec.remote.aio(payload)
        # response_json = response.json()
        return (
            response_json["output"],
            response_json["error"],
            response_json["returncode"],
        )


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
    # async with httpx.AsyncClient(timeout=httpx_timeout, limits=httpx_limits) as client:
    # async with aiohttp.ClientSession(timeout=aio_timeout) as client:
    tasks = []
    task_args = []
    for i in range(len(answer)):
        task_args.append({"code": code, "input": answer[i]})
        tasks.append(submit_code(code, answer[i]))

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

    results = await asyncio.gather(*tasks)
    return [result for result in results]


class CodeConfig(BaseEnvConfig):
    dataset_name: str = Field(
        "normal", description="dataset name, should be normal or deepmind"
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
            max_eval_workers=12,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.25,
            max_token_length=32768,
            min_items_sent_before_logging=1,
            wandb_name="coding_rl",
            dataset_name="normal",
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
        system_msg = {"role": "system", "content": system_prompt}
        user_msg = {
            "role": "user",
            "content": get_prompt(
                item["problem"], item["problem_type"], item["starter_code"]
            ),
        }
        # print(system_msg)
        # print(user_msg)

        prompt_tokens = tokenize_for_trainer(
            self.tokenizer, chat=[system_msg, user_msg]
        )
        buffer = 32

        async def generate_and_score(index: int) -> Tuple[List[int], List[int], float]:
            # Step 1: Generate single completion

            max_tokens = (
                self.config.max_token_length - len(prompt_tokens["tokens"]) - buffer
            )

            chat_completion = await self.server.chat_completion(
                messages=[system_msg, user_msg],
                n=1,  # CHANGE
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
            messages = [system_msg, user_msg, assistant_msg]
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
            tasks = [generate_and_score(i) for i in range(16)]
        # tasks = [generate_and_score(i) for i in range(1)] ### CHANGE
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
        # for error, code in zip(errors, codes):
        #    print("ERROR:", error)
        # if 'output' in error and error['output'] == '':
        #    print(code)
        print(f"Elapsed time: {dt:.2f} seconds")

        async with self.lock_heap:
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

        """
        async with self.lock2:
            if all([scored_data["scores"][0] == score for score in scored_data["scores"]]):
                if round(scored_data["scores"][0]) == 1:
                    with open("solved_2.txt", "a") as f:
                        f.write(f"{cur_id}\n")
                elif round(scored_data["scores"][0]) == -1:
                    with open("unsolved_2.txt", "a") as f: ### CHANGE
                        f.write(f"{cur_id}\n")
            else:
                with open("mixed_2.txt", "a") as f: ### CHANGE
                    f.write(f"{cur_id}\n")
        """

        return scored_data, []

    async def rollout_and_score_eval(self, test_item):
        """Rollout and score evaluation with detailed sample data collection."""
        scored_data, _ = await self.collect_trajectories(test_item)

        scores = scored_data["scores"]
        num_correct = sum(1 for x in scores if math.isclose(x, 1.0))
        total_attempts = len(scores)

        # Calculate pass@1 estimate for this single item
        def estimator(n: int, c: int, k: int) -> float:
            """Calculates 1 - comb(n - c, k) / comb(n, k)."""
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        pass_at_1 = estimator(total_attempts, num_correct, 1)

        # Create sample data
        sample = {
            "problem_description": test_item.get("problem_description", ""),
            "difficulty": test_item.get("difficulty", "unknown"),
            "total_attempts": total_attempts,
            "correct_attempts": num_correct,
            "pass_at_1": pass_at_1,
            "scores": scores,
            "avg_completion_length": (
                sum([len(x) for x in scored_data["tokens"]])
                / len(scored_data["tokens"])
                if scored_data["tokens"]
                else 0
            ),
        }

        return {"score": pass_at_1, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps
        """
        print("EVALUATION")
        start_time = time.time()
        test_data = self.test

        # Use rollout_and_score_eval for each test item with semaphore
        async def eval_with_semaphore(item):
            async with async_semaphore:
                return await self.rollout_and_score_eval(item)

        eval_tasks = []
        for item in test_data:
            eval_tasks.append(eval_with_semaphore(item))

        results = await asyncio.gather(*eval_tasks)

        # Extract scores and samples
        samples = [result["sample"] for result in results]

        # Aggregate metrics by difficulty
        all_scores, easy_scores, medium_scores, hard_scores = [], [], [], []
        for result in results:
            sample = result["sample"]
            score = result["score"]
            all_scores.append(score)

            difficulty = sample["difficulty"]
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
            sum(sample["avg_completion_length"] for sample in samples) / len(samples)
            if samples
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
        if self.config.dataset_name == "deepmind":
            self.train = load_dataset("deepmind/code_contests", split="train")  # CHANGE
        else:
            self.train = load_dataset(
                "NousResearch/RL_Agentica_STDIN", split="train"
            )  # CHANGE
        test = load_dataset("NousResearch/lcb_test", split="test")
        self.test = []
        for problem in test:
            self.test.append(problem)
            self.test[-1]["idx"] = len(self.test) - 1
            self.test[-1]["split"] = "test"
        self.iter = 0  # CHANGE
        self.lock = asyncio.Lock()
        self.lock2 = asyncio.Lock()
        self.lock_heap = asyncio.Lock()

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
        await self.evaluate()  # CHANGE

    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out
        """
        async with self.lock:
            if not self.deq:
                while len(self.next_heap) > 0:
                    self.deq.append(heapq.heappop(self.next_heap)[1])
            cur_idx = self.deq.popleft()
            next_item = self.train[cur_idx]
            # next_item = self.train[(self.good_indices[self.iter]) % len(self.train)]
            # self.iter += 1
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

        results = []
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
                    res, metadata = await run_test.remote.aio(test_cases, code)
                    """
                    fn_name = test_cases["tests"]["fn_name"]
                    inputs = test_cases["tests"]["input"]
                    outputs = test_cases["tests"]["output"]
                    starmap_entries = [
                        ({"tests": {"input": [inp], "output": [out], "fn_name": fn_name}}, code)
                        for inp, out in zip(inputs, outputs)
                    ]
                    #tasks = [run_test.remote.aio(entry) for entry in starmap_entries]
                    #res = await asyncio.gather(*tasks)

                    res = []
                    metadata = []
                    try:
                        async for x, y in run_test.starmap.aio(starmap_entries, return_exceptions=True):
                            res.extend(x)
                            metadata.append(y)
                    except Exception as e:
                        metadata.append(e)
                        print(e)
                        print(code)
                        print("segfault", cur_id)
                    """
                    # print(res)
                    # print(metadata)

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
                # print(res)
                # print(metadata)
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
        old_error = errors[0]
        # print(old_error)
        # print(type(old_error))
        # print(errors[0])
        old_score = scores
        true_score = scores["scores"][0]

        """
        STOP _-------------
        """
        # print("Rollout group data", rollout_group_data)
        assert len(rollout_group_data) == 1
        cur_id = rollout_group_data[0][2]

        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["overrides"] = list()
        # random.shuffle(rollout_group_data)
        results = []
        codes = []
        lengths = []
        errors = []
        printing = []
        for item in rollout_group_data:
            scores["overrides"].append(dict())
            scores["overrides"][-1]["set_advantage_to_zero"] = False
            if item[3] == "length":
                scores["overrides"][-1]["set_advantage_to_zero"] = True
            else:
                assert item[3] == "stop"
            out_dict = tokenize_for_trainer(self.tokenizer, item[0], item[3])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            lengths.append(len(tokens))
            """
            CALCULATE REWARD NOW
            """
            code = self.extract_python_code_blocks(item[0][-1]["content"])
            if len(code) > 0:
                code = code[-1]
            else:
                code = None
            test_cases = item[1]["input"]
            x = get_results(code, test_cases)
            results.append(x)
            codes.append(code)

            # if len([1 for i in masks if i != -100]) < 10:
            #    continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)

        results = await asyncio.gather(*results)
        for x, item, code in zip(results, rollout_group_data, codes):
            # output_cases = list(item[1][1]) + list(item[1][3])
            output_cases = item[1]["output"]

            """x = np.asarray(x)
            ret_codes = x[:,2]
            err = x[:,1]
            x = x[:,0]"""
            err = [x[i][1] for i in range(len(x))]
            x = [x[i][0] for i in range(len(x))]

            """
            print("CODE:\n")
            print(code)
            for k in range(len(x)):
                print("CASE")
                print("INPUT", item[1]["input"][k])
                print("Expected output\n", x[k])
                print("output\n", output_cases[k])
            """

            """
            for k in range(len(x)):
                if ret_codes[k] == -8:
                    print(code)
                    print("INPUT: ", item[1]["input"][k])
                    print("OUTPUT: ", x[k])
                    sys.exit(0)
            """
            # if err[0]:
            #    print("ERROR:", err[0])
            errors.append(err[0])
            assert len(x) == len(output_cases)
            reward = True
            for k in range(len(x)):
                if x[k].strip() != output_cases[k].strip():
                    reward = False
                    break
            # print("CODE ", cur_id, "\n", code, "\nREWARD\n", reward, "\n")
            printing.append("CODE " + str(cur_id))
            printing.append(code)
            printing.append("REWARD ")
            printing.append(reward)

            scores["scores"].append(1.0 if reward else -1.0)

        """async with lock2:
            if all([scores["scores"][0] == score for score in scores["scores"]]):
                if round(scores["scores"][0]) == 1:
                    with open("solved.txt", "a") as f:
                        f.write(f"{cur_id}\n")
                elif round(scores["scores"][1]) == -1:
                    with open("unsolved.txt", "a") as f:
                        f.write(f"{cur_id}\n")
            else:
                with open("mixed.txt", "a") as f:
                    f.write(f"{cur_id}\n")"""
        assert len(codes) == 1 and len(errors) == 1 and len(lengths) == 1

        if (
            (true_score != scores["scores"][0])
            and item[1]["fn_name"] == "none"
            and code is not None
        ):
            print("INDEX", cur_id)
            print("NOT WORK", true_score, scores["scores"][0])
            # print("CODE:\n", codes[0])
            # print("INPUTS:\n", item[1]["input"])
            print("ERROR:\n", errors[0])
            print("LCB ERROR:\n", old_error)
        elif true_score == -1.0 and item[1]["fn_name"] == "none" and code is not None:
            if not (
                "error_code" in old_error
                and old_error["error_code"] == -2
                and not errors[0]
            ) and not (
                "error_code" in old_error
                and old_error["error_code"] == -3
                and "Time" in errors[0]
            ):
                print("INDEX", cur_id)
                print("NOT WORK", true_score, scores["scores"][0])
                # print("CODE:\n", codes[0])
                # print("INPUTS:\n", item[1]["input"])
                print("ERROR:\n", errors[0])
                print("LCB ERROR:\n", old_error)
            # print("OUTPUTS:\n")
            # print(x, item[1]["output"])
        assert true_score >= scores["scores"][0]
        return old_score, (codes[0], errors[0], lengths[0])


if __name__ == "__main__":
    print(system_prompt)
    CodingEnv.cli()
