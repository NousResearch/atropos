import asyncio
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import modal
import numpy as np
import regex as re
import wandb
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

system_prompt = ""

system_prompt += (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin, solve the problem, and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs "
    "it reads the inputs runs the algorithm and writes output to STDOUT."
)

async_semaphore = asyncio.Semaphore(100)


def get_prompt(question, problem_type, starter_code=None):
    prompt = ""
    prompt += f"Question: {question}\n\n"
    if problem_type == "func" and starter_code:
        prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    elif problem_type == "func" and not starter_code:
        pass
    else:
        prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


run_test = modal.Function.from_name("joeli-lcb", "run_test")


class CodeConfig(BaseEnvConfig):
    dataset_name: str = Field(
        "normal", description="dataset name, should be normal or deepmind"
    )
    temperature: float = Field(0.6, description="model temperature")
    eval_temperature: float = Field(
        0.6, description="model temperature during evaluation"
    )
    top_p: float = Field(0.95, description="top p")
    eval_top_p: float = Field(0.95, description="eval top p")
    start_idx: int = Field(0, description="start index in training set")
    max_eval_token_length: int = Field(
        40960, description="max sequence length during evaluation"
    )


class CodingEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.id = 0
        self.complete = []
        self.total = []
        self.complement = []
        self.eval_metrics = []
        self.train_metrics = {
            "rewards": [],
            "overlong_ratio": [],
            "pass_gs": [],
            "completion_lengths": [],
            "correct_completion_lengths": [],
            "incorrect_completion_lengths": [],
        }
        self.cur_time = datetime.now().strftime("%Y-%m-%d %H.%M.%S")

        self.temp_metrics = {
            "indices": [],
            "num_correct": [],
        }

        self.blacklist = set()
        self.deq = deque()

    @classmethod
    def config_init(cls) -> Tuple[CodeConfig, List[APIServerConfig]]:
        env_config = CodeConfig(
            tokenizer_name="Qwen/Qwen3-14B",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=600,
            batch_size=-1,
            steps_per_eval=10,
            max_batches_offpolicy=2,
            max_eval_workers=128,  # max workers per node * num gpus
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            eval_limit_ratio=0.25,
            max_token_length=32768,
            min_items_sent_before_logging=0,
            wandb_name="qwen_gspo_32k_mod",
            dataset_name="normal",
            worker_timeout=7200,
            temperature=1.0,
            eval_temperature=0.6,
            top_p=1.0,
            eval_top_p=0.95,
            start_idx=0,
            max_eval_token_length=40960,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-14B",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
                timeout=2400,
            ),
        ]

        return env_config, server_configs

    # item looks like {problem, tests, problem_type, idx}
    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        rprint("COLLECTING TRAJECTORIES")
        train_or_valid = item["split"]
        system_msg = {"role": "system", "content": system_prompt}
        user_msg = {
            "role": "user",
            "content": get_prompt(
                item["problem"], item["problem_type"], item.get("starter_code", None)
            ),
        }

        prompt_tokens = tokenize_for_trainer(
            self.tokenizer, chat=[system_msg, user_msg]
        )
        buffer = 32

        async def generate_and_score(index: int) -> Tuple[List[int], List[int], float]:
            # Step 1: Generate single completion

            max_tokens = (
                self.config.max_token_length - len(prompt_tokens["tokens"]) - buffer
            )
            temp = self.config.temperature
            top_p = self.config.top_p
            if train_or_valid == "test":
                max_tokens = (
                    self.config.max_eval_token_length
                    - len(prompt_tokens["tokens"])
                    - buffer
                )
                top_p = self.config.eval_top_p
                temp = self.config.eval_temperature

            chat_completion = await self.server.chat_completion(
                messages=[system_msg, user_msg],
                n=1,  # CHANGE
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
            )
            content = chat_completion.choices[0].message.content
            assistant_msg = {"role": "assistant", "content": content}
            messages = [system_msg, user_msg, assistant_msg]

            if "fn_name" not in item:
                fn_name = "none"
            else:
                fn_name = item["fn_name"]

            tests = item["tests"]
            if isinstance(tests, str):
                tests = json.loads(tests)
            tests["fn_name"] = fn_name

            score_input = [
                (messages, tests, item["idx"], chat_completion.choices[0].finish_reason)
            ]
            scored_group, tup = await self.score(score_input)
            return (
                scored_group["tokens"][0],
                scored_group["masks"][0],
                scored_group["scores"][0],
                scored_group["overrides"][0],
                tup[0],
                tup[1],
                tup[2],
                assistant_msg,
            )

        start_time = time.time()
        if train_or_valid == "train":
            self.total.append(item["idx"])  # LOCK
            self.complement = sorted(list(set(self.total) - set(self.complete)))
            print("TOTAL: ", self.complement)
        if train_or_valid == "train":
            tasks = [generate_and_score(i) for i in range(self.config.group_size)]
        else:
            tasks = [generate_and_score(i) for i in range(16)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        dt = end_time - start_time
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = [r[0] for r in results]
        scored_data["masks"] = [r[1] for r in results]
        scored_data["scores"] = [r[2] for r in results]
        scored_data["overrides"] = [r[3] for r in results]

        codes = [r[4] for r in results]
        errors = [r[5] for r in results]
        lengths = [r[6] for r in results]
        asst_messages = [r[7] for r in results]
        cur_id = item["idx"]
        if train_or_valid == "train":
            self.complete.append(cur_id)  # LOCK
            self.complement = sorted(list(set(self.total) - set(self.complete)))

        rprint(train_or_valid)
        rprint("CURRENT ID: ", cur_id, item["problem_type"])
        print("GEN LENGTHS: ", lengths)
        print("SCORES ", scored_data["scores"])
        print("MISSING ", self.complement)
        print("CURRENT TIME", time.time())
        # for error, code in zip(errors, codes):
        #    print("ERROR:", error)
        # if 'output' in error and error['output'] == '':
        #    print(code)
        print(f"Elapsed time: {dt:.2f} seconds")

        async with self.lock:
            heap_sum = 0
            for x in scored_data["scores"]:
                if math.isclose(1.0, x):
                    heap_sum += 1
            num_override = sum(
                [x["set_advantage_to_zero"] for x in scored_data["overrides"]]
            )

            if train_or_valid == "train":
                self.train_metrics["rewards"].append(
                    1.0 * heap_sum / self.config.group_size
                )
                self.train_metrics["overlong_ratio"].append(
                    1.0 * num_override / self.config.group_size
                )
                self.train_metrics["pass_gs"].append(1.0 * (heap_sum > 0))
                self.train_metrics["completion_lengths"].append(
                    sum([len(x) for x in scored_data["tokens"]])
                    / len(scored_data["tokens"])
                )
                self.train_metrics["correct_completion_lengths"].append(
                    sum(
                        [
                            len(x)
                            for x, y in zip(
                                scored_data["tokens"], scored_data["scores"]
                            )
                            if math.isclose(y, 1.0)
                        ]
                    )
                )
                self.train_metrics["incorrect_completion_lengths"].append(
                    sum(
                        [
                            len(x)
                            for x, y in zip(
                                scored_data["tokens"], scored_data["scores"]
                            )
                            if math.isclose(y, -1.0)
                        ]
                    )
                )
                self.temp_metrics["indices"].append(cur_id)
                self.temp_metrics["num_correct"].append(heap_sum)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            """
            if heap_sum > self.config.group_size - 2 and train_or_valid == "train":
                self.blacklist.add(cur_id)
                file_path = os.path.join(script_dir, "blacklist_" + self.cur_time + ".txt")
                with open(file_path, "a") as f:
                    f.write(json.dumps(cur_id) + "\n")
            """
            if train_or_valid == "train":
                script_dir = os.path.join(script_dir, "train_logs")
            else:
                script_dir = os.path.join(script_dir, "eval_logs")
            file_path = os.path.join(
                script_dir, "qwen_data_dump_" + self.cur_time + ".txt"
            )
            file_path_long = os.path.join(
                script_dir, "qwen_data_dump_long" + self.cur_time + ".txt"
            )
            with open(file_path, "a") as f:
                mp = {
                    "cur_id": cur_id,
                    "num_correct": heap_sum,
                    "total": self.config.group_size,
                    "scores": scored_data["scores"],
                    "lengths": lengths,
                }
                f.write(json.dumps(mp) + "\n")
            with open(file_path_long, "a") as f:
                mp = {
                    "cur_id": cur_id,
                    "num_correct": heap_sum,
                    "total": self.config.group_size,
                    "scores": scored_data["scores"],
                    "lengths": lengths,
                    "errors": errors,
                    "codes": codes,
                    "gen": asst_messages[0],
                }
                f.write(json.dumps(mp) + "\n")

        return scored_data, []

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps

        Included here is an example on how to use eval workers to run a task.

        You may however do whatever you want in this method.

        :param args:
        :param kwargs:
        :return: None.
        """
        rprint("EVALUATION")
        test_data = self.test

        sema = asyncio.Semaphore(self.config.max_eval_workers)

        all_total, all_correct = [], []
        easy_total, easy_correct = [], []
        medium_total, medium_correct = [], []
        hard_total, hard_correct = [], []

        temp_completion_lengths = []

        correct_completion_lengths = []
        incorrect_completion_lengths = []

        overlong_ratio = []
        pass_gs = []

        async def eval(curr_step):
            async with sema:
                item = test_data[curr_step]
                scored_data, _ = await self.collect_trajectories(item)

                scores = scored_data["scores"]
                num_correct = sum(1 for x in scores if math.isclose(x, 1.0))
                num_overlong = sum(
                    x["set_advantage_to_zero"] for x in scored_data["overrides"]
                )

                async with self.lock2:
                    overlong_ratio.append(num_overlong / len(scores))
                    pass_gs.append(num_correct > 0)
                    temp_completion_lengths.append(
                        sum([len(x) for x in scored_data["tokens"]])
                        / len(scored_data["tokens"])
                    )

                    correct_completion_lengths.extend(
                        [
                            len(x)
                            for x, y in zip(
                                scored_data["tokens"], scored_data["scores"]
                            )
                            if math.isclose(y, 1.0)
                        ]
                    )
                    incorrect_completion_lengths.extend(
                        [
                            len(x)
                            for x, y in zip(
                                scored_data["tokens"], scored_data["scores"]
                            )
                            if math.isclose(y, -1.0)
                        ]
                    )
                    all_total.append(len(scores))
                    all_correct.append(num_correct)
                    if item["difficulty"] == "easy":
                        easy_total.append(len(scores))
                        easy_correct.append(num_correct)
                    elif item["difficulty"] == "medium":
                        medium_total.append(len(scores))
                        medium_correct.append(num_correct)
                    elif item["difficulty"] == "hard":
                        hard_total.append(len(scores))
                        hard_correct.append(num_correct)

        tasks = [asyncio.create_task(eval(i)) for i in range(len(test_data))]
        await asyncio.gather(*tasks)

        def estimator(n: int, c: int, k: int) -> float:
            """Calculates 1 - comb(n - c, k) / comb(n, k)."""
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        all_pass_1 = sum(
            [estimator(n, c, 1) for n, c in zip(all_total, all_correct)]
        ) / len(all_total)
        easy_pass_1 = sum(
            [estimator(n, c, 1) for n, c in zip(easy_total, easy_correct)]
        ) / (len(easy_total) + 1e-6)
        medium_pass_1 = sum(
            [estimator(n, c, 1) for n, c in zip(medium_total, medium_correct)]
        ) / (len(medium_total) + 1e-6)
        hard_pass_1 = sum(
            [estimator(n, c, 1) for n, c in zip(hard_total, hard_correct)]
        ) / (len(hard_total) + 1e-6)

        self.eval_metrics.append(
            ("eval/overlong_ratio", 1.0 * sum(overlong_ratio) / len(overlong_ratio))
        )
        self.eval_metrics.append(
            ("eval/pass@group_size", 1.0 * sum(pass_gs) / len(pass_gs))
        )

        avg_comp_len = sum(temp_completion_lengths) / len(temp_completion_lengths)

        self.eval_metrics.append(("eval/pass_1", all_pass_1))
        self.eval_metrics.append(("eval/easy_pass_1", easy_pass_1))
        self.eval_metrics.append(("eval/medium_pass_1", medium_pass_1))
        self.eval_metrics.append(("eval/hard_pass_1", hard_pass_1))
        self.eval_metrics.append(("eval/completion_length", avg_comp_len))

        self.eval_metrics.append(
            (
                "eval/correct_completion_length",
                sum(correct_completion_lengths) / len(correct_completion_lengths),
            )
        )
        self.eval_metrics.append(
            (
                "eval/incorrect_completion_length",
                sum(incorrect_completion_lengths) / len(incorrect_completion_lengths),
            )
        )
        print("STATS", self.eval_metrics)

        return

    async def offline_filter(self):
        rprint("OFFLINE FILTERING")
        train_data = self.train

        sema = asyncio.Semaphore(256)

        async def filter_temp(curr_step):
            async with sema:
                item = train_data[curr_step]
                item["split"] = "train"
                item["idx"] = curr_step
                rprint("OFFLINE FILTERING INDEX", curr_step)
                scored_data, _ = await self.collect_trajectories(item)

                scores = scored_data["scores"]
                num_correct = sum(1 for x in scores if math.isclose(x, 1.0))

                async with self.lock2:
                    if num_correct == self.config.group_size:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        file_path = os.path.join(script_dir, "perfect_indices.txt")
                        with open(file_path, "a") as f:
                            f.write(json.dumps(curr_step) + "\n")

        tasks = [asyncio.create_task(filter_temp(i)) for i in range(len(train_data))]
        await asyncio.gather(*tasks)
        print("DONE")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]

        async with self.lock:
            if self.wandb_step == 1 and self.curr_step > 5:
                self.wandb_step = self.curr_step
            cnt = sum(
                (i != 0 and i != self.config.group_size)
                for i in self.temp_metrics["num_correct"]
            )
            print(
                "TEMP METRICS INDICES: ",
                self.temp_metrics["indices"],
                "\n",
                self.temp_metrics["num_correct"],
                cnt,
                len(self.temp_metrics["num_correct"]),
            )
            for k, v in wandb_metrics.items():
                print(k, v)

            old_idx = 0
            idx = 0
            good_updates = 0

            old_completion_idx = 0
            new_completion_idx = 0

            # assert len(self.temp_metrics["num_correct"]) == len(self.completion_lengths),
            # f"TEMP METRICS {len(self.temp_metrics['num_correct'])} and
            # COMPLETION LENGTHS {len(self.completion_lengths)} MUST BE SAME LENGTH"
            print(
                "TEMP METRICS LENGTHS",
                len(self.temp_metrics["num_correct"]),
                "COMPLETION LENGTHS",
                len(self.completion_lengths),
            )
            print("BATCH SZ", self.config.batch_size)

            while idx < len(self.temp_metrics["num_correct"]):
                if (
                    self.temp_metrics["num_correct"][idx] != 0
                    and self.temp_metrics["num_correct"][idx] != self.config.group_size
                ):
                    good_updates += 1
                idx += 1
                if good_updates == self.config.batch_size // self.config.group_size:
                    wandb_metrics["train_rewards/rewards"] = sum(
                        self.train_metrics["rewards"][old_idx:idx]
                    ) / len(self.train_metrics["rewards"][old_idx:idx])
                    wandb_metrics["train_rewards/overlong_ratio"] = sum(
                        self.train_metrics["overlong_ratio"][old_idx:idx]
                    ) / len(self.train_metrics["overlong_ratio"][old_idx:idx])
                    wandb_metrics["train_rewards/pass@group_size"] = sum(
                        self.train_metrics["pass_gs"][old_idx:idx]
                    ) / len(self.train_metrics["pass_gs"][old_idx:idx])
                    wandb_metrics["train_rewards/completion_lengths"] = sum(
                        self.train_metrics["completion_lengths"][old_idx:idx]
                    ) / len(self.train_metrics["completion_lengths"][old_idx:idx])
                    wandb_metrics["train_rewards/correct_completion_lengths"] = sum(
                        self.train_metrics["correct_completion_lengths"][old_idx:idx]
                    ) / sum(self.temp_metrics["num_correct"][old_idx:idx])
                    wandb_metrics["train_rewards/incorrect_completion_lengths"] = sum(
                        self.train_metrics["incorrect_completion_lengths"][old_idx:idx]
                    ) / (
                        self.config.group_size * (idx - old_idx)
                        - sum(self.temp_metrics["num_correct"][old_idx:idx])
                    )

                    new_completion_idx += self.config.batch_size

                    assert old_completion_idx <= len(self.completion_lengths), (
                        f"OLD COMPLETION IDX {old_completion_idx} and "
                        f"COMPLETION LENGTHS {len(self.completion_lengths)} MUST BE SMALLER"
                    )

                    wandb_metrics["train/completion_lengths"] = sum(
                        self.completion_lengths[old_completion_idx:new_completion_idx]
                    ) / len(
                        self.completion_lengths[old_completion_idx:new_completion_idx]
                    )
                    wandb_metrics["train/completion_lengths_std"] = np.std(
                        self.completion_lengths[old_completion_idx:new_completion_idx]
                    )
                    wandb_metrics["train/completion_lengths_max"] = np.max(
                        self.completion_lengths[old_completion_idx:new_completion_idx]
                    )
                    wandb_metrics["train/completion_lengths_min"] = np.min(
                        self.completion_lengths[old_completion_idx:new_completion_idx]
                    )
                    wandb_metrics["train/completion_lengths_p95"] = (
                        np.array(
                            self.completion_lengths[
                                old_completion_idx:new_completion_idx
                            ]
                        )
                        > (0.95 * self.max_token_len)
                    ).mean()

                    if self.wandb_prepend is not None:
                        wandb_metrics = {
                            f"{self.wandb_prepend}_{k}": v
                            for k, v in wandb_metrics.items()
                        }
                    print("WANDB LOG")
                    wandb.log(wandb_metrics, step=self.wandb_step, commit=True)
                    wandb_metrics = {}

                    good_updates = 0
                    self.wandb_step += 1
                    old_idx = idx
                    old_completion_idx = new_completion_idx

            self.completion_lengths = self.completion_lengths[old_completion_idx:]
            self.temp_metrics["indices"] = self.temp_metrics["indices"][old_idx:]
            self.temp_metrics["num_correct"] = self.temp_metrics["num_correct"][
                old_idx:
            ]
            self.train_metrics["rewards"] = self.train_metrics["rewards"][old_idx:]
            self.train_metrics["overlong_ratio"] = self.train_metrics["overlong_ratio"][
                old_idx:
            ]
            self.train_metrics["pass_gs"] = self.train_metrics["pass_gs"][old_idx:]
            self.train_metrics["completion_lengths"] = self.train_metrics[
                "completion_lengths"
            ][old_idx:]
            self.train_metrics["correct_completion_lengths"] = self.train_metrics[
                "correct_completion_lengths"
            ][old_idx:]
            self.train_metrics["incorrect_completion_lengths"] = self.train_metrics[
                "incorrect_completion_lengths"
            ][old_idx:]

        print("WANDB STEPS vs STATUS STEP", self.wandb_step, self.curr_step)

        self.eval_metrics = list()

        for i, server in enumerate(self.server.servers):
            server_wandb_metrics = await server.wandb_metrics({}, f"server_{i}")

        wandb_metrics = await self.create_rollout_table(wandb_metrics)
        wandb_metrics = self.perf_stats(wandb_metrics)
        self.rollouts_for_wandb = []

        if self.config.use_wandb:
            if self.wandb_prepend is not None:
                wandb_metrics = {
                    f"{self.wandb_prepend}_{k}": v for k, v in wandb_metrics.items()
                }
            # add server metrics to wandb without prepend to collate them all
            wandb_metrics.update(server_wandb_metrics)
            wandb.log(wandb_metrics, step=self.curr_step, commit=True)

    async def setup(self):
        """Setup the environment"""
        if self.config.dataset_name == "deepmind":
            self.train = load_dataset(
                "deepmind/code_contests", split="train"
            )  # CHANGE
        else:
            self.train = load_dataset(
                "NousResearch/RLVR_Coding_Problems", split="train"
            )
        test = load_dataset("NousResearch/lcb_test", split="test")
        self.test = []
        for problem in test:
            self.test.append(problem)
            self.test[-1]["idx"] = len(self.test) - 1
            self.test[-1]["split"] = "test"

        self.iter = 0  # CHANGE
        self.lock = asyncio.Lock()
        self.lock2 = asyncio.Lock()

        self.wandb_step = 1

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "perfect_indices.txt")
        with open(file_path, "r") as f:
            for line in f:
                a = int(line.strip())
                self.blacklist.add(a)

        self.good_indices = []
        if self.config.dataset_name != "deepmind":
            arr_short = []
            for i in range(len(self.train)):
                if i not in self.blacklist:
                    arr_short.append(i)
            print("ARR SHORT LENGTH", len(arr_short))
            temp_arr = arr_short * 100
            self.deq = deque(temp_arr[self.config.start_idx :])
        else:
            self.good_indices = [i for i in range(10000)]
            for i in range(10000):
                self.deq.append(i)

        rprint("NUM FILTERED EXAMPLES:", len(self.deq))
        rprint(self.config.batch_size)

        """
        rprint("BEFORE SETUP, do blacklisting")
        await self.offline_filter()
        rprint("FINISH blacklisting")
        """

        """
        st = time.time()
        await self.evaluate() ### CHANGE
        ed = time.time()
        rprint("ELAPSED TIME FOR EVALUATION: ", ed-st)
        """

    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out
        """
        async with self.lock:
            cur_idx = self.deq.popleft()
            while cur_idx in self.blacklist:
                print("IN BLACKLIST, SKIPPING", cur_idx)
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
                    rprint("FINISH REASON", item[3])
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
                    async with async_semaphore:
                        res, metadata = await run_test.remote.aio(test_cases, code)
                except modal.exception.RemoteError:
                    res = [False]
                    rprint("index", cur_id)
                    rprint(test_cases["tests"]["fn_name"])
                    metadata = {"error": "segmentation fault"}
                    rprint("FAULT")
                    rprint("code:\n", code)
                    rprint(metadata)
                except Exception as f:
                    rprint("index", cur_id)
                    rprint(test_cases["tests"]["fn_name"])
                    rprint("except:", code)
                    res = [False]
                    metadata = {"error": "segmentation fault"}
                    rprint("NON SEGFAULT")
                    rprint("FAULT", f)
                for x in res:
                    if not isinstance(x, (int, bool)):
                        rprint("WARNING")
                        rprint(res)
                if set(res) == {True}:
                    scores["scores"].append(1.0)
                else:
                    scores["scores"].append(-1.0)
                    rprint("index", cur_id)
                    rprint(metadata)

                errors.append(metadata)

        return scores, (codes[0], errors[0], lengths[0])


if __name__ == "__main__":
    print(system_prompt)
    CodingEnv.cli()
