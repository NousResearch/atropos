"""
This file contains code inspired by and adapted from the Open-Reasoner-Zero project.
Original Repository: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
"""

import asyncio
import os
import random
import re
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import wandb
from datasets import load_dataset

# Set up logging for debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
    ServerBaseline,
)
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

prompt_format = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant "
    "first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning "
    "process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}\nAssistant: <think>"
)

problem_format = """You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
This is the problem:
{problem}
"""  # noqa: E501

stop_list = ["User:", "Human:", "Assistant:", "</answer>"]


class RSConfig(BaseEnvConfig):
    run_evaluation: bool = Field(True, description="If this should run evaluation")
    mask_too_long_completions: bool = Field(
        True, description="If this should mask too long completions"
    )
    percent_length_penalty: float = Field(
        0.0, description="The percentage of items to have length penalty"
    )
    start_tok_length: int = Field(
        8192,
        description="The starting length of the token length, scaled linearly to the max_token_length",
    )


def score_answer(gold, resp) -> Optional[bool]:
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    matches = pattern.findall(resp)
    resp = matches[-1] if matches else None
    if resp is None:
        return False
    try:
        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    except (Exception, TimeoutException, KeyError, TypeError, NotImplementedError):
        return None
    if len(gold_parsed) != 0:
        try:
            answer_parsed = parse(
                resp,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
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
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            # Can't parse, so we skip
            return None
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            return verify(answer_parsed, gold_parsed)
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            return None
    return None


class MathEnv(BaseEnv):

    name = "math"
    env_config_cls = RSConfig

    def __init__(
        self,
        config: RSConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
        slurm=True,
        testing=False,
    ):
        print("Initializing MathEnv")
        print(f"Slurm: {slurm}, Testing: {testing}")
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.mp_executor = ProcessPoolExecutor(64)
        self.percent_overanswer = list()
        self.correct_answer_len = list()
        self.incorrect_answer_len = list()
        self.normal_rollouts = list()
        self.pass_at_groupsize = list()
        self.iter = 0
        
        # Debug: Print distillation config
        print("=" * 60)
        print("[MATH_DEBUG] DISTILLATION CONFIGURATION:")
        print(f"[MATH_DEBUG]   distillation_enabled = {config.distillation_enabled}")
        print(f"[MATH_DEBUG]   teacher_base_url = {config.teacher_base_url}")
        print(f"[MATH_DEBUG]   teacher_model_name = {getattr(config, 'teacher_model_name', 'N/A')}")
        print(f"[MATH_DEBUG]   teacher_top_logprobs = {getattr(config, 'teacher_top_logprobs', 'N/A')}")
        print("=" * 60)

    @classmethod
    def config_init(cls) -> Tuple[RSConfig, List[APIServerConfig]]:
        # Allow configuration via environment variables for running multiple instances
        model_name = os.environ.get("MATH_ENV_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        rollout_url = os.environ.get("MATH_ENV_ROLLOUT_URL", "http://localhost:8000")
        vllm_url = os.environ.get("MATH_ENV_VLLM_URL", "http://localhost:9001/v1")
        wandb_name = os.environ.get("MATH_ENV_WANDB_NAME", "math-zero-env")
        max_token_length = int(os.environ.get("MATH_ENV_MAX_TOKENS", "32000"))
        worker_timeout = float(os.environ.get("MATH_ENV_WORKER_TIMEOUT", "1500"))
        
        env_config = RSConfig(
            tokenizer_name=model_name,
            group_size=8,
            use_wandb=True,
            rollout_server_url=rollout_url,
            total_steps=120,
            batch_size=64,
            steps_per_eval=20,
            max_token_length=max_token_length,
            start_tok_length=max_token_length,
            wandb_name=wandb_name,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            max_num_workers_per_node=24,
            worker_timeout=worker_timeout,
        )
        server_configs = [
            APIServerConfig(
                model_name=model_name,
                base_url=vllm_url,
                api_key="x",
                num_requests_for_eval=256,
                server_type="vllm",
                weight=1.0,
            )
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = dict()
        if len(self.pass_at_groupsize) > 0:
            wandb_metrics["train/pass_at_groupsize"] = sum(
                self.pass_at_groupsize
            ) / len(self.pass_at_groupsize)
            self.pass_at_groupsize = list()
        if len(self.percent_correct_buffer) > 0:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
            wandb_metrics["train/percent_overanswer"] = sum(
                self.percent_overanswer
            ) / len(self.percent_overanswer)
            self.percent_overthink = list()
            self.percent_overanswer = list()
            self.percent_correct_buffer = list()
        if len(self.correct_answer_len) > 0:
            wandb_metrics["train/avg_correct_answer_len"] = sum(
                self.correct_answer_len
            ) / len(self.correct_answer_len)
            self.correct_answer_len = list()
        if len(self.incorrect_answer_len) > 0:
            wandb_metrics["train/avg_incorrect_answer_len"] = sum(
                self.incorrect_answer_len
            ) / len(self.incorrect_answer_len)
            self.incorrect_answer_len = list()
        # create tables
        if len(self.normal_rollouts) > 0:
            table = wandb.Table(columns=["problem", "solution", "answer", "score"])
            for group in self.normal_rollouts:
                table.add_data(group[0], group[1], group[2], group[3])
            wandb_metrics["train/normal_rollouts"] = table
        wandb_metrics["train/iter"] = self.iter
        curr_length = self.config.max_token_length - self.config.start_tok_length
        curr_length = int(curr_length * (self.curr_step / self.config.total_steps))
        curr_length += self.config.start_tok_length
        wandb_metrics["train/curr_token_length"] = curr_length
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(
            seed=42
        )
        aime_test_data = load_dataset("HuggingFaceH4/aime_2024", split="train")
        math500_test_data = load_dataset("HuggingFaceH4/math-500", split="test")
        amc_test_data = load_dataset("math-ai/amc23", split="test")
        minerva_test_data = load_dataset("math-ai/minervamath", split="test")
        olympiad_test_data = load_dataset("math-ai/olympiadbench", split="test")
        self.test = list()
        for name, t_dataset in zip(
            ["aime24", "math500"], [aime_test_data, math500_test_data]
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["problem"])
                        ),
                        item["answer"],
                        name,
                    )
                )
        for name, t_dataset in zip(
            ["amc23", "minerva"],
            [amc_test_data, minerva_test_data],
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["question"])
                        ),
                        item["answer"],
                        name,
                    )
                )
        for name, t_dataset in zip(["olympiad"], [olympiad_test_data]):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["question"])
                        ),
                        item["final_answer"][0],
                        name,
                    )
                )
        
        # Debug: Test teacher connectivity if distillation is enabled
        if self.config.distillation_enabled and self.config.teacher_base_url:
            await self._test_teacher_connectivity()
        
        return
    
    async def _test_teacher_connectivity(self):
        """Test if the teacher model API is reachable."""
        print("=" * 60)
        print("[MATH_DEBUG] TESTING TEACHER CONNECTIVITY...")
        print(f"[MATH_DEBUG] Teacher URL: {self.config.teacher_base_url}")
        print(f"[MATH_DEBUG] Teacher Model: {getattr(self.config, 'teacher_model_name', 'default')}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Health check
                health_url = self.config.teacher_base_url.replace("/v1", "") + "/health"
                print(f"[MATH_DEBUG] Testing health endpoint: {health_url}")
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        print(f"[MATH_DEBUG] Health check status: {resp.status}")
                        if resp.status == 200:
                            print("[MATH_DEBUG] ✓ Teacher health check PASSED")
                        else:
                            print(f"[MATH_DEBUG] ✗ Teacher health check FAILED: {await resp.text()}")
                except Exception as e:
                    print(f"[MATH_DEBUG] ✗ Teacher health check ERROR: {e}")
                
                # Test 2: Models endpoint
                models_url = f"{self.config.teacher_base_url}/models"
                print(f"[MATH_DEBUG] Testing models endpoint: {models_url}")
                try:
                    async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        print(f"[MATH_DEBUG] Models endpoint status: {resp.status}")
                        if resp.status == 200:
                            data = await resp.json()
                            models = [m.get("id", m) for m in data.get("data", [])]
                            print(f"[MATH_DEBUG] ✓ Available models: {models}")
                        else:
                            print(f"[MATH_DEBUG] ✗ Models endpoint FAILED: {await resp.text()}")
                except Exception as e:
                    print(f"[MATH_DEBUG] ✗ Models endpoint ERROR: {e}")
                
                # Test 3: Simple completion test
                completions_url = f"{self.config.teacher_base_url}/completions"
                teacher_model = getattr(self.config, 'teacher_model_name', 'default')
                test_payload = {
                    "model": teacher_model,
                    "prompt": "Hello",
                    "max_tokens": 5,
                    "logprobs": 5,
                    "echo": True,
                }
                print(f"[MATH_DEBUG] Testing completions endpoint: {completions_url}")
                print(f"[MATH_DEBUG] Test payload: {test_payload}")
                try:
                    async with session.post(
                        completions_url,
                        json=test_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        print(f"[MATH_DEBUG] Completions status: {resp.status}")
                        resp_text = await resp.text()
                        if resp.status == 200:
                            print(f"[MATH_DEBUG] ✓ Teacher completions WORKING!")
                            print(f"[MATH_DEBUG] Response preview: {resp_text[:500]}")
                        else:
                            print(f"[MATH_DEBUG] ✗ Teacher completions FAILED: {resp_text[:500]}")
                except Exception as e:
                    print(f"[MATH_DEBUG] ✗ Teacher completions ERROR: {e}")
                    
        except Exception as e:
            print(f"[MATH_DEBUG] ✗ Teacher connectivity test FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)

    async def rollout_and_score_eval(self, question, answer, subset):
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.completion(
                prompt=question,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
                split="eval",
                stop=stop_list,
            )
        loop = asyncio.get_event_loop()
        gold = "\\boxed{" + answer + "}" if "\\boxed" not in answer else answer
        resp = completion.choices[0].text
        if completion.choices[0].finish_reason == "stop":
            if ("</answer>" not in completion.choices[0].text) and (
                "<answer>" in completion.choices[0].text
            ):
                # assume it stopped on </answer>
                resp = resp + "</answer>"
        task = loop.run_in_executor(self.mp_executor, score_answer, gold, resp)
        reward = await task
        if reward is None:
            return 0, subset
        score = 1 if reward else 0
        return score, subset

    async def evaluate(self, *args, **kwargs):
        if not self.config.run_evaluation:
            return
        import time

        start_time = time.time()

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(item[0], item[1], item[2]))
        parsing_data = await tqdm_asyncio.gather(*eval_tasks)
        task_lists = dict()
        for score, subset in parsing_data:
            if subset not in task_lists:
                task_lists[subset] = list()
            task_lists[subset].append(score)

        # Build metrics dictionary for saving
        metrics = {}

        # Now get the average per subset
        for subset, scores in task_lists.items():
            accuracy = sum(scores) / len(scores)
            metrics[f"{subset}_accuracy"] = accuracy
            metrics[f"{subset}_total"] = len(scores)
            metrics[f"{subset}_correct"] = sum(scores)
            self.eval_metrics.append((f"eval/{subset}_percent_correct", accuracy))

        # overall score
        all_scores = []
        for subset, score in task_lists.items():
            all_scores.extend(score)
        overall_accuracy = sum(all_scores) / len(all_scores)
        metrics["overall_accuracy"] = overall_accuracy
        metrics["overall_total"] = len(all_scores)
        metrics["overall_correct"] = sum(all_scores)
        self.eval_metrics.append(("eval/overall_percent_correct", overall_accuracy))

        end_time = time.time()

        # Print results to console
        print("\n" + "=" * 60)
        print("Math Zero Evaluation Results")
        print("=" * 60)
        print(
            f"Overall Accuracy: {overall_accuracy:.2%} ({sum(all_scores)}/{len(all_scores)})"
        )
        print("\nPer-subset breakdown:")
        for subset, scores in sorted(task_lists.items()):
            acc = sum(scores) / len(scores)
            print(f"  {subset}: {acc:.2%} ({sum(scores)}/{len(scores)})")
        print("=" * 60 + "\n")

        # Save results to disk
        await self.evaluate_log(
            metrics=metrics,
            task_name="math_zero",
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "max_tokens": self.config.max_token_length,
                "temperature": 0.0,
            },
        )

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        thinking_len = self.config.max_token_length
        user_prompt = prompt_format.format(
            prompt=problem_format.format(problem=item[0])
        )
        thinking_len = thinking_len - len(self.tokenizer.encode(user_prompt))
        curr_length = self.config.max_token_length - self.config.start_tok_length
        curr_length = int(curr_length * (self.curr_step / self.config.total_steps))
        curr_length += self.config.start_tok_length
        thinking_len = min(thinking_len, curr_length)

        # ============================================================================
        # MANAGED SERVER USAGE - Automatic Token & Logprob Tracking
        # ============================================================================
        # This is the RECOMMENDED approach for handling inference in Atropos environments.
        # ManagedServer automatically:
        # 1. Tokenizes the prompt and completion
        # 2. Applies proper masking (-100 for prompt tokens, actual IDs for completion)
        # 3. Applies proper logprob masking (1.0 for prompt, actual values for completion)
        # 4. Ensures perfect alignment between tokens and logprobs
        # 5. Handles the n>1 case (multiple completions from same prompt)
        #
        # Benefits over manual handling:
        # - No manual tokenization needed
        # - No off-by-one errors
        # - No manual masking calculations
        # - Guaranteed correct alignment
        # - Clean, simple code
        #
        # See: atroposlib/envs/server_handling/MANAGED_SERVER.md for full documentation
        # ============================================================================

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            # Call completion as usual, but through the managed server wrapper
            # This returns a standard OpenAI-compatible Completion object
            completion = await managed.completion(
                prompt=user_prompt,
                n=self.config.group_size,  # Generate multiple completions for GRPO
                max_tokens=thinking_len,
                temperature=1.0,
                top_p=1.0,
                stop=stop_list,
            )

            # Get the tracked sequences with aligned tokens and logprobs
            state = managed.get_state()
            nodes = state["nodes"]  # List of SequenceNode objects, one per completion

        # ============================================================================
        # Extract Pre-Computed Data from SequenceNodes
        # ============================================================================
        # Each SequenceNode contains:
        # - full_text: Complete text (prompt + completion)
        # - tokens: Full unmasked token sequence [1, 2, 3, ..., N]
        # - masked_tokens: Training format [-100, -100, ..., -100, actual, actual, ...]
        # - logprobs: Training format [1.0, 1.0, ..., 1.0, -0.5, -0.3, ...]
        # - metadata: Contains finish_reason, etc.
        #
        # Note: -100 is used for prompt token masking (standard PyTorch ignore_index)
        #       1.0 is used for prompt logprob masking (obviously bad probability)
        # ============================================================================

        to_score = list()
        to_backlog = list()
        for i, (choice, node) in enumerate(zip(completion.choices, nodes)):
            to_score.append(
                (
                    node.full_text,  # Complete text (prompt + completion)
                    item[1],  # Ground truth answer
                    choice.finish_reason,  # "stop" or "length"
                    node.tokens,  # Full unmasked tokens [prompt + completion]
                    node.masked_tokens,  # [-100, ..., -100, tok1, tok2, ...]
                    node.logprobs,  # [1.0, ..., 1.0, logp1, logp2, ...]
                )
            )
        to_postprocess = await self.score(to_score)
        if to_postprocess is None:
            return None, to_backlog
        if all(
            [to_postprocess["scores"][0] == score for score in to_postprocess["scores"]]
        ):
            return None, to_backlog
        self.normal_rollouts.append(
            (
                prompt_format.format(prompt=problem_format.format(problem=item[0])),
                to_postprocess["messages"][0][-1]["content"],
                item[1],
                to_postprocess["scores"][0],
            )
        )
        if len(self.normal_rollouts) > self.config.num_rollouts_to_keep:
            self.normal_rollouts.pop(0)
        print(f"Collected {len(to_postprocess['scores'])} trajectories")
        return to_postprocess, to_backlog

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["overrides"] = list()
        scores["messages"] = list()
        scores["inference_logprobs"] = list()
        gold = rollout_group_data[0][1]
        loop = asyncio.get_event_loop()
        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            scores["overrides"].append(dict())
            resp = item[0]
            finish_reason = item[2]  # Now a clean string like "stop" or "length"
            # ManagedServer already provides properly formatted data
            tokens = item[3]  # Full token sequence
            masks = item[4]  # Masked tokens (already formatted)
            inf_logp = item[5]  # Logprobs (already formatted)

            if finish_reason == "length":
                reward = False
                if self.config.mask_too_long_completions:
                    scores["overrides"][-1]["set_advantage_to_zero"] = True
            else:
                task = loop.run_in_executor(self.mp_executor, score_answer, gold, resp)
                reward = await task
                if reward is None:
                    return None

            assert len(inf_logp) == len(
                masks
            ), f"{len(inf_logp)}, {len(masks)} mismatch"
            user_prompt = resp.split("<think>")[0]
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": resp[len(user_prompt) :]},
            ]
            # remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue
            if (finish_reason == "length") and (
                not self.config.mask_too_long_completions
            ):
                scores["overrides"][-1]["set_advantage_to_zero"] = True
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            scores["messages"].append(messages)
            scores["inference_logprobs"].append(inf_logp)
            if len(scores["tokens"]) >= self.config.group_size:
                break
        if any([score == 1.0 for score in scores["scores"]]):
            self.pass_at_groupsize.append(1.0)
        else:
            self.pass_at_groupsize.append(0.0)
        if len(scores["tokens"]) < self.config.group_size:
            # We don't have enough data to score
            return None
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))
        self.percent_overanswer.extend(
            [item[2] == "length" for item in rollout_group_data]
        )
        # check if all the same
        # print(scores['scores'])
        # Fill in the correct/incorrect lens after so we're only looking at actual training data
        self.correct_answer_len.extend(
            [
                len(scores["tokens"][i])
                for i in range(len(scores["scores"]))
                if scores["scores"][i] == 1.0
            ]
        )
        self.incorrect_answer_len.extend(
            [
                len(scores["tokens"][i])
                for i in range(len(scores["scores"]))
                if (scores["scores"][i] == -1.0)
                and (not scores["overrides"][i].get("set_advantage_to_zero", False))
            ]
        )
        
        # Debug: Log scored group creation
        print(f"[MATH_DEBUG] Created ScoredDataGroup with {len(scores['tokens'])} sequences")
        print(f"[MATH_DEBUG]   Scores: {scores['scores']}")
        print(f"[MATH_DEBUG]   Token lengths: {[len(t) for t in scores['tokens']]}")
        print(f"[MATH_DEBUG]   Has onpolicydistill_logprobs: {'onpolicydistill_logprobs' in scores}")
        
        return scores
    
    async def handle_send_to_api(
        self,
        scored_data,
        item=None,
        do_send_to_api: bool = True,
        abort_on_any_max_length_exceeded: bool = True,
    ):
        """Override to add debugging for distillation."""
        print(f"[MATH_DEBUG] handle_send_to_api called")
        print(f"[MATH_DEBUG]   distillation_enabled: {self.config.distillation_enabled}")
        print(f"[MATH_DEBUG]   teacher_base_url: {self.config.teacher_base_url}")
        
        if isinstance(scored_data, list):
            for i, group in enumerate(scored_data):
                if group:
                    has_distill = 'onpolicydistill_logprobs' in group and group.get('onpolicydistill_logprobs') is not None
                    print(f"[MATH_DEBUG]   Group {i}: {len(group.get('tokens', []))} seqs, has_distill_logprobs={has_distill}")
        elif scored_data:
            has_distill = 'onpolicydistill_logprobs' in scored_data and scored_data.get('onpolicydistill_logprobs') is not None
            print(f"[MATH_DEBUG]   Single group: {len(scored_data.get('tokens', []))} seqs, has_distill_logprobs={has_distill}")
        
        # Call parent implementation which does the actual distillation fetch
        result = await super().handle_send_to_api(
            scored_data, item, do_send_to_api, abort_on_any_max_length_exceeded
        )
        
        # Debug: Check if distillation was added after parent call
        if isinstance(scored_data, list):
            for i, group in enumerate(scored_data):
                if group:
                    has_distill = 'onpolicydistill_logprobs' in group and group.get('onpolicydistill_logprobs') is not None
                    print(f"[MATH_DEBUG]   AFTER: Group {i} has_distill_logprobs={has_distill}")
        elif scored_data:
            has_distill = 'onpolicydistill_logprobs' in scored_data and scored_data.get('onpolicydistill_logprobs') is not None
            print(f"[MATH_DEBUG]   AFTER: Single group has_distill_logprobs={has_distill}")
        
        return result

    async def get_next_item(self):
        while True:
            next_item = self.train[self.iter % len(self.train)]
            self.iter += 1
            prompt = next_item["question"]
            try:
                answer = (
                    ("\\boxed{" + next_item["final_answer"] + "}")
                    if "\\boxed" not in next_item["final_answer"]
                    else next_item["final_answer"]
                )
                break
            except TypeError:
                print(
                    f"Error in getting next item, trying again, "
                    f"data: {next_item['question']} -> {next_item['final_answer']}"
                )
        return (prompt, answer, "normal")


if __name__ == "__main__":
    MathEnv.cli()
