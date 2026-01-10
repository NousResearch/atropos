"""
Agent Trace Environment for generating coding agent traces with logprobs.

This environment uses Ollama's native API to generate code solutions with
full logprobs tracking, suitable for training RL agents on code generation tasks.

The pipeline:
1. Load coding problems from dataset
2. Generate code solutions using Ollama with logprobs
3. Execute code in sandboxed environment (Modal)
4. Score solutions based on test case results
5. Output agent traces with tokens, logprobs, and rewards
"""

import asyncio
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import regex as re
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
from atroposlib.type_definitions import AgentStep, GameHistory, Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Try to import Modal, fallback to local executor
try:
    import modal
    run_test = modal.Function.from_name("joeli-lcb", "run_test")
    USE_MODAL = True
except Exception:
    from .local_executor import run_test_local
    run_test = None
    USE_MODAL = False


# System prompt for code generation
SYSTEM_PROMPT = """You are an expert Python programmer. You will be given a coding problem and will generate a correct Python program that solves it.

For each problem:
1. Analyze the requirements carefully
2. Think through your approach step by step
3. Write clean, efficient code
4. Test your logic mentally before finalizing

Enclose your final code within ```python and ``` delimiters."""

# Formatting instructions
FORMATTING_WITH_STARTER = (
    "Use the following starter code to write your solution. "
    "Complete the function as specified."
)

FORMATTING_STDIN_STDOUT = (
    "Read inputs from stdin, solve the problem, and write the answer to stdout. "
    "Enclose your code within ```python and ``` delimiters."
)

# Semaphore for code execution calls
async_semaphore = asyncio.Semaphore(100)


def build_prompt(question: str, problem_type: str, starter_code: Optional[str] = None) -> str:
    """Build the user prompt for a coding problem."""
    prompt = f"### Problem:\n{question}\n\n"

    if problem_type == "func" and starter_code:
        prompt += f"### Instructions:\n{FORMATTING_WITH_STARTER}\n\n"
        prompt += f"### Starter Code:\n```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Instructions:\n{FORMATTING_STDIN_STDOUT}\n\n"

    prompt += "### Your Solution:\n"
    return prompt


class AgentTraceConfig(BaseEnvConfig):
    """Configuration for the Agent Trace Environment."""

    dataset_name: str = Field(
        "NousResearch/RLVR_Coding_Problems",
        description="Dataset to use for coding problems",
    )
    temperature: float = Field(0.7, description="Sampling temperature for generation")
    eval_temperature: float = Field(0.3, description="Temperature during evaluation")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    max_code_tokens: int = Field(4096, description="Maximum tokens for code generation")
    top_logprobs: int = Field(5, description="Number of top logprobs to collect")
    collect_logprobs: bool = Field(True, description="Whether to collect logprobs")
    output_dir: str = Field("agent_traces", description="Directory for output traces")
    use_ollama: bool = Field(True, description="Use Ollama server for generation")


class AgentTraceEnv(BaseEnv):
    """
    Environment for generating agent traces for code generation tasks.

    This environment generates coding solutions using an LLM (preferably Ollama
    with logprobs support), executes the code in a sandbox, and outputs
    structured agent traces with:
    - Token sequences
    - Logprobs for each token
    - Execution results and scores
    - Full conversation history
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_id = 0
        self.completed_traces = []
        self.cur_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Metrics tracking
        self.metrics = {
            "total_problems": 0,
            "correct_solutions": 0,
            "avg_completion_length": [],
            "avg_logprob": [],
        }

        self.problem_queue = deque()
        self.blacklist = set()

    @classmethod
    def config_init(cls) -> Tuple[AgentTraceConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = AgentTraceConfig(
            tokenizer_name="Qwen/Qwen3-14B",  # For tokenization
            group_size=4,  # Number of solutions per problem
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=-1,
            steps_per_eval=50,
            max_token_length=8192,
            wandb_name="agent_trace_deepseek",
            temperature=0.7,
            eval_temperature=0.3,
            collect_logprobs=True,
            use_ollama=True,
        )

        # Ollama Cloud configuration with DeepSeek V3.2
        server_configs = [
            APIServerConfig(
                model_name=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
                api_key=os.getenv("OLLAMA_API_KEY", ""),
                server_type="ollama",
                num_requests_for_eval=32,
                timeout=300,
                health_check=False,
            ),
        ]

        return env_config, server_configs

    async def setup(self):
        """Setup the environment and load dataset."""
        rprint("[bold green]Setting up Agent Trace Environment[/bold green]")

        # Load dataset
        self.train = load_dataset(self.config.dataset_name, split="train")
        rprint(f"Loaded {len(self.train)} training problems")

        # Try to load test set
        try:
            test = load_dataset("NousResearch/lcb_test", split="test")
            self.test = []
            for i, problem in enumerate(test):
                problem["idx"] = i
                problem["split"] = "test"
                self.test.append(problem)
            rprint(f"Loaded {len(self.test)} test problems")
        except Exception as e:
            rprint(f"[yellow]Could not load test set: {e}[/yellow]")
            self.test = []

        # Initialize problem queue
        for i in range(len(self.train)):
            if i not in self.blacklist:
                self.problem_queue.append(i)

        # Create output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.config.output_dir,
            self.cur_time,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.lock = asyncio.Lock()
        self.lock2 = asyncio.Lock()

        rprint(f"[green]Setup complete. Output dir: {self.output_dir}[/green]")

    async def get_next_item(self) -> Item:
        """Get the next coding problem to solve."""
        async with self.lock:
            if not self.problem_queue:
                # Refill queue
                for i in range(len(self.train)):
                    if i not in self.blacklist:
                        self.problem_queue.append(i)

            cur_idx = self.problem_queue.popleft()
            next_item = dict(self.train[cur_idx])

        next_item["idx"] = cur_idx
        next_item["split"] = "train"

        return next_item

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup | None, List[Item]]:
        """
        Collect agent trajectories for a coding problem.

        This generates multiple solutions, executes them, and returns
        the scored data group with tokens, masks, logprobs, and scores.
        """
        rprint(f"[cyan]Collecting trajectories for problem {item['idx']}[/cyan]")

        split = item.get("split", "train")

        # Build messages
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {
            "role": "user",
            "content": build_prompt(
                item["problem"],
                item.get("problem_type", "func"),
                item.get("starter_code"),
            ),
        }

        # Tokenize prompt for tracking
        prompt_tokens = tokenize_for_trainer(
            self.tokenizer, chat=[system_msg, user_msg]
        )

        # Determine generation parameters
        temp = self.config.eval_temperature if split == "test" else self.config.temperature
        max_tokens = self.config.max_code_tokens
        n_samples = 16 if split == "test" else self.config.group_size

        async def generate_and_score(idx: int) -> Dict[str, Any]:
            """Generate a single solution and score it."""

            # Generate completion
            chat_completion = await self.server.chat_completion(
                messages=[system_msg, user_msg],
                n=1,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self.config.top_p,
            )

            content = chat_completion.choices[0].message.content
            finish_reason = chat_completion.choices[0].finish_reason
            assistant_msg = {"role": "assistant", "content": content}

            # Extract code from response
            code = self.extract_python_code(content)

            # Score the solution
            tests = item.get("tests", {})
            if isinstance(tests, str):
                tests = json.loads(tests)

            fn_name = item.get("fn_name", "none")
            tests["fn_name"] = fn_name

            score, error_info = await self.execute_and_score(
                code, tests, item["idx"]
            )

            # Tokenize full conversation
            messages = [system_msg, user_msg, assistant_msg]
            out_dict = tokenize_for_trainer(self.tokenizer, messages, finish_reason)

            return {
                "tokens": out_dict["tokens"],
                "masks": out_dict["masks"],
                "score": score,
                "code": code,
                "error": error_info,
                "finish_reason": finish_reason,
                "content": content,
                "messages": messages,
            }

        # Generate multiple solutions in parallel
        start_time = time.time()
        tasks = [generate_and_score(i) for i in range(n_samples)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Build ScoredDataGroup
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = [r["tokens"] for r in results]
        scored_data["masks"] = [r["masks"] for r in results]
        scored_data["scores"] = [r["score"] for r in results]
        scored_data["overrides"] = [
            {"set_advantage_to_zero": r["finish_reason"] == "length"}
            for r in results
        ]

        # Calculate metrics
        num_correct = sum(1 for s in scored_data["scores"] if math.isclose(s, 1.0))
        avg_length = sum(len(r["tokens"]) for r in results) / len(results)

        rprint(f"Problem {item['idx']}: {num_correct}/{n_samples} correct, "
               f"avg_len={avg_length:.0f}, time={elapsed:.1f}s")

        # Save trace to file
        await self.save_trace(item, results, scored_data)

        # Update metrics
        async with self.lock:
            self.metrics["total_problems"] += 1
            self.metrics["correct_solutions"] += num_correct
            self.metrics["avg_completion_length"].append(avg_length)

        return scored_data, []

    def extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    async def execute_and_score(
        self, code: Optional[str], tests: Dict, problem_idx: int
    ) -> Tuple[float, Dict]:
        """Execute code and score based on test results."""
        if code is None:
            return -1.0, {"error": "No code extracted"}

        try:
            async with async_semaphore:
                test_input = {"tests": tests}

                if USE_MODAL and run_test is not None:
                    # Use Modal for sandboxed execution
                    try:
                        res, metadata = await run_test.remote.aio(test_input, code)
                    except Exception as e:
                        rprint(f"[yellow]Modal error, falling back to local: {e}[/yellow]")
                        res, metadata = await run_test_local(test_input, code)
                else:
                    # Use local executor
                    res, metadata = await run_test_local(test_input, code)

            if set(res) == {True}:
                return 1.0, metadata
            else:
                return -1.0, metadata

        except Exception as e:
            rprint(f"[red]Execution error for problem {problem_idx}: {e}[/red]")
            return -1.0, {"error": str(e)}

    async def save_trace(
        self, item: Item, results: List[Dict], scored_data: ScoredDataGroup
    ):
        """Save agent trace to output file."""
        trace = {
            "problem_idx": item["idx"],
            "problem": item.get("problem", ""),
            "problem_type": item.get("problem_type", "func"),
            "timestamp": datetime.now().isoformat(),
            "solutions": [],
        }

        for i, result in enumerate(results):
            solution = {
                "index": i,
                "score": result["score"],
                "code": result["code"],
                "content": result["content"],
                "finish_reason": result["finish_reason"],
                "token_count": len(result["tokens"]),
                "error": result.get("error"),
            }
            trace["solutions"].append(solution)

        # Summary stats
        trace["summary"] = {
            "total_solutions": len(results),
            "correct_solutions": sum(1 for r in results if r["score"] == 1.0),
            "avg_token_count": sum(len(r["tokens"]) for r in results) / len(results),
        }

        # Save to file
        trace_file = os.path.join(self.output_dir, f"trace_{item['idx']:06d}.json")
        with open(trace_file, "w") as f:
            json.dump(trace, f, indent=2)

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set."""
        if not self.test:
            rprint("[yellow]No test set available for evaluation[/yellow]")
            return

        rprint("[bold]Running evaluation...[/bold]")

        sema = asyncio.Semaphore(self.config.max_eval_workers)
        results = {"total": 0, "correct": 0, "pass_at_1": []}

        async def eval_problem(idx: int):
            async with sema:
                item = self.test[idx]
                scored_data, _ = await self.collect_trajectories(item)

                scores = scored_data["scores"]
                num_correct = sum(1 for s in scores if math.isclose(s, 1.0))

                async with self.lock2:
                    results["total"] += len(scores)
                    results["correct"] += num_correct
                    results["pass_at_1"].append(1.0 if num_correct > 0 else 0.0)

        tasks = [asyncio.create_task(eval_problem(i)) for i in range(len(self.test))]
        await asyncio.gather(*tasks)

        # Calculate metrics
        pass_at_1 = sum(results["pass_at_1"]) / len(results["pass_at_1"])
        rprint(f"[bold green]Evaluation Results:[/bold green]")
        rprint(f"  Pass@1: {pass_at_1:.2%}")
        rprint(f"  Correct: {results['correct']}/{results['total']}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        async with self.lock:
            if self.metrics["avg_completion_length"]:
                wandb_metrics["train/avg_completion_length"] = np.mean(
                    self.metrics["avg_completion_length"][-100:]
                )
            if self.metrics["total_problems"] > 0:
                wandb_metrics["train/accuracy"] = (
                    self.metrics["correct_solutions"] /
                    (self.metrics["total_problems"] * self.config.group_size)
                )

        await super().wandb_log(wandb_metrics)


class OllamaAgentTraceEnv(AgentTraceEnv):
    """
    Agent Trace Environment specifically optimized for Ollama Cloud with logprobs.

    This variant uses Ollama's native API to get detailed logprobs for
    each generated token, enabling more sophisticated RL training.

    Default configuration uses Ollama Cloud with DeepSeek V3.2.
    """

    @classmethod
    def config_init(cls) -> Tuple[AgentTraceConfig, List[APIServerConfig]]:
        """Initialize with Ollama Cloud configuration for DeepSeek V3.2."""
        env_config = AgentTraceConfig(
            tokenizer_name="Qwen/Qwen3-14B",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            temperature=0.7,
            top_logprobs=5,
            collect_logprobs=True,
            use_ollama=True,
            output_dir="deepseek_agent_traces",
        )

        # Ollama Cloud with DeepSeek V3.2
        server_configs = [
            APIServerConfig(
                model_name=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
                api_key=os.getenv("OLLAMA_API_KEY", ""),
                server_type="ollama",
                timeout=300,
                health_check=False,
            ),
        ]

        return env_config, server_configs

    async def collect_trajectories_with_logprobs(
        self, item: Item
    ) -> Tuple[ScoredDataGroup | None, List[Dict[str, Any]]]:
        """
        Collect trajectories with detailed logprobs from Ollama.

        Returns both the scored data group and raw logprobs data.
        """
        from atroposlib.envs.server_handling.ollama_server import OllamaServer

        split = item.get("split", "train")

        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {
            "role": "user",
            "content": build_prompt(
                item["problem"],
                item.get("problem_type", "func"),
                item.get("starter_code"),
            ),
        }

        messages = [system_msg, user_msg]
        temp = self.config.eval_temperature if split == "test" else self.config.temperature
        n_samples = self.config.group_size

        all_results = []
        all_logprobs_data = []

        for _ in range(n_samples):
            # Get the underlying Ollama server
            server = self.server.servers[0]
            if isinstance(server, OllamaServer):
                completion, logprobs = await server.chat_completion_with_logprobs(
                    messages=messages,
                    max_tokens=self.config.max_code_tokens,
                    temperature=temp,
                    top_p=self.config.top_p,
                    top_logprobs=self.config.top_logprobs,
                )
                all_logprobs_data.append(logprobs[0])
            else:
                completion = await self.server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.max_code_tokens,
                    temperature=temp,
                    top_p=self.config.top_p,
                )
                all_logprobs_data.append([])

            content = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            assistant_msg = {"role": "assistant", "content": content}

            # Extract and score code
            code = self.extract_python_code(content)
            tests = item.get("tests", {})
            if isinstance(tests, str):
                tests = json.loads(tests)
            tests["fn_name"] = item.get("fn_name", "none")

            score, error_info = await self.execute_and_score(code, tests, item["idx"])

            # Tokenize
            full_messages = [system_msg, user_msg, assistant_msg]
            out_dict = tokenize_for_trainer(self.tokenizer, full_messages, finish_reason)

            all_results.append({
                "tokens": out_dict["tokens"],
                "masks": out_dict["masks"],
                "score": score,
                "code": code,
                "error": error_info,
                "finish_reason": finish_reason,
                "content": content,
                "messages": full_messages,
            })

        # Build ScoredDataGroup
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = [r["tokens"] for r in all_results]
        scored_data["masks"] = [r["masks"] for r in all_results]
        scored_data["scores"] = [r["score"] for r in all_results]
        scored_data["overrides"] = [
            {"set_advantage_to_zero": r["finish_reason"] == "length"}
            for r in all_results
        ]

        # Add logprobs to the data
        scored_data["inference_logprobs"] = all_logprobs_data

        return scored_data, all_results


if __name__ == "__main__":
    # Run the environment
    OllamaAgentTraceEnv.cli()
