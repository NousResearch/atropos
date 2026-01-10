"""
Structured Agent Trace Environment with Planning-Action-Reflection loop.

This environment generates agent traces following a strict interleaved reasoning structure:
1. PLANNING - Agent analyzes the problem and plans approach
2. ACTION - Agent writes code to solve the problem
3. REFLECTION - Agent reviews results and iterates if needed

Each trace captures the full reasoning chain with logprobs for RL training.
"""

import asyncio
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Local executor for code execution
try:
    import modal
    run_test = modal.Function.from_name("joeli-lcb", "run_test")
    USE_MODAL = True
except Exception:
    from .local_executor import run_test_local
    run_test = None
    USE_MODAL = False


class AgentPhase(Enum):
    """Phases of the agent reasoning loop."""
    PLANNING = "planning"
    ACTION = "action"
    REFLECTION = "reflection"


@dataclass
class AgentStep:
    """A single step in the agent trace."""
    phase: AgentPhase
    content: str
    tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """Complete agent trace with planning-action-reflection structure."""
    problem_idx: int
    problem: str
    steps: List[AgentStep] = field(default_factory=list)
    final_code: Optional[str] = None
    score: float = 0.0
    execution_result: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    num_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "problem_idx": self.problem_idx,
            "problem": self.problem,
            "steps": [
                {
                    "phase": step.phase.value,
                    "content": step.content,
                    "token_count": len(step.tokens),
                    "avg_logprob": np.mean(step.logprobs) if step.logprobs else 0.0,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
            "final_code": self.final_code,
            "score": self.score,
            "execution_result": self.execution_result,
            "total_tokens": self.total_tokens,
            "num_iterations": self.num_iterations,
        }


# System prompts for each phase
SYSTEM_PROMPT = """You are an expert Python programmer solving coding problems through structured reasoning.

You MUST follow this exact structure for your response:

<planning>
Analyze the problem and plan your approach:
- What are the inputs and expected outputs?
- What algorithm or data structure will you use?
- What are the edge cases to consider?
- Break down the solution into steps
</planning>

<action>
Write your Python solution here. Enclose code in ```python blocks.
</action>

<reflection>
Review your solution:
- Does it handle all edge cases?
- Is it efficient enough?
- Are there any bugs or issues?
- What would you improve?
</reflection>

Always complete ALL THREE sections in order."""


PLANNING_PROMPT = """Analyze the following coding problem and create a detailed plan.

Problem:
{problem}

Respond with your analysis inside <planning> tags:
- Understand the inputs and outputs
- Identify the algorithm/approach to use
- List edge cases to handle
- Break down implementation steps

<planning>
"""

ACTION_PROMPT = """Based on your plan, now write the Python code to solve the problem.

Your plan was:
{plan}

Write your solution inside <action> tags with ```python code blocks:

<action>
"""

REFLECTION_PROMPT = """Your code has been executed with the following result:
{result}

Reflect on your solution inside <reflection> tags:
- Analyze whether the solution is correct
- Identify any issues or bugs
- Suggest improvements if needed
- Decide if another iteration is needed

<reflection>
"""

ITERATION_PROMPT = """Your previous attempt was incorrect. Here's the feedback:

Previous code:
```python
{code}
```

Execution result:
{result}

Please try again. Start with updated planning:

<planning>
"""


async_semaphore = asyncio.Semaphore(50)


class StructuredAgentConfig(BaseEnvConfig):
    """Configuration for structured agent traces."""

    dataset_name: str = Field(
        "NousResearch/RLVR_Coding_Problems",
        description="Dataset for coding problems",
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    max_iterations: int = Field(3, description="Max planning-action-reflection iterations")
    max_tokens_per_phase: int = Field(1024, description="Max tokens per phase")
    collect_logprobs: bool = Field(True, description="Collect logprobs for each phase")
    output_dir: str = Field("structured_traces", description="Output directory")


class StructuredAgentEnv(BaseEnv):
    """
    Environment for generating structured agent traces.

    Each trace follows the Planning-Action-Reflection pattern:
    1. PLANNING: Analyze problem, plan approach
    2. ACTION: Write code solution
    3. REFLECTION: Review results, iterate if needed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_id = 0
        self.cur_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.problem_queue = deque()
        self.metrics = {
            "total_traces": 0,
            "successful_traces": 0,
            "avg_iterations": [],
            "phase_lengths": {"planning": [], "action": [], "reflection": []},
        }

    @classmethod
    def config_init(cls) -> Tuple[StructuredAgentConfig, List[APIServerConfig]]:
        env_config = StructuredAgentConfig(
            tokenizer_name="Qwen/Qwen3-14B",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            temperature=0.7,
            max_iterations=3,
            wandb_name="structured_agent_deepseek",
        )

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

    async def setup(self):
        """Setup environment."""
        rprint("[bold green]Setting up Structured Agent Environment[/bold green]")

        self.train = load_dataset(self.config.dataset_name, split="train")
        rprint(f"Loaded {len(self.train)} training problems")

        for i in range(len(self.train)):
            self.problem_queue.append(i)

        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.config.output_dir,
            self.cur_time,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.lock = asyncio.Lock()

    async def get_next_item(self) -> Item:
        async with self.lock:
            if not self.problem_queue:
                for i in range(len(self.train)):
                    self.problem_queue.append(i)
            cur_idx = self.problem_queue.popleft()
            item = dict(self.train[cur_idx])
        item["idx"] = cur_idx
        item["split"] = "train"
        return item

    def extract_phase_content(self, text: str, phase: str) -> str:
        """Extract content from a specific phase tag."""
        pattern = rf"<{phase}>(.*?)</{phase}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from text."""
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    async def execute_code(self, code: str, tests: Dict) -> Tuple[bool, Dict]:
        """Execute code and return success status and metadata."""
        if not code:
            return False, {"error": "No code provided"}

        try:
            async with async_semaphore:
                test_input = {"tests": tests}
                if USE_MODAL and run_test is not None:
                    try:
                        res, metadata = await run_test.remote.aio(test_input, code)
                    except Exception as e:
                        res, metadata = await run_test_local(test_input, code)
                else:
                    res, metadata = await run_test_local(test_input, code)

            success = set(res) == {True}
            return success, metadata
        except Exception as e:
            return False, {"error": str(e)}

    async def generate_phase(
        self,
        messages: List[Dict],
        phase: AgentPhase,
        max_tokens: int = 1024
    ) -> Tuple[str, List[int], List[float]]:
        """Generate content for a specific phase."""
        try:
            completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
            )
            content = completion.choices[0].message.content

            # Tokenize for logprobs tracking
            assistant_msg = {"role": "assistant", "content": content}
            full_messages = messages + [assistant_msg]
            out_dict = tokenize_for_trainer(
                self.tokenizer,
                full_messages,
                completion.choices[0].finish_reason
            )

            return content, out_dict.get("tokens", []), []

        except Exception as e:
            rprint(f"[red]Generation error in {phase.value}: {e}[/red]")
            return "", [], []

    async def run_agent_loop(self, item: Item) -> AgentTrace:
        """
        Run the full Planning-Action-Reflection loop for a problem.
        """
        trace = AgentTrace(
            problem_idx=item["idx"],
            problem=item.get("problem", ""),
        )

        tests = item.get("tests", {})
        if isinstance(tests, str):
            tests = json.loads(tests)
        tests["fn_name"] = item.get("fn_name", "none")

        iteration = 0
        success = False
        last_code = None
        last_result = None

        while iteration < self.config.max_iterations and not success:
            iteration += 1
            trace.num_iterations = iteration

            # === PLANNING PHASE ===
            if iteration == 1:
                planning_prompt = PLANNING_PROMPT.format(problem=item.get("problem", ""))
            else:
                planning_prompt = ITERATION_PROMPT.format(
                    code=last_code or "",
                    result=json.dumps(last_result, indent=2) if last_result else "No result"
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ]

            planning_content, planning_tokens, planning_logprobs = await self.generate_phase(
                messages, AgentPhase.PLANNING, self.config.max_tokens_per_phase
            )

            plan = self.extract_phase_content(planning_content, "planning")
            if not plan:
                plan = planning_content  # Use full content if no tags

            trace.steps.append(AgentStep(
                phase=AgentPhase.PLANNING,
                content=plan,
                tokens=planning_tokens,
                logprobs=planning_logprobs,
                metadata={"iteration": iteration}
            ))

            # === ACTION PHASE ===
            action_prompt = ACTION_PROMPT.format(plan=plan)
            messages.append({"role": "assistant", "content": f"<planning>\n{plan}\n</planning>"})
            messages.append({"role": "user", "content": action_prompt})

            action_content, action_tokens, action_logprobs = await self.generate_phase(
                messages, AgentPhase.ACTION, self.config.max_tokens_per_phase
            )

            code = self.extract_code(action_content)
            if not code:
                # Try to extract from action tags
                action_text = self.extract_phase_content(action_content, "action")
                code = self.extract_code(action_text) if action_text else None

            trace.steps.append(AgentStep(
                phase=AgentPhase.ACTION,
                content=action_content,
                tokens=action_tokens,
                logprobs=action_logprobs,
                metadata={"iteration": iteration, "has_code": code is not None}
            ))

            # Execute code
            if code:
                last_code = code
                success, exec_result = await self.execute_code(code, tests)
                last_result = exec_result
                trace.execution_result = exec_result
                trace.final_code = code
            else:
                success = False
                last_result = {"error": "No code extracted"}
                exec_result = last_result

            # === REFLECTION PHASE ===
            result_str = "SUCCESS - All tests passed!" if success else f"FAILED - {json.dumps(exec_result)}"
            reflection_prompt = REFLECTION_PROMPT.format(result=result_str)

            messages.append({"role": "assistant", "content": f"<action>\n{action_content}\n</action>"})
            messages.append({"role": "user", "content": reflection_prompt})

            reflection_content, reflection_tokens, reflection_logprobs = await self.generate_phase(
                messages, AgentPhase.REFLECTION, self.config.max_tokens_per_phase // 2
            )

            reflection = self.extract_phase_content(reflection_content, "reflection")
            if not reflection:
                reflection = reflection_content

            trace.steps.append(AgentStep(
                phase=AgentPhase.REFLECTION,
                content=reflection,
                tokens=reflection_tokens,
                logprobs=reflection_logprobs,
                metadata={"iteration": iteration, "success": success}
            ))

            if success:
                break

        # Calculate total tokens
        trace.total_tokens = sum(len(step.tokens) for step in trace.steps)
        trace.score = 1.0 if success else -1.0

        return trace

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup | None, List[Item]]:
        """Collect structured agent trajectories."""
        rprint(f"[cyan]Collecting structured trace for problem {item['idx']}[/cyan]")

        start_time = time.time()

        # Run multiple agent loops in parallel
        tasks = [self.run_agent_loop(item) for _ in range(self.config.group_size)]
        traces = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Build ScoredDataGroup
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["overrides"] = []

        for trace in traces:
            # Concatenate all tokens from all steps
            all_tokens = []
            for step in trace.steps:
                all_tokens.extend(step.tokens)

            scored_data["tokens"].append(all_tokens)
            scored_data["masks"].append(all_tokens)  # Simplified
            scored_data["scores"].append(trace.score)
            scored_data["overrides"].append({"set_advantage_to_zero": False})

        # Log metrics
        num_success = sum(1 for t in traces if t.score > 0)
        avg_iterations = np.mean([t.num_iterations for t in traces])

        rprint(f"Problem {item['idx']}: {num_success}/{len(traces)} success, "
               f"avg_iter={avg_iterations:.1f}, time={elapsed:.1f}s")

        # Save traces
        await self.save_traces(item, traces)

        # Update metrics
        async with self.lock:
            self.metrics["total_traces"] += len(traces)
            self.metrics["successful_traces"] += num_success
            self.metrics["avg_iterations"].append(avg_iterations)

        return scored_data, []

    async def save_traces(self, item: Item, traces: List[AgentTrace]):
        """Save structured traces to file."""
        output = {
            "problem_idx": item["idx"],
            "problem": item.get("problem", ""),
            "timestamp": datetime.now().isoformat(),
            "traces": [trace.to_dict() for trace in traces],
            "summary": {
                "total_traces": len(traces),
                "successful": sum(1 for t in traces if t.score > 0),
                "avg_iterations": np.mean([t.num_iterations for t in traces]),
                "avg_tokens": np.mean([t.total_tokens for t in traces]),
            }
        }

        trace_file = os.path.join(
            self.output_dir,
            f"structured_trace_{item['idx']:06d}.json"
        )
        with open(trace_file, "w") as f:
            json.dump(output, f, indent=2)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        async with self.lock:
            if self.metrics["total_traces"] > 0:
                wandb_metrics["train/success_rate"] = (
                    self.metrics["successful_traces"] / self.metrics["total_traces"]
                )
            if self.metrics["avg_iterations"]:
                wandb_metrics["train/avg_iterations"] = np.mean(
                    self.metrics["avg_iterations"][-100:]
                )

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    StructuredAgentEnv.cli()
