#!/usr/bin/env python3
"""
Standalone script to run the structured Planning-Action-Reflection pipeline.

This script demonstrates the full agent trace generation with:
1. PLANNING - Problem analysis and approach planning
2. ACTION - Code generation
3. REFLECTION - Result analysis and iteration

Usage:
    export OLLAMA_API_KEY=your_key
    python run_structured_pipeline.py

    # With custom problem
    python run_structured_pipeline.py --problem "Write a function to check if a number is prime"

    # Save traces to file
    python run_structured_pipeline.py --output traces.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


console = Console()


# ============================================================================
# Prompts for structured reasoning
# ============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer. You solve problems through structured reasoning.

For EVERY response, you MUST use this EXACT format with XML tags:

<planning>
1. **Problem Analysis**: What are we solving?
2. **Inputs/Outputs**: What format?
3. **Approach**: Which algorithm/technique?
4. **Edge Cases**: What could go wrong?
5. **Steps**: Break down the implementation
</planning>

<action>
```python
# Your complete Python code here
```
</action>

<reflection>
1. **Correctness**: Does the solution work?
2. **Edge Cases**: Are they handled?
3. **Efficiency**: Time/space complexity?
4. **Improvements**: What could be better?
</reflection>

IMPORTANT: Always include ALL THREE sections with proper XML tags."""


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    phase: str  # planning, action, reflection
    content: str
    tokens_count: int = 0
    duration_ms: float = 0


@dataclass
class AgentTrace:
    """Complete structured agent trace."""
    problem: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_code: Optional[str] = None
    execution_result: Optional[Dict] = None
    success: bool = False
    iterations: int = 0
    total_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "problem": self.problem,
            "steps": [
                {
                    "phase": s.phase,
                    "content": s.content,
                    "tokens_count": s.tokens_count,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
            "final_code": self.final_code,
            "execution_result": self.execution_result,
            "success": self.success,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp,
        }


class StructuredAgent:
    """Agent that follows Planning-Action-Reflection structure."""

    def __init__(
        self,
        base_url: str = "https://ollama.com",
        api_key: str = "",
        model: str = "deepseek-v3.2",
        max_iterations: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations

    def _get_api_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3] + "/api/chat"
        return self.base_url + "/api/chat"

    async def _call_llm(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Call Ollama API."""
        import aiohttp

        url = self._get_api_url()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=300) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    def _extract_section(self, text: str, tag: str) -> str:
        """Extract content between XML tags."""
        import re
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from markdown blocks."""
        import re
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None

    async def _execute_code(self, code: str, test_cases: Optional[Dict] = None) -> Dict:
        """Execute code locally."""
        from local_executor import execute_code_safe

        if not code:
            return {"success": False, "error": "No code"}

        if test_cases:
            results, metadata = execute_code_safe(code, test_cases, timeout=10.0)
            return {
                "success": all(results),
                "results": results,
                "metadata": metadata,
            }
        else:
            # Just try to run the code
            try:
                exec_globals = {}
                exec(code, exec_globals)
                return {"success": True, "executed": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def solve(
        self,
        problem: str,
        test_cases: Optional[Dict] = None,
        verbose: bool = True,
    ) -> AgentTrace:
        """
        Solve a problem using Planning-Action-Reflection structure.
        """
        trace = AgentTrace(problem=problem)

        iteration = 0
        success = False
        last_code = None
        last_result = None

        while iteration < self.max_iterations and not success:
            iteration += 1
            trace.iterations = iteration

            if verbose:
                console.print(f"\n[bold blue]━━━ Iteration {iteration} ━━━[/bold blue]")

            # Build prompt
            if iteration == 1:
                user_content = f"""Solve this problem:

{problem}

Remember to use the structured format with <planning>, <action>, and <reflection> tags."""
            else:
                user_content = f"""Your previous solution was incorrect.

Previous code:
```python
{last_code}
```

Result: {json.dumps(last_result, indent=2)}

Please try again with the structured format. Analyze what went wrong in <planning>."""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Get LLM response
            import time
            start_time = time.time()
            response = await self._call_llm(messages)
            duration = (time.time() - start_time) * 1000

            # Extract phases
            planning = self._extract_section(response, "planning")
            action = self._extract_section(response, "action")
            reflection = self._extract_section(response, "reflection")

            # Store planning step
            if planning:
                trace.steps.append(ReasoningStep(
                    phase="planning",
                    content=planning,
                    duration_ms=duration / 3,
                ))
                if verbose:
                    console.print(Panel(
                        Markdown(planning),
                        title="[green]PLANNING[/green]",
                        border_style="green",
                    ))

            # Store action step and extract code
            code = self._extract_code(action) or self._extract_code(response)
            if action or code:
                trace.steps.append(ReasoningStep(
                    phase="action",
                    content=action or f"```python\n{code}\n```",
                    duration_ms=duration / 3,
                ))
                if verbose and code:
                    console.print(Panel(
                        f"```python\n{code}\n```",
                        title="[yellow]ACTION (Code)[/yellow]",
                        border_style="yellow",
                    ))

            # Execute code
            if code:
                last_code = code
                trace.final_code = code
                exec_result = await self._execute_code(code, test_cases)
                last_result = exec_result
                trace.execution_result = exec_result
                success = exec_result.get("success", False)

                if verbose:
                    status = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
                    console.print(f"\nExecution: {status}")
                    if not success and "error" in exec_result:
                        console.print(f"Error: {exec_result['error']}")
            else:
                last_result = {"error": "No code extracted"}
                success = False

            # Store reflection step
            if reflection:
                trace.steps.append(ReasoningStep(
                    phase="reflection",
                    content=reflection,
                    duration_ms=duration / 3,
                ))
                if verbose:
                    console.print(Panel(
                        Markdown(reflection),
                        title="[cyan]REFLECTION[/cyan]",
                        border_style="cyan",
                    ))

            if success:
                break

        trace.success = success
        trace.total_tokens = sum(len(s.content.split()) * 2 for s in trace.steps)  # Approximate

        return trace


# ============================================================================
# Example problems
# ============================================================================

EXAMPLE_PROBLEMS = [
    {
        "problem": """Write a function `two_sum(nums, target)` that takes a list of integers and a target sum.
Return the indices of two numbers that add up to the target.

Example:
- two_sum([2, 7, 11, 15], 9) -> [0, 1]
- two_sum([3, 2, 4], 6) -> [1, 2]""",
        "tests": {
            "fn_name": "two_sum",
            "inputs": [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6]],
            "outputs": [[0, 1], [1, 2], [0, 1]],
        },
    },
    {
        "problem": """Write a function `is_palindrome(s)` that checks if a string is a palindrome.
Only consider alphanumeric characters and ignore case.

Example:
- is_palindrome("A man, a plan, a canal: Panama") -> True
- is_palindrome("race a car") -> False""",
        "tests": {
            "fn_name": "is_palindrome",
            "inputs": [["A man, a plan, a canal: Panama"], ["race a car"], [""]],
            "outputs": [True, False, True],
        },
    },
    {
        "problem": """Write a function `max_subarray(nums)` that finds the contiguous subarray with the largest sum.
Return the maximum sum.

Example:
- max_subarray([-2,1,-3,4,-1,2,1,-5,4]) -> 6 (subarray [4,-1,2,1])
- max_subarray([1]) -> 1""",
        "tests": {
            "fn_name": "max_subarray",
            "inputs": [[[-2, 1, -3, 4, -1, 2, 1, -5, 4]], [[1]], [[-1]]],
            "outputs": [6, 1, -1],
        },
    },
]


async def main():
    parser = argparse.ArgumentParser(description="Run structured agent pipeline")
    parser.add_argument("--problem", type=str, help="Custom problem to solve")
    parser.add_argument("--output", type=str, help="Output file for traces (JSONL)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--example", type=int, help="Run example problem (0-2)")
    args = parser.parse_args()

    console.print("[bold magenta]" + "=" * 60)
    console.print("[bold magenta]Structured Agent Pipeline (Planning-Action-Reflection)")
    console.print("[bold magenta]" + "=" * 60)

    # Configuration
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    model = os.getenv("OLLAMA_MODEL", "deepseek-v3.2")

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Base URL: {base_url}")
    console.print(f"  Model: {model}")
    console.print(f"  API Key: {'[set]' if api_key else '[NOT SET]'}")

    if not api_key and "ollama.com" in base_url:
        console.print("\n[red]Warning: OLLAMA_API_KEY not set for Ollama Cloud[/red]")

    agent = StructuredAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_iterations=args.max_iterations,
    )

    traces = []

    if args.problem:
        # Custom problem
        console.print(f"\n[bold]Solving custom problem...[/bold]")
        trace = await agent.solve(args.problem)
        traces.append(trace)
    elif args.example is not None:
        # Single example
        example = EXAMPLE_PROBLEMS[args.example % len(EXAMPLE_PROBLEMS)]
        console.print(f"\n[bold]Solving example {args.example}...[/bold]")
        trace = await agent.solve(example["problem"], example.get("tests"))
        traces.append(trace)
    else:
        # All examples
        console.print(f"\n[bold]Running all {len(EXAMPLE_PROBLEMS)} example problems...[/bold]")
        for i, example in enumerate(EXAMPLE_PROBLEMS):
            console.print(f"\n[bold yellow]━━━ Problem {i + 1}/{len(EXAMPLE_PROBLEMS)} ━━━[/bold yellow]")
            console.print(f"[dim]{example['problem'][:100]}...[/dim]")
            trace = await agent.solve(example["problem"], example.get("tests"))
            traces.append(trace)

    # Summary
    console.print("\n[bold magenta]" + "=" * 60)
    console.print("[bold magenta]Summary")
    console.print("[bold magenta]" + "=" * 60)

    for i, trace in enumerate(traces):
        status = "[green]SUCCESS[/green]" if trace.success else "[red]FAILED[/red]"
        console.print(f"\nTrace {i + 1}: {status}")
        console.print(f"  Iterations: {trace.iterations}")
        console.print(f"  Steps: {len(trace.steps)}")
        console.print(f"  Phases: {[s.phase for s in trace.steps]}")

    # Save traces
    if args.output:
        with open(args.output, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        console.print(f"\n[green]Traces saved to {args.output}[/green]")

    success_rate = sum(1 for t in traces if t.success) / len(traces) if traces else 0
    console.print(f"\n[bold]Overall Success Rate: {success_rate:.0%}[/bold]")

    return all(t.success for t in traces)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
