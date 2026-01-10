#!/usr/bin/env python3
"""
Interleaved Reasoning Agent - True step-by-step thinking.

Unlike the structured Planning-Action-Reflection pipeline, this agent
interleaves reasoning with each coding step:

    Think → Code snippet → Think → Code snippet → Verify → Think → ...

This catches bugs during generation, not after.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()


# ============================================================================
# Interleaved Reasoning Prompt
# ============================================================================

INTERLEAVED_SYSTEM_PROMPT = """You are an expert Python programmer who thinks step-by-step while coding.

You MUST interleave your reasoning with code. For EVERY line or block you write, FIRST explain your thinking.

Use this format with [THINK] and [CODE] markers:

[THINK] First, I need to understand what we're solving...
[THINK] The function signature should take nums (list) and target (int)
[CODE]
def two_sum(nums, target):
[/CODE]

[THINK] I need a hash map to store seen numbers and their indices for O(n) lookup
[CODE]
    seen = {}
[/CODE]

[THINK] Now iterate through the array. For each number, I calculate what complement we need.
[CODE]
    for i, num in enumerate(nums):
        complement = target - num
[/CODE]

[THINK] Check if complement exists in seen. If yes, we found our pair!
[THINK] WAIT - seen[complement] gives the INDEX, not the value. Let me verify:
       - seen = {2: 0} means value 2 is at index 0
       - So return [seen[complement], i] gives [index_of_complement, current_index]
       - That's correct!
[CODE]
        if complement in seen:
            return [seen[complement], i]
[/CODE]

[THINK] If not found, add current number to seen. Key = number, Value = index
[THINK] Let me verify: seen[num] = i means "number 'num' is at index 'i'"
       - Later lookup seen[complement] will give the index - correct!
[CODE]
        seen[num] = i
[/CODE]

[THINK] Edge case: what if no solution exists? Return empty list.
[CODE]
    return []
[/CODE]

[VERIFY]
Let me trace through with nums=[2,7,11,15], target=9:
- i=0, num=2, complement=7, seen={} → 7 not in seen → seen={2:0}
- i=1, num=7, complement=2, seen={2:0} → 2 in seen! → return [0,1] ✓
[/VERIFY]

IMPORTANT RULES:
1. EVERY code block MUST be preceded by [THINK]
2. Use [THINK] WAIT when you catch a potential bug
3. Use [VERIFY] to trace through your solution with an example
4. Show your working - don't just state conclusions"""


@dataclass
class ThinkingStep:
    """A single step in the interleaved trace."""
    step_type: str  # "think", "code", "verify", "wait"
    content: str
    line_number: int = 0


@dataclass
class InterleavedTrace:
    """Complete interleaved reasoning trace."""
    problem: str
    steps: List[ThinkingStep] = field(default_factory=list)
    final_code: Optional[str] = None
    execution_result: Optional[Dict] = None
    success: bool = False
    iterations: int = 0
    think_count: int = 0
    wait_count: int = 0  # Bug catches during reasoning
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "problem": self.problem,
            "steps": [
                {"type": s.step_type, "content": s.content, "line": s.line_number}
                for s in self.steps
            ],
            "final_code": self.final_code,
            "execution_result": self.execution_result,
            "success": self.success,
            "iterations": self.iterations,
            "think_count": self.think_count,
            "wait_count": self.wait_count,
            "timestamp": self.timestamp,
        }


class InterleavedAgent:
    """Agent with true interleaved reasoning."""

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

    async def _call_llm(self, messages: List[Dict], max_tokens: int = 4096) -> str:
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
                "temperature": 0.3,  # Lower temp for more consistent reasoning
                "num_predict": max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=300
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    def _parse_interleaved_response(self, text: str) -> Tuple[List[ThinkingStep], str]:
        """Parse interleaved [THINK]/[CODE]/[VERIFY] blocks."""
        steps = []
        code_parts = []
        line_num = 0

        # Also try to extract from markdown code blocks as fallback
        # Pattern for [CODE]...[/CODE] blocks
        code_block_pattern = r'\[CODE\](.*?)\[/CODE\]'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

        # Split by markers
        pattern = r'\[(THINK|CODE|VERIFY|WAIT)\](.*?)(?=\[(?:THINK|CODE|VERIFY|WAIT|/CODE)\]|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for marker, content in matches:
            marker = marker.upper()

            # Handle /CODE closing tag - keep original whitespace for code
            if marker == "CODE":
                # Find the matching code block to preserve indentation
                content = re.sub(r'\[/CODE\].*$', '', content, flags=re.DOTALL)
                # Remove only leading/trailing blank lines, not indentation
                lines = content.split('\n')
                # Strip leading empty lines
                while lines and not lines[0].strip():
                    lines.pop(0)
                # Strip trailing empty lines
                while lines and not lines[-1].strip():
                    lines.pop()
                content = '\n'.join(lines)
            else:
                content = content.strip()

            if not content:
                continue

            if marker == "CODE":
                code_lines = content.split('\n')
                line_num += len(code_lines)
                code_parts.append(content)
                steps.append(ThinkingStep(
                    step_type="code",
                    content=content,
                    line_number=line_num,
                ))
            elif marker == "THINK":
                # Check if it's actually a WAIT
                if content.upper().startswith("WAIT"):
                    steps.append(ThinkingStep(
                        step_type="wait",
                        content=content,
                    ))
                else:
                    steps.append(ThinkingStep(
                        step_type="think",
                        content=content,
                    ))
            elif marker == "WAIT":
                steps.append(ThinkingStep(
                    step_type="wait",
                    content=content,
                ))
            elif marker == "VERIFY":
                steps.append(ThinkingStep(
                    step_type="verify",
                    content=content,
                ))

        # Reconstruct full code - maintain structure
        full_code = '\n'.join(code_parts)

        # Fallback: if no code found, try markdown blocks
        if not full_code.strip():
            md_pattern = r'```python\s*(.*?)```'
            md_matches = re.findall(md_pattern, text, re.DOTALL)
            if md_matches:
                full_code = md_matches[-1].strip()

        return steps, full_code

    async def _execute_code(
        self, code: str, test_cases: Optional[Dict] = None, verbose: bool = True
    ) -> Dict:
        """Execute code with tests."""
        from local_executor import execute_code_safe

        if not code or not code.strip():
            return {"success": False, "error": "No code provided"}

        if test_cases:
            results, metadata = execute_code_safe(code, test_cases, timeout=10.0)

            if verbose and "results" in metadata:
                actual = metadata.get("results", [])
                expected = metadata.get("expected", [])
                console.print("\n[bold]Test Execution:[/bold]")
                for i, (res, exp, passed) in enumerate(
                    zip(actual, expected, results)
                ):
                    status = "[green]✓[/green]" if passed else "[red]✗[/red]"
                    console.print(f"  {status} Test {i+1}: ", end="")
                    if passed:
                        console.print(f"{exp}")
                    else:
                        console.print(f"expected {exp}, got {res}")

            return {
                "success": all(results),
                "results": results,
                "passed": sum(results),
                "total": len(results),
                "metadata": metadata,
            }
        else:
            try:
                exec_globals = {}
                exec(code, exec_globals)
                return {"success": True, "executed": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _display_step(self, step: ThinkingStep):
        """Display a reasoning step with appropriate formatting."""
        if step.step_type == "think":
            console.print(f"[dim cyan]💭 {step.content}[/dim cyan]")
        elif step.step_type == "wait":
            console.print(f"[bold yellow]⚠️  WAIT: {step.content}[/bold yellow]")
        elif step.step_type == "verify":
            console.print(Panel(
                step.content,
                title="[magenta]🔍 VERIFY[/magenta]",
                border_style="magenta",
            ))
        elif step.step_type == "code":
            syntax = Syntax(step.content, "python", theme="monokai", line_numbers=True)
            console.print(syntax)

    async def solve(
        self,
        problem: str,
        test_cases: Optional[Dict] = None,
        verbose: bool = True,
    ) -> InterleavedTrace:
        """Solve with interleaved reasoning."""
        trace = InterleavedTrace(problem=problem)

        iteration = 0
        success = False
        last_code = None
        last_result = None

        while iteration < self.max_iterations and not success:
            iteration += 1
            trace.iterations = iteration

            if verbose:
                console.print(f"\n[bold blue]{'═' * 50}[/bold blue]")
                console.print(f"[bold blue]  Iteration {iteration}[/bold blue]")
                console.print(f"[bold blue]{'═' * 50}[/bold blue]\n")

            # Build prompt
            if iteration == 1:
                user_content = f"""Solve this problem step-by-step:

{problem}

Remember: Use [THINK] before EVERY code block. Use [VERIFY] to trace through your solution."""
            else:
                user_content = f"""Your previous solution failed.

Previous code:
```python
{last_code}
```

Error: {json.dumps(last_result, indent=2)}

Try again with more careful step-by-step reasoning. Use [THINK] WAIT when you spot potential issues."""

            messages = [
                {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Get response
            import time
            start = time.time()
            response = await self._call_llm(messages)
            duration = (time.time() - start) * 1000

            # Parse interleaved steps
            steps, code = self._parse_interleaved_response(response)

            # Display and count
            for step in steps:
                trace.steps.append(step)
                if verbose:
                    self._display_step(step)

                if step.step_type == "think":
                    trace.think_count += 1
                elif step.step_type == "wait":
                    trace.wait_count += 1

            # Execute
            if code:
                last_code = code
                trace.final_code = code
                exec_result = await self._execute_code(code, test_cases, verbose)
                last_result = exec_result
                trace.execution_result = exec_result
                success = exec_result.get("success", False)

                if verbose:
                    passed = exec_result.get("passed", 0)
                    total = exec_result.get("total", 0)
                    if success:
                        console.print(f"\n[bold green]✓ All {total} tests passed![/bold green]")
                    else:
                        console.print(f"\n[bold red]✗ Failed: {passed}/{total} tests[/bold red]")
            else:
                last_result = {"error": "No code extracted"}
                if verbose:
                    console.print("[red]No code found in response[/red]")

        trace.success = success
        return trace


# ============================================================================
# Example problems with adversarial tests
# ============================================================================

PROBLEMS = [
    {
        "problem": """Write a function `two_sum(nums, target)` that returns indices of two numbers that add up to target.

Example:
- two_sum([2, 7, 11, 15], 9) -> [0, 1]
- two_sum([3, 2, 4], 6) -> [1, 2]""",
        "tests": {
            "fn_name": "two_sum",
            "inputs": [
                [[2, 7, 11, 15], 9],
                [[3, 2, 4], 6],
                [[3, 3], 6],
                [[-1, -2, -3, -4, -5], -8],
            ],
            "outputs": [[0, 1], [1, 2], [0, 1], [2, 4]],
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
            "inputs": [
                ["A man, a plan, a canal: Panama"],
                ["race a car"],
                [""],
                [".,"],
                ["0P"],
            ],
            "outputs": [True, False, True, True, False],
        },
    },
]


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interleaved Reasoning Agent")
    parser.add_argument("--example", type=int, default=0, help="Problem index (0-1)")
    parser.add_argument("--output", type=str, help="Save trace to file")
    args = parser.parse_args()

    console.print("[bold magenta]" + "=" * 60)
    console.print("[bold magenta]  Interleaved Reasoning Agent")
    console.print("[bold magenta]  Think → Code → Think → Code → Verify")
    console.print("[bold magenta]" + "=" * 60)

    # Config
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    model = os.getenv("OLLAMA_MODEL", "deepseek-v3.2")

    console.print(f"\n[dim]Model: {model}[/dim]")
    console.print(f"[dim]API Key: {'set' if api_key else 'NOT SET'}[/dim]")

    agent = InterleavedAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    problem = PROBLEMS[args.example % len(PROBLEMS)]
    console.print(f"\n[bold]Problem:[/bold]\n{problem['problem']}\n")

    trace = await agent.solve(problem["problem"], problem.get("tests"))

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Trace Summary:[/bold]")
    console.print(f"  Success: {'✓' if trace.success else '✗'}")
    console.print(f"  Iterations: {trace.iterations}")
    console.print(f"  Think steps: {trace.think_count}")
    console.print(f"  WAIT (bug catches): {trace.wait_count}")
    console.print(f"  Total steps: {len(trace.steps)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        console.print(f"\n[green]Saved to {args.output}[/green]")

    return trace.success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
