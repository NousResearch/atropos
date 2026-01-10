#!/usr/bin/env python3
"""
Interleaved Reasoning + Tool Use Trace Generator

Combines BOTH dimensions:
1. TRUE INTERLEAVING: Think→Code→Think→Code (1-3 lines per code block)
2. TOOL USE: Real code execution with [RESULT]/[ERROR] feedback

This produces the IDEAL training traces:

    [THINK] I need to initialize pointers for binary search
    [CODE]
    def binary_search(nums, target):
        left, right = 0, len(nums) - 1
    [/CODE]

    [THINK] Now I need a loop while the interval is valid
    [CODE]
        while left <= right:
            mid = (left + right) // 2
    [/CODE]

    [THINK] Compare mid element with target, three cases
    [CODE]
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
    [/CODE]

    [THINK] If we exit the loop, target wasn't found
    [CODE]
        return -1
    [/CODE]

    << SYSTEM executes accumulated code >>

    [RESULT]
    Test 1: PASS - binary_search([1,2,3,4,5], 3) = 2
    Test 2: PASS - binary_search([1,2,3,4,5], 6) = -1
    All 4 tests passed!
    [/RESULT]

    [VERIFY] The binary search divides the search space in half each iteration...

Usage:
    python trace_generator_interleaved_tools.py --output traces.jsonl --num-traces 10
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Local imports
from local_executor import execute_code_safe

# Optional rich for nice output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


# ============================================================================
# System Prompts - STRICT one-step-at-a-time with tool awareness
# ============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer who thinks step-by-step while coding.

## CRITICAL: ONE STEP AT A TIME

You MUST output EXACTLY ONE of these per response:

1. [THINK] your reasoning (1-2 sentences)
2. [CODE]
   1-3 lines of code ONLY
   [/CODE]
3. [VERIFY] trace through solution (only after seeing [RESULT] with all tests passed)

## RULES

- Each [CODE] block: MAXIMUM 3 lines of code
- After [CODE], STOP IMMEDIATELY - wait for next prompt
- Do NOT output [RESULT] or [ERROR] - those come from SYSTEM only
- Do NOT write the entire function at once
- Build the function incrementally, piece by piece

## WORKFLOW

1. [THINK] about what to write next (1-2 sentences)
2. [CODE] write 1-3 lines [/CODE]
3. << STOP and wait >>
4. Repeat steps 1-3 until function is complete
5. << SYSTEM executes your accumulated code and shows [RESULT] >>
6. If errors: [THINK] about the bug, then [CODE] fix
7. If success: [VERIFY] trace through the solution

## EXAMPLE (your output only)

[THINK] I'll start by defining the function and initializing the hash map.
[CODE]
def two_sum(nums, target):
    seen = {}
[/CODE]

<< STOP HERE - wait for prompt to continue >>

Next turn:
[THINK] Now I need to iterate through nums and check for complement.
[CODE]
    for i, num in enumerate(nums):
        complement = target - num
[/CODE]

<< STOP HERE >>

Next turn:
[THINK] Check if complement exists, if so return indices.
[CODE]
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
[/CODE]

<< STOP HERE >>

Next turn:
[THINK] Need to handle case when no solution found.
[CODE]
    return []
[/CODE]

<< SYSTEM will now execute and show [RESULT] >>"""


INITIAL_PROMPT = """Solve this problem by writing code step-by-step.

Problem:
{problem}

IMPORTANT:
- Output ONE [THINK] followed by ONE [CODE] block (max 3 lines)
- STOP after [/CODE] and wait for the next prompt
- Do NOT write the entire solution at once

Start now with your first [THINK] and [CODE]:"""


CONTINUE_PROMPT = """Continue building the function.

Remember:
- ONE [THINK] + ONE [CODE] (max 3 lines)
- STOP after [/CODE]
- Do NOT repeat previous code"""


FUNCTION_COMPLETE_PROMPT = """Your function looks complete. I will now execute it against the test cases.

<< Executing... >>

{result}

{next_instruction}"""


AFTER_SUCCESS_PROMPT = """All tests passed! Now use [VERIFY] to trace through your solution step-by-step.

Show how the algorithm works with a concrete example."""


AFTER_FAILURE_PROMPT = """Some tests failed. Use [THINK] to analyze the bug, then [CODE] to fix it.

Remember: Only output the FIX (the lines that need to change), not the entire function."""


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TraceStep:
    """Single step in reasoning trace."""
    step_type: str  # think, code, result, error, verify
    content: str


@dataclass
class GeneratedTrace:
    """Complete generated trace with interleaving + tool use."""
    problem: str
    messages: List[Dict[str, str]]
    code: Optional[str] = None
    score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    think_count: int = 0
    code_block_count: int = 0  # Key metric for interleaving
    execution_count: int = 0   # How many times code was executed
    has_verify: bool = False
    had_errors: bool = False
    trace: List[TraceStep] = field(default_factory=list)
    error: Optional[str] = None
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "problem": self.problem,
            "messages": self.messages,
            "code": self.code,
            "score": self.score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "think_count": self.think_count,
            "code_block_count": self.code_block_count,
            "execution_count": self.execution_count,
            "has_verify": self.has_verify,
            "had_errors": self.had_errors,
            "trace": [{"type": s.step_type, "content": s.content} for s in self.trace],
            "error": self.error,
            "model": self.model,
            "timestamp": self.timestamp,
            # Quality metrics
            "is_interleaved": self.code_block_count >= 3,
            "has_tool_feedback": self.execution_count > 0,
            "is_ideal": self.code_block_count >= 3 and self.execution_count > 0 and self.score > 0,
        }

    def to_training_format(self) -> Dict:
        """Convert to training format with full trace as assistant message."""
        combined = self._combine_trace()
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INITIAL_PROMPT.format(problem=self.problem)},
                {"role": "assistant", "content": combined},
            ],
            "score": self.score,
            "think_count": self.think_count,
            "code_block_count": self.code_block_count,
            "is_ideal": self.code_block_count >= 3 and self.execution_count > 0 and self.score > 0,
        }

    def _combine_trace(self) -> str:
        parts = []
        for step in self.trace:
            if step.step_type == "think":
                parts.append(f"[THINK] {step.content}")
            elif step.step_type == "code":
                parts.append(f"[CODE]\n{step.content}\n[/CODE]")
            elif step.step_type == "result":
                parts.append(f"[RESULT]\n{step.content}\n[/RESULT]")
            elif step.step_type == "error":
                parts.append(f"[ERROR]\n{step.content}\n[/ERROR]")
            elif step.step_type == "verify":
                parts.append(f"[VERIFY]\n{step.content}\n[/VERIFY]")
        return "\n\n".join(parts)


# ============================================================================
# Built-in Problems
# ============================================================================

BUILTIN_PROBLEMS = [
    {
        "id": "two_sum",
        "prompt": """Write a function `two_sum(nums, target)` that takes a list of integers and a target sum.
Return the indices of two numbers that add up to the target.

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
        "id": "is_palindrome",
        "prompt": """Write a function `is_palindrome(s)` that checks if a string is a palindrome.
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
    {
        "id": "max_subarray",
        "prompt": """Write a function `max_subarray(nums)` that finds the contiguous subarray with the largest sum.
Return the maximum sum (Kadane's algorithm).

Example:
- max_subarray([-2,1,-3,4,-1,2,1,-5,4]) -> 6
- max_subarray([1]) -> 1""",
        "tests": {
            "fn_name": "max_subarray",
            "inputs": [
                [[-2, 1, -3, 4, -1, 2, 1, -5, 4]],
                [[1]],
                [[-1]],
                [[-2, -1]],
            ],
            "outputs": [6, 1, -1, -1],
        },
    },
    {
        "id": "binary_search",
        "prompt": """Write a function `binary_search(nums, target)` that returns the index of target in sorted array.
Return -1 if not found.

Example:
- binary_search([1,2,3,4,5], 3) -> 2
- binary_search([1,2,3,4,5], 6) -> -1""",
        "tests": {
            "fn_name": "binary_search",
            "inputs": [
                [[1, 2, 3, 4, 5], 3],
                [[1, 2, 3, 4, 5], 6],
                [[1], 1],
                [[], 5],
            ],
            "outputs": [2, -1, 0, -1],
        },
    },
    {
        "id": "valid_parentheses",
        "prompt": """Write a function `is_valid(s)` that checks if parentheses are valid.
A string is valid if every open bracket has a matching close bracket in correct order.

Example:
- is_valid("()") -> True
- is_valid("()[]{}") -> True
- is_valid("(]") -> False""",
        "tests": {
            "fn_name": "is_valid",
            "inputs": [["()"], ["()[]{}"], ["(]"], ["([)]"], ["{[]}"]],
            "outputs": [True, True, False, False, True],
        },
    },
    {
        "id": "climbing_stairs",
        "prompt": """Write a function `climb_stairs(n)` that returns the number of ways to climb n stairs.
Each time you can climb 1 or 2 steps.

Example:
- climb_stairs(2) -> 2 (1+1 or 2)
- climb_stairs(3) -> 3 (1+1+1, 1+2, 2+1)""",
        "tests": {
            "fn_name": "climb_stairs",
            "inputs": [[1], [2], [3], [4], [5]],
            "outputs": [1, 2, 3, 5, 8],
        },
    },
]


# ============================================================================
# Interleaved + Tool Trace Generator
# ============================================================================


class InterleavedToolTraceGenerator:
    """Generates traces with BOTH interleaving AND tool execution."""

    def __init__(
        self,
        base_url: str = "https://ollama.com",
        api_key: str = "",
        model: str = "deepseek-v3.2",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_steps: int = 25,
        max_fix_attempts: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_fix_attempts = max_fix_attempts

    def _get_api_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3] + "/api/chat"
        return self.base_url + "/api/chat"

    async def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM with strict stop sequences."""
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
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                # CRITICAL: Stop after [/CODE] to force interleaving
                # Also prevent hallucinated results
                "stop": [
                    "[/CODE]\n\n[THINK]",  # Don't continue after code
                    "[/CODE]\n\n[CODE]",   # Don't chain code blocks
                    "[RESULT]",             # Don't hallucinate results
                    "[ERROR]",              # Don't hallucinate errors
                    "\n\n\n",               # Stop on excessive newlines
                ],
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=300
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response = data.get("message", {}).get("content", "")

                # Strip any hallucinated results
                response = self._strip_hallucinations(response)
                return response

    def _strip_hallucinations(self, text: str) -> str:
        """Remove hallucinated [RESULT] or [ERROR] blocks."""
        text = re.sub(r'\[RESULT\].*?(?:\[/RESULT\]|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[ERROR\].*?(?:\[/ERROR\]|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def _parse_step(self, text: str) -> Tuple[Optional[str], str]:
        """Parse a single step. Returns (step_type, content)."""
        text = text.strip()

        # Check for [VERIFY]
        verify_match = re.search(r'\[VERIFY\](.*?)(?:\[/VERIFY\]|$)', text, re.DOTALL | re.IGNORECASE)
        if verify_match:
            return "verify", verify_match.group(1).strip()

        # Check for [CODE]...[/CODE]
        code_match = re.search(r'\[CODE\](.*?)\[/CODE\]', text, re.DOTALL | re.IGNORECASE)
        if code_match:
            content = code_match.group(1)
            lines = [l for l in content.split('\n') if l.strip() or l == '']
            # Trim empty lines at start/end
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            return "code", '\n'.join(lines)

        # Check for [THINK]
        think_match = re.search(r'\[THINK\]\s*(.*?)(?=\[|$)', text, re.DOTALL | re.IGNORECASE)
        if think_match:
            return "think", think_match.group(1).strip()

        return None, ""

    def _looks_complete(self, code: str) -> bool:
        """Check if accumulated code looks like a complete function."""
        if not code.strip():
            return False

        # Must have function definition
        if not re.search(r'def\s+\w+\s*\(', code):
            return False

        # Must have return statement
        if not re.search(r'\breturn\b', code):
            return False

        # Count indentation levels - if we're back to base level after return, likely complete
        lines = code.strip().split('\n')
        if len(lines) < 2:
            return False

        # Find the last non-empty line
        last_content_line = None
        for line in reversed(lines):
            if line.strip():
                last_content_line = line
                break

        if not last_content_line:
            return False

        # If last line has return and is indented (part of function), likely complete
        if 'return' in last_content_line:
            return True

        return False

    def _execute_code(self, code: str, tests: Dict) -> Tuple[str, bool, int, int]:
        """Execute accumulated code and return formatted results."""
        if not code.strip():
            return "Error: No code to execute", False, 0, 0

        try:
            results, metadata = execute_code_safe(code, tests, timeout=10.0)

            if "error" in metadata:
                return f"Error: {metadata['error']}", False, 0, len(tests.get("outputs", []))

            fn_name = tests.get("fn_name", "func")
            inputs = tests.get("inputs", [])
            outputs = tests.get("outputs", [])

            lines = []
            passed = 0
            for i, (inp, expected, result) in enumerate(zip(inputs, outputs, results)):
                args_str = ", ".join(repr(x) for x in inp) if isinstance(inp, list) else repr(inp)
                if result:
                    lines.append(f"Test {i+1}: PASS - {fn_name}({args_str})")
                    passed += 1
                else:
                    lines.append(f"Test {i+1}: FAIL - {fn_name}({args_str}) expected {repr(expected)}")

            total = len(results)
            all_passed = passed == total

            if all_passed:
                lines.append(f"\nAll {total} tests passed!")
            else:
                lines.append(f"\n{passed}/{total} tests passed")

            return "\n".join(lines), all_passed, passed, total

        except Exception as e:
            return f"Execution error: {str(e)}", False, 0, 0

    async def generate_trace(self, problem: Dict) -> GeneratedTrace:
        """Generate a trace with both interleaving and tool execution."""
        prompt_text = problem.get("prompt", "")
        tests = problem.get("tests", {})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INITIAL_PROMPT.format(problem=prompt_text)},
        ]

        trace_steps: List[TraceStep] = []
        code_blocks: List[str] = []  # Accumulated code pieces
        think_count = 0
        code_block_count = 0
        execution_count = 0
        has_verify = False
        had_errors = False
        fix_attempts = 0
        step = 0

        try:
            while step < self.max_steps:
                step += 1

                # Get model response (one step at a time)
                response = await self._call_llm(messages)
                response = response.strip()

                if not response:
                    # Empty response, prompt again
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "Please continue with [THINK] or [CODE]."})
                    continue

                step_type, content = self._parse_step(response)

                if step_type == "think":
                    think_count += 1
                    trace_steps.append(TraceStep(step_type="think", content=content))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "Now write the next [CODE] block (max 3 lines)."})

                elif step_type == "code":
                    code_block_count += 1
                    code_blocks.append(content)
                    trace_steps.append(TraceStep(step_type="code", content=content))
                    messages.append({"role": "assistant", "content": response})

                    # Check if function looks complete
                    accumulated = '\n'.join(code_blocks)
                    if self._looks_complete(accumulated):
                        # Execute the accumulated code
                        execution_count += 1
                        result_text, all_passed, passed, total = self._execute_code(accumulated, tests)

                        if "Error" in result_text or "error" in result_text.lower():
                            had_errors = True
                            trace_steps.append(TraceStep(step_type="error", content=result_text))

                            if fix_attempts < self.max_fix_attempts:
                                fix_attempts += 1
                                messages.append({
                                    "role": "user",
                                    "content": FUNCTION_COMPLETE_PROMPT.format(
                                        result=f"[ERROR]\n{result_text}\n[/ERROR]",
                                        next_instruction=AFTER_FAILURE_PROMPT
                                    )
                                })
                            else:
                                break  # Give up
                        else:
                            trace_steps.append(TraceStep(step_type="result", content=result_text))

                            if all_passed:
                                messages.append({
                                    "role": "user",
                                    "content": FUNCTION_COMPLETE_PROMPT.format(
                                        result=f"[RESULT]\n{result_text}\n[/RESULT]",
                                        next_instruction=AFTER_SUCCESS_PROMPT
                                    )
                                })
                            else:
                                had_errors = True
                                if fix_attempts < self.max_fix_attempts:
                                    fix_attempts += 1
                                    messages.append({
                                        "role": "user",
                                        "content": FUNCTION_COMPLETE_PROMPT.format(
                                            result=f"[RESULT]\n{result_text}\n[/RESULT]",
                                            next_instruction=AFTER_FAILURE_PROMPT
                                        )
                                    })
                                else:
                                    break
                    else:
                        # Not complete yet, ask for more
                        messages.append({"role": "user", "content": CONTINUE_PROMPT})

                elif step_type == "verify":
                    has_verify = True
                    trace_steps.append(TraceStep(step_type="verify", content=content))
                    messages.append({"role": "assistant", "content": response})
                    break  # Done!

                else:
                    # Unknown format, try to salvage
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "Invalid format. Please output exactly ONE [THINK] or [CODE] block."
                    })

            # Calculate final score
            accumulated = '\n'.join(code_blocks)
            if execution_count == 0 and accumulated:
                # Never executed, try now
                result_text, all_passed, passed, total = self._execute_code(accumulated, tests)
                execution_count = 1
                if all_passed:
                    trace_steps.append(TraceStep(step_type="result", content=result_text))
                else:
                    trace_steps.append(TraceStep(step_type="error", content=result_text))
            else:
                # Use last execution results
                _, _, passed, total = self._execute_code(accumulated, tests)

            score = -1.0 + (2.0 * passed / total) if total > 0 else -1.0

            return GeneratedTrace(
                problem=prompt_text,
                messages=messages,
                code=accumulated,
                score=score,
                tests_passed=passed,
                tests_total=total,
                think_count=think_count,
                code_block_count=code_block_count,
                execution_count=execution_count,
                has_verify=has_verify,
                had_errors=had_errors,
                trace=trace_steps,
                model=self.model,
            )

        except Exception as e:
            return GeneratedTrace(
                problem=prompt_text,
                messages=messages,
                error=str(e),
                model=self.model,
            )

    async def generate_traces(
        self,
        problems: List[Dict],
        num_per_problem: int = 1,
        only_success: bool = False,
        only_ideal: bool = False,
        verbose: bool = True,
    ) -> List[GeneratedTrace]:
        """Generate multiple traces."""
        traces = []
        total = len(problems) * num_per_problem

        if verbose and HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Generating {total} interleaved+tool traces...", total=total
                )

                for problem in problems:
                    for _ in range(num_per_problem):
                        trace = await self.generate_trace(problem)

                        # Filter criteria
                        is_ideal = trace.code_block_count >= 3 and trace.execution_count > 0 and trace.score > 0

                        if only_ideal and not is_ideal:
                            progress.update(task, advance=1)
                            continue

                        if only_success and trace.score <= 0:
                            progress.update(task, advance=1)
                            continue

                        traces.append(trace)
                        progress.update(task, advance=1)
        else:
            for i, problem in enumerate(problems):
                for j in range(num_per_problem):
                    if verbose:
                        print(f"Generating trace {i * num_per_problem + j + 1}/{total}...")
                    trace = await self.generate_trace(problem)

                    is_ideal = trace.code_block_count >= 3 and trace.execution_count > 0 and trace.score > 0

                    if only_ideal and not is_ideal:
                        continue
                    if only_success and trace.score <= 0:
                        continue

                    traces.append(trace)

        return traces


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Generate traces with BOTH interleaving AND tool execution"
    )
    parser.add_argument("--output", "-o", type=str, default="traces_interleaved_tools.jsonl")
    parser.add_argument("--num-traces", "-n", type=int, default=10)
    parser.add_argument("--only-success", action="store_true", help="Only save successful traces")
    parser.add_argument("--only-ideal", action="store_true", help="Only save ideal traces (interleaved + tool + success)")
    parser.add_argument("--training-format", action="store_true", help="Save in training format")
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per trace")
    args = parser.parse_args()

    api_key = os.getenv("OLLAMA_API_KEY", "")

    print("=" * 70)
    print("INTERLEAVED + TOOL-BASED TRACE GENERATOR")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Traces per problem: {args.num_traces}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Only success: {args.only_success}")
    print(f"  Only ideal: {args.only_ideal}")
    print(f"  Problems: {len(BUILTIN_PROBLEMS)}")
    print()
    print("This generates IDEAL traces with:")
    print("  - TRUE interleaving (Think→Code→Think→Code, 1-3 lines per block)")
    print("  - REAL tool execution (with [RESULT]/[ERROR] feedback)")
    print("=" * 70)
    print()

    generator = InterleavedToolTraceGenerator(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )

    traces = await generator.generate_traces(
        problems=BUILTIN_PROBLEMS,
        num_per_problem=args.num_traces,
        only_success=args.only_success,
        only_ideal=args.only_ideal,
        verbose=True,
    )

    # Save
    with open(args.output, "w") as f:
        for trace in traces:
            if args.training_format:
                f.write(json.dumps(trace.to_training_format()) + "\n")
            else:
                f.write(json.dumps(trace.to_dict()) + "\n")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Generated {len(traces)} traces")

    if traces:
        scores = [t.score for t in traces]
        success_count = sum(1 for t in traces if t.score > 0)
        avg_thinks = sum(t.think_count for t in traces) / len(traces)
        avg_code_blocks = sum(t.code_block_count for t in traces) / len(traces)
        avg_executions = sum(t.execution_count for t in traces) / len(traces)
        verify_rate = sum(1 for t in traces if t.has_verify) / len(traces)
        ideal_count = sum(
            1 for t in traces
            if t.code_block_count >= 3 and t.execution_count > 0 and t.score > 0
        )

        print(f"\nResults:")
        print(f"  Success rate: {success_count}/{len(traces)} ({100*success_count/len(traces):.1f}%)")
        print(f"  Avg score: {sum(scores)/len(scores):.2f}")

        print(f"\nInterleaving Metrics:")
        print(f"  Avg [THINK] count: {avg_thinks:.1f}")
        print(f"  Avg [CODE] blocks: {avg_code_blocks:.1f}")
        if avg_code_blocks >= 3:
            print(f"  ✓ Good interleaving!")
        else:
            print(f"  ⚠ Low interleaving")

        print(f"\nTool Use Metrics:")
        print(f"  Avg executions: {avg_executions:.1f}")
        print(f"  [VERIFY] rate: {100*verify_rate:.1f}%")
        print(f"  Error recovery traces: {sum(1 for t in traces if t.had_errors and t.score > 0)}")

        print(f"\nIdeal Traces (interleaved + tool + success):")
        print(f"  {ideal_count}/{len(traces)} ({100*ideal_count/len(traces):.1f}%)")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
