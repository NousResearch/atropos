#!/usr/bin/env python3
"""
Synthetic Trace Generator for Interleaved Code Reasoning

Generates training data in JSONL format for fine-tuning or RLHF.
Does NOT require Atropos infrastructure - standalone script.

Output format (JSONL):
{
    "messages": [...],           # Full conversation
    "problem": "...",            # Original problem
    "code": "...",               # Final generated code
    "score": 0.85,               # Execution score (-1 to 1)
    "tests_passed": 3,
    "tests_total": 5,
    "think_count": 4,            # Number of [THINK] markers
    "has_verify": true,          # Whether [VERIFY] was used
    "trace": [...]               # Detailed reasoning steps
}

Usage:
    # Generate traces for built-in problems
    python trace_generator.py --output traces.jsonl --num-traces 100

    # Use custom dataset
    python trace_generator.py --dataset openai/openai_humaneval --output traces.jsonl

    # Filter only successful traces
    python trace_generator.py --output traces.jsonl --only-success
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
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
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer who thinks step-by-step while coding.

You MUST interleave your reasoning with code using these markers:

[THINK] Your reasoning about the problem or next step
[CODE]
your_python_code_here()
[/CODE]
[VERIFY] Trace through your solution with test cases

Example format:
[THINK] I need to understand what we're solving - find two indices that sum to target
[THINK] I'll use a hash map for O(n) lookup instead of O(n²) brute force
[CODE]
def two_sum(nums, target):
    seen = {}
[/CODE]
[THINK] For each number, check if its complement exists in seen
[CODE]
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
[/CODE]
[VERIFY]
Test with nums=[2,7,11,15], target=9:
- i=0: num=2, complement=7, seen={} → not found, add seen[2]=0
- i=1: num=7, complement=2, seen={2:0} → found! return [0,1] ✓
[/VERIFY]

RULES:
1. Use [THINK] before EVERY code block
2. Use [THINK] WAIT if you catch a potential bug
3. Use [VERIFY] to trace through with a concrete example
4. Return ONLY the function, no test code or print statements"""


# System prompt for forced interleaving - strict one-step-at-a-time
FORCED_INTERLEAVE_SYSTEM = """You are an expert Python programmer who thinks step-by-step.

CRITICAL: You must output EXACTLY ONE step at a time!

Valid step patterns:
1. [THINK] your reasoning here (1-2 sentences max)
2. [CODE]\nyour_code_lines\n[/CODE] (max 3 lines of code)
3. [VERIFY]\ntest trace\n[/VERIFY] (only when code is complete)

RULES:
- Output ONLY ONE [THINK] or [CODE] or [VERIFY] per response
- After [CODE], STOP immediately and wait for next prompt
- Do NOT write the entire solution at once
- Each [CODE] block should have 1-3 lines maximum
- Use [VERIFY] only when the function is complete

Example of a SINGLE valid response:
[THINK] I need to initialize a hash map to store seen values and their indices"""


FORCED_CONTINUE_PROMPT = """Continue with the next step.

Remember:
- Output EXACTLY ONE step: either [THINK] or [CODE] (1-3 lines max)
- If the function is complete, use [VERIFY] to trace through a test case
- Do NOT repeat previous code
- Do NOT write the entire solution"""


FORCED_VERIFY_PROMPT = """The function appears complete. Now verify it works correctly.

Output a [VERIFY] block that traces through this test case step by step:
{test_case}

Format:
[VERIFY]
Step-by-step trace showing each variable value
[/VERIFY]"""


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TraceStep:
    """Single step in reasoning trace."""

    step_type: str  # think, code, verify, wait
    content: str


@dataclass
class GeneratedTrace:
    """Complete generated trace."""

    problem: str
    messages: List[Dict[str, str]]
    code: Optional[str] = None
    score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    think_count: int = 0
    has_verify: bool = False
    trace: List[TraceStep] = field(default_factory=list)
    error: Optional[str] = None
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem": self.problem,
            "messages": self.messages,
            "code": self.code,
            "score": self.score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "think_count": self.think_count,
            "has_verify": self.has_verify,
            "trace": [{"type": s.step_type, "content": s.content} for s in self.trace],
            "error": self.error,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    def to_chat_format(self) -> Dict:
        """Convert to simple chat format for fine-tuning."""
        return {
            "messages": self.messages,
            "score": self.score,
        }


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
                ["Aa"],
            ],
            "outputs": [True, False, True, True, False, True],
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
                [[5, 4, -1, 7, 8]],
            ],
            "outputs": [6, 1, -1, -1, 23],
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
            "inputs": [["()"], ["()[]{}"], ["(]"], ["([)]"], ["{[]}"], [""]],
            "outputs": [True, True, False, False, True, True],
        },
    },
    {
        "id": "merge_sorted_lists",
        "prompt": """Write a function `merge_lists(list1, list2)` that merges two sorted lists into one sorted list.

Example:
- merge_lists([1,2,4], [1,3,4]) -> [1,1,2,3,4,4]
- merge_lists([], [0]) -> [0]""",
        "tests": {
            "fn_name": "merge_lists",
            "inputs": [
                [[1, 2, 4], [1, 3, 4]],
                [[], [0]],
                [[1], []],
                [[], []],
            ],
            "outputs": [[1, 1, 2, 3, 4, 4], [0], [1], []],
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
        "id": "reverse_string",
        "prompt": """Write a function `reverse_string(s)` that reverses a string in-place conceptually.
Return the reversed string.

Example:
- reverse_string("hello") -> "olleh"
- reverse_string("") -> "" """,
        "tests": {
            "fn_name": "reverse_string",
            "inputs": [["hello"], [""], ["a"], ["ab"]],
            "outputs": ["olleh", "", "a", "ba"],
        },
    },
]


# ============================================================================
# Trace Generator
# ============================================================================


class TraceGenerator:
    """Generates interleaved reasoning traces."""

    def __init__(
        self,
        base_url: str = "https://ollama.com",
        api_key: str = "",
        model: str = "deepseek-v3.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_api_url(self) -> str:
        """Get correct API endpoint."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3] + "/api/chat"
        return self.base_url + "/api/chat"

    async def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM API."""
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
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=300
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    def _parse_response(self, text: str) -> Tuple[List[TraceStep], str, int, bool]:
        """
        Parse interleaved response.

        Returns: (steps, code, think_count, has_verify)
        """
        steps = []
        code_parts = []
        think_count = 0
        has_verify = False

        # Pattern for markers
        pattern = r"\[(THINK|CODE|VERIFY|WAIT)\](.*?)(?=\[(?:THINK|CODE|VERIFY|WAIT|/CODE)\]|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for marker, content in matches:
            marker = marker.upper()

            if marker == "CODE":
                content = re.sub(r"\[/CODE\].*$", "", content, flags=re.DOTALL)
                lines = content.split("\n")
                while lines and not lines[0].strip():
                    lines.pop(0)
                while lines and not lines[-1].strip():
                    lines.pop()
                content = "\n".join(lines)

                if content.strip():
                    code_parts.append(content)
                    steps.append(TraceStep(step_type="code", content=content))

            elif marker == "THINK":
                content = content.strip()
                if content:
                    if content.upper().startswith("WAIT"):
                        steps.append(TraceStep(step_type="wait", content=content))
                    else:
                        think_count += 1
                        steps.append(TraceStep(step_type="think", content=content))

            elif marker == "VERIFY":
                content = content.strip()
                if content:
                    has_verify = True
                    steps.append(TraceStep(step_type="verify", content=content))

            elif marker == "WAIT":
                content = content.strip()
                if content:
                    steps.append(TraceStep(step_type="wait", content=content))

        # Combine code
        full_code = "\n".join(code_parts)

        # Fallback to markdown
        if not full_code.strip():
            md_pattern = r"```python\s*(.*?)```"
            md_matches = re.findall(md_pattern, text, re.DOTALL)
            if md_matches:
                full_code = md_matches[-1].strip()

        return steps, full_code, think_count, has_verify

    def _execute_and_score(
        self, code: str, tests: Dict
    ) -> Tuple[float, int, int, Optional[str]]:
        """
        Execute code and return score.

        Returns: (score, passed, total, error)
        """
        if not code.strip():
            return -1.0, 0, 0, "No code generated"

        if not tests or not tests.get("fn_name"):
            # Just check syntax
            try:
                compile(code, "<string>", "exec")
                return 0.0, 0, 0, None
            except SyntaxError as e:
                return -1.0, 0, 0, str(e)

        try:
            results, metadata = execute_code_safe(code, tests, timeout=10.0)

            if "error" in metadata:
                return -1.0, 0, len(tests.get("outputs", [])), metadata["error"]

            passed = sum(results)
            total = len(results)

            if total == 0:
                return 0.0, 0, 0, None

            # Score: -1 to 1 based on pass rate
            score = -1.0 + (2.0 * passed / total)
            return score, passed, total, None

        except Exception as e:
            return -1.0, 0, 0, str(e)

    async def generate_trace(
        self, problem: Dict, force_interleave: bool = False
    ) -> GeneratedTrace:
        """Generate a single trace for a problem."""
        if force_interleave:
            return await self._generate_trace_forced(problem)
        return await self._generate_trace_single(problem)

    async def _generate_trace_single(self, problem: Dict) -> GeneratedTrace:
        """Generate trace with single LLM call (original method)."""
        prompt = problem.get("prompt", "")
        tests = problem.get("tests", {})
        problem_id = problem.get("id", "unknown")

        # Build messages
        user_content = f"""Solve this coding problem step-by-step:

{prompt}

Use [THINK], [CODE], and [VERIFY] markers. Return ONLY the function."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # Call LLM
            response = await self._call_llm(messages)

            # Parse response
            steps, code, think_count, has_verify = self._parse_response(response)

            # Execute and score
            score, passed, total, error = self._execute_and_score(code, tests)

            # Build full messages with assistant response
            full_messages = messages + [{"role": "assistant", "content": response}]

            return GeneratedTrace(
                problem=prompt,
                messages=full_messages,
                code=code,
                score=score,
                tests_passed=passed,
                tests_total=total,
                think_count=think_count,
                has_verify=has_verify,
                trace=steps,
                error=error,
                model=self.model,
            )

        except Exception as e:
            return GeneratedTrace(
                problem=prompt,
                messages=messages,
                error=str(e),
                model=self.model,
            )

    async def _generate_trace_forced(self, problem: Dict) -> GeneratedTrace:
        """
        Generate trace with FORCED interleaving via multi-turn conversation.

        Forces the model to output one step at a time by stopping after each
        [/CODE] or [THINK] and re-prompting to continue.
        """
        prompt = problem.get("prompt", "")
        tests = problem.get("tests", {})

        # Build initial message
        initial_user = f"""Solve this problem step by step. Output ONLY your first [THINK] step.

Problem:
{prompt}

Start with ONE [THINK] explaining your approach (1-2 sentences only):"""

        messages = [
            {"role": "system", "content": FORCED_INTERLEAVE_SYSTEM},
            {"role": "user", "content": initial_user},
        ]

        all_steps: List[TraceStep] = []
        code_parts: List[str] = []
        think_count = 0
        has_verify = False
        max_turns = 20  # Safety limit
        turn = 0

        try:
            while turn < max_turns:
                turn += 1

                # Call LLM for one step
                response = await self._call_llm(messages)
                response = response.strip()

                # Parse this single response
                step_type, content = self._parse_single_step(response)

                if step_type is None:
                    # Model didn't follow format, try to salvage
                    # Check if it output a full solution
                    steps, code, tc, hv = self._parse_response(response)
                    if code:
                        # Model ignored instruction, use what we got
                        for s in steps:
                            all_steps.append(s)
                            if s.step_type == "think":
                                think_count += 1
                            elif s.step_type == "code":
                                code_parts.append(s.content)
                        has_verify = hv
                        think_count = max(think_count, tc)
                        break
                    else:
                        # Retry with stricter prompt
                        messages.append({"role": "assistant", "content": response})
                        messages.append(
                            {
                                "role": "user",
                                "content": "Invalid format. Output EXACTLY ONE step: [THINK] your thought OR [CODE]\\ncode\\n[/CODE]",
                            }
                        )
                        continue

                # Add step
                all_steps.append(TraceStep(step_type=step_type, content=content))
                messages.append({"role": "assistant", "content": response})

                if step_type == "think":
                    think_count += 1
                    # Prompt for code or next think
                    messages.append(
                        {
                            "role": "user",
                            "content": "Now output the next step: either another [THINK] or [CODE]\\nyour_code\\n[/CODE] (max 3 lines)",
                        }
                    )

                elif step_type == "code":
                    code_parts.append(content)
                    # Check if function seems complete
                    combined_code = "\n".join(code_parts)
                    if self._looks_complete(combined_code):
                        # Ask for verification
                        test_example = self._get_test_example(tests)
                        messages.append(
                            {
                                "role": "user",
                                "content": FORCED_VERIFY_PROMPT.format(
                                    test_case=test_example
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {"role": "user", "content": FORCED_CONTINUE_PROMPT}
                        )

                elif step_type == "verify":
                    has_verify = True
                    break  # Done!

                elif step_type == "wait":
                    # Bug catch, continue
                    messages.append(
                        {
                            "role": "user",
                            "content": "Good catch! Now continue with [CODE] to fix it or [THINK] to reason more.",
                        }
                    )

            # Combine all code
            full_code = "\n".join(code_parts)

            # Execute and score
            score, passed, total, error = self._execute_and_score(full_code, tests)

            # Build combined assistant response for training
            combined_response = self._combine_steps_to_response(all_steps)
            training_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },  # Use original system prompt for training
                {
                    "role": "user",
                    "content": f"Solve this coding problem step-by-step:\n\n{prompt}\n\nUse [THINK], [CODE], and [VERIFY] markers.",
                },
                {"role": "assistant", "content": combined_response},
            ]

            return GeneratedTrace(
                problem=prompt,
                messages=training_messages,
                code=full_code,
                score=score,
                tests_passed=passed,
                tests_total=total,
                think_count=think_count,
                has_verify=has_verify,
                trace=all_steps,
                error=error,
                model=self.model,
            )

        except Exception as e:
            return GeneratedTrace(
                problem=prompt,
                messages=messages,
                error=str(e),
                model=self.model,
            )

    def _parse_single_step(self, text: str) -> Tuple[Optional[str], str]:
        """Parse a single step from response. Returns (step_type, content) or (None, '') if invalid."""
        text = text.strip()

        # Check for [VERIFY]
        verify_match = re.search(
            r"\[VERIFY\](.*?)(?:\[/VERIFY\]|$)", text, re.DOTALL | re.IGNORECASE
        )
        if verify_match:
            return "verify", verify_match.group(1).strip()

        # Check for [CODE]...[/CODE]
        code_match = re.search(
            r"\[CODE\](.*?)\[/CODE\]", text, re.DOTALL | re.IGNORECASE
        )
        if code_match:
            content = code_match.group(1)
            lines = content.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            return "code", "\n".join(lines)

        # Check for [THINK]
        think_match = re.search(
            r"\[THINK\]\s*(.*?)(?:\[|$)", text, re.DOTALL | re.IGNORECASE
        )
        if think_match:
            content = think_match.group(1).strip()
            if content.upper().startswith("WAIT"):
                return "wait", content
            return "think", content

        # Check for [WAIT]
        wait_match = re.search(
            r"\[WAIT\]\s*(.*?)(?:\[|$)", text, re.DOTALL | re.IGNORECASE
        )
        if wait_match:
            return "wait", wait_match.group(1).strip()

        return None, ""

    def _looks_complete(self, code: str) -> bool:
        """Check if code looks like a complete function."""
        if not code.strip():
            return False

        # Must have a function definition
        if not re.search(r"def\s+\w+\s*\(", code):
            return False

        # Must have a return statement
        if not re.search(r"\breturn\b", code):
            return False

        # Check indentation suggests function body is complete
        lines = code.strip().split("\n")
        if len(lines) < 2:
            return False

        # Last non-empty line should be at function body level (indented)
        last_line = lines[-1]
        if (
            last_line.strip()
            and not last_line.startswith(" ")
            and not last_line.startswith("\t")
        ):
            # Last line is at module level, might be incomplete
            return "return" in last_line

        return True

    def _get_test_example(self, tests: Dict) -> str:
        """Get a test example for verification prompt."""
        if not tests or not tests.get("inputs"):
            return "an example input"

        fn_name = tests.get("fn_name", "func")
        inputs = tests["inputs"][0] if tests["inputs"] else []
        expected = tests["outputs"][0] if tests.get("outputs") else "expected_output"

        # Format inputs
        if isinstance(inputs, list):
            args = ", ".join(repr(x) for x in inputs)
        else:
            args = repr(inputs)

        return f"{fn_name}({args}) should return {expected}"

    def _combine_steps_to_response(self, steps: List[TraceStep]) -> str:
        """Combine steps into a single formatted response for training."""
        parts = []
        for step in steps:
            if step.step_type == "think":
                parts.append(f"[THINK] {step.content}")
            elif step.step_type == "wait":
                parts.append(f"[THINK] WAIT - {step.content}")
            elif step.step_type == "code":
                parts.append(f"[CODE]\n{step.content}\n[/CODE]")
            elif step.step_type == "verify":
                parts.append(f"[VERIFY]\n{step.content}\n[/VERIFY]")
        return "\n".join(parts)

    async def generate_traces(
        self,
        problems: List[Dict],
        num_per_problem: int = 1,
        only_success: bool = False,
        force_interleave: bool = False,
        verbose: bool = True,
    ) -> List[GeneratedTrace]:
        """Generate multiple traces."""
        traces = []
        total = len(problems) * num_per_problem

        mode_str = "FORCED interleaved" if force_interleave else "single-call"

        if verbose and HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Generating {total} traces ({mode_str})...", total=total
                )

                for problem in problems:
                    for _ in range(num_per_problem):
                        trace = await self.generate_trace(
                            problem, force_interleave=force_interleave
                        )

                        if only_success and trace.score <= 0:
                            progress.update(task, advance=1)
                            continue

                        traces.append(trace)
                        progress.update(task, advance=1)
        else:
            for i, problem in enumerate(problems):
                for j in range(num_per_problem):
                    if verbose:
                        print(
                            f"Generating trace {i * num_per_problem + j + 1}/{total} ({mode_str})..."
                        )
                    trace = await self.generate_trace(
                        problem, force_interleave=force_interleave
                    )
                    if only_success and trace.score <= 0:
                        continue
                    traces.append(trace)

        return traces


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic traces for code reasoning"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="traces.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--num-traces",
        "-n",
        type=int,
        default=10,
        help="Number of traces per problem",
    )
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Only save successful traces (score > 0)",
    )
    parser.add_argument(
        "--chat-format",
        action="store_true",
        help="Save in simple chat format (messages + score only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
        help="Model to use",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        help="API base URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--force-interleave",
        action="store_true",
        help="Force true interleaving via multi-turn conversation (slower but more granular)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OLLAMA_API_KEY", "")

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Output: {args.output}")
    print(f"  Traces per problem: {args.num_traces}")
    print(f"  Only success: {args.only_success}")
    print(f"  Force interleave: {args.force_interleave}")
    print(f"  Problems: {len(BUILTIN_PROBLEMS)}")
    print()

    if args.force_interleave:
        print(
            "⚡ Using FORCED interleaving mode (multi-turn, slower but true granular reasoning)"
        )
        print()

    # Initialize generator
    generator = TraceGenerator(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
    )

    # Generate traces
    traces = await generator.generate_traces(
        problems=BUILTIN_PROBLEMS,
        num_per_problem=args.num_traces,
        only_success=args.only_success,
        force_interleave=args.force_interleave,
        verbose=True,
    )

    # Save to JSONL
    with open(args.output, "w") as f:
        for trace in traces:
            if args.chat_format:
                f.write(json.dumps(trace.to_chat_format()) + "\n")
            else:
                f.write(json.dumps(trace.to_dict()) + "\n")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Generated {len(traces)} traces")

    if traces:
        scores = [t.score for t in traces]
        success_count = sum(1 for t in traces if t.score > 0)
        avg_thinks = sum(t.think_count for t in traces) / len(traces)
        verify_rate = sum(1 for t in traces if t.has_verify) / len(traces)

        # Count code blocks to measure interleaving
        avg_code_blocks = sum(
            sum(1 for s in t.trace if s.step_type == "code") for t in traces
        ) / len(traces)

        print(
            f"  Success rate: {success_count}/{len(traces)} ({100*success_count/len(traces):.1f}%)"
        )
        print(f"  Avg score: {sum(scores)/len(scores):.2f}")
        print(f"  Avg [THINK] count: {avg_thinks:.1f}")
        print(
            f"  Avg [CODE] blocks: {avg_code_blocks:.1f}"
        )  # Key metric for interleaving
        print(f"  [VERIFY] usage: {100*verify_rate:.1f}%")

        if args.force_interleave:
            print(
                f"\n  ⚡ Interleaving ratio: {avg_thinks:.1f} thinks / {avg_code_blocks:.1f} code blocks"
            )
            if avg_code_blocks >= 3:
                print(f"  ✓ Good interleaving achieved!")
            else:
                print(f"  ⚠ Low interleaving - model may be combining code blocks")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
