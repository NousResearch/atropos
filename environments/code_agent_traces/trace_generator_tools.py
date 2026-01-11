#!/usr/bin/env python3
"""
Tool-Based Trace Generator for Interleaved Code Reasoning

Unlike trace_generator.py which uses marker-only approach, this version
implements TRUE tool use where the model:
1. Outputs code in [CODE] blocks
2. Receives execution results via [RESULT] or [ERROR]
3. Can iterate and fix bugs based on feedback
4. Creates richer training data with error-recovery patterns

This produces traces like:
    [THINK] I'll implement two_sum using a hash map
    [CODE]
    def two_sum(nums, target):
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen:
                return [seen[target-num], i]
            seen[num] = i
    [/CODE]
    [RESULT]
    Test 1: two_sum([2,7,11,15], 9) = [0, 1] - PASS
    Test 2: two_sum([3,2,4], 6) = None - FAIL (expected [1, 2])
    [/RESULT]
    [THINK] Forgot to return empty list when no solution. Let me fix.
    [CODE]
    def two_sum(nums, target):
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen:
                return [seen[target-num], i]
            seen[num] = i
        return []
    [/CODE]
    [RESULT]
    All 4 tests passed!
    [/RESULT]
    [VERIFY] Solution uses O(n) time with hash map lookup...

Usage:
    python trace_generator_tools.py --output traces.jsonl --num-traces 10
"""

import argparse
import asyncio
import json
import os
import re
import sys
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
# System Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer with access to a code execution tool.

## CRITICAL RULES

1. You can ONLY output these XML tags:
   - <think>your reasoning</think>
   - <code>your code</code>
   - <verify>your verification</verify> (only after tests pass)

2. You must NEVER output <result> or <error> - those come from the SYSTEM only!

3. After writing <code>...</code>, STOP and WAIT for test results.

## Workflow

1. <think> - Reason about the problem
2. <code> - Write code, then STOP
3. << SYSTEM provides <result> or <error> >>
4. <think> - Analyze results, fix if needed
5. Repeat until success
6. <verify> - Explain your solution

## Example (YOUR output only, not system messages)

<think>I need to find two indices that sum to target. I'll use a hash map.</think>
<code>
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target-num], i]
        seen[num] = i
    return []
</code>

<< STOP HERE - wait for system to execute and return results >>

## After receiving results, you may continue:

<think>The test failed because... Let me fix it.</think>
<code>
# fixed code here
</code>

## When all tests pass:

<verify>
The solution works by... (trace through the algorithm)
</verify>"""


INITIAL_PROMPT_TEMPLATE = """Solve this coding problem. Your code will be executed and tested automatically.

Problem:
{problem}

Instructions:
1. Start with <think> to analyze the problem
2. Write your solution in <code>...</code>
3. STOP after </code> - do NOT write <result> or test output
4. Wait for the system to execute your code and show results"""


CONTINUE_AFTER_RESULT = """The code was executed. Review the results above.

If all tests passed: Use <verify> to trace through your solution and confirm it's correct.
If tests failed: Use <think> to analyze the failure, then <code> to fix it.

Remember: Maximum 5 code iterations. Current iteration: {iteration}/5"""


CONTINUE_AFTER_ERROR = """Your code raised an error. Review the error message above.

Use <think> to understand what went wrong, then <code> to fix it.

Iteration: {iteration}/5"""


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
    """Complete generated trace with tool interactions."""

    problem: str
    messages: List[Dict[str, str]]  # Full conversation for training
    code: Optional[str] = None  # Final working code
    score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    think_count: int = 0
    code_iterations: int = 0  # How many times code was executed
    has_verify: bool = False
    had_errors: bool = False  # Did model encounter and fix errors?
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
            "code_iterations": self.code_iterations,
            "has_verify": self.has_verify,
            "had_errors": self.had_errors,
            "trace": [{"type": s.step_type, "content": s.content} for s in self.trace],
            "error": self.error,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    def to_training_format(self) -> Dict:
        """Convert to training format - single assistant response with full trace."""
        # Combine all steps into single assistant message
        combined_response = self._combine_trace()

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": INITIAL_PROMPT_TEMPLATE.format(problem=self.problem),
                },
                {"role": "assistant", "content": combined_response},
            ],
            "score": self.score,
            "had_errors": self.had_errors,
            "code_iterations": self.code_iterations,
        }

    def _combine_trace(self) -> str:
        """Combine trace steps into formatted response."""
        parts = []
        for step in self.trace:
            if step.step_type == "think":
                parts.append(f"<think>{step.content}</think>")
            elif step.step_type == "code":
                parts.append(f"<code>\n{step.content}\n</code>")
            elif step.step_type == "result":
                parts.append(f"<result>\n{step.content}\n</result>")
            elif step.step_type == "error":
                parts.append(f"<error>\n{step.content}\n</error>")
            elif step.step_type == "verify":
                parts.append(f"<verify>\n{step.content}\n</verify>")
        return "\n\n".join(parts)


# ============================================================================
# Built-in Problems (same as trace_generator.py)
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
# Tool-Based Trace Generator
# ============================================================================


class ToolBasedTraceGenerator:
    """Generates traces with actual tool execution and feedback."""

    def __init__(
        self,
        base_url: str = "https://ollama.com",
        api_key: str = "",
        model: str = "deepseek-v3.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_iterations: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations

    def _get_api_url(self) -> str:
        """Get correct API endpoint."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3] + "/api/chat"
        return self.base_url + "/api/chat"

    async def _call_llm(
        self, messages: List[Dict], stop_after_code: bool = True
    ) -> str:
        """Call LLM API with stop sequences to prevent hallucination."""
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
                # Stop sequences to prevent model from hallucinating results
                "stop": (
                    ["<result>", "<error>", "\n<result>", "\n<error>"]
                    if stop_after_code
                    else []
                ),
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=300
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response = data.get("message", {}).get("content", "")

                # Extra safety: strip any hallucinated results that got through
                response = self._strip_hallucinated_results(response)
                return response

    def _strip_hallucinated_results(self, text: str) -> str:
        """Remove any hallucinated <result> or <error> blocks from model output."""
        # These tags should ONLY come from the system, not the model
        # If model outputs them, it's hallucinating

        # Find and remove <result>...</result> or <result>... to end
        text = re.sub(
            r"<result>.*?(?:</result>|$)", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Find and remove <error>...</error> or <error>... to end
        text = re.sub(
            r"<error>.*?(?:</error>|$)", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        return text.strip()

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from <code>...</code> block."""
        match = re.search(r"<code>(.*?)</code>", text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1)
            lines = code.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            return "\n".join(lines)
        return None

    def _extract_thinks(self, text: str) -> List[str]:
        """Extract all <think> blocks."""
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    def _extract_verify(self, text: str) -> Optional[str]:
        """Extract <verify> block."""
        match = re.search(r"<verify>(.*?)</verify>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _execute_code(self, code: str, tests: Dict) -> Tuple[str, bool, int, int]:
        """
        Execute code and format results.

        Returns: (result_text, all_passed, passed_count, total_count)
        """
        if not code.strip():
            return "Error: No code provided", False, 0, 0

        try:
            results, metadata = execute_code_safe(code, tests, timeout=10.0)

            if "error" in metadata:
                return (
                    f"Error: {metadata['error']}",
                    False,
                    0,
                    len(tests.get("outputs", [])),
                )

            # Format results
            fn_name = tests.get("fn_name", "func")
            inputs = tests.get("inputs", [])
            outputs = tests.get("outputs", [])

            lines = []
            passed = 0
            for i, (inp, expected, result) in enumerate(zip(inputs, outputs, results)):
                if isinstance(inp, list):
                    args_str = ", ".join(repr(x) for x in inp)
                else:
                    args_str = repr(inp)

                if result:
                    lines.append(f"Test {i+1}: PASS - {fn_name}({args_str})")
                    passed += 1
                else:
                    # Try to get actual output for debugging
                    lines.append(
                        f"Test {i+1}: FAIL - {fn_name}({args_str}) expected {repr(expected)}"
                    )

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
        """Generate a trace with tool execution feedback loop."""
        prompt_text = problem.get("prompt", "")
        tests = problem.get("tests", {})

        # Initialize conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INITIAL_PROMPT_TEMPLATE.format(problem=prompt_text),
            },
        ]

        trace_steps: List[TraceStep] = []
        think_count = 0
        code_iterations = 0
        had_errors = False
        has_verify = False
        last_code = None
        last_passed = 0
        last_total = 0
        all_passed = False

        try:
            while code_iterations < self.max_iterations:
                # Get model response
                response = await self._call_llm(messages)
                response = response.strip()

                # Extract thinks
                thinks = self._extract_thinks(response)
                for think in thinks:
                    trace_steps.append(TraceStep(step_type="think", content=think))
                    think_count += 1

                # Check for verify (means model thinks it's done)
                verify = self._extract_verify(response)
                if verify:
                    trace_steps.append(TraceStep(step_type="verify", content=verify))
                    has_verify = True
                    messages.append({"role": "assistant", "content": response})
                    break

                # Extract and execute code
                code = self._extract_code(response)
                if code:
                    last_code = code
                    code_iterations += 1
                    trace_steps.append(TraceStep(step_type="code", content=code))

                    # Execute code
                    result_text, all_passed, passed, total = self._execute_code(
                        code, tests
                    )
                    last_passed = passed
                    last_total = total

                    if "Error" in result_text or "error" in result_text.lower():
                        had_errors = True
                        trace_steps.append(
                            TraceStep(step_type="error", content=result_text)
                        )
                        # Add to conversation
                        messages.append({"role": "assistant", "content": response})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"<error>\n{result_text}\n</error>\n\n{CONTINUE_AFTER_ERROR.format(iteration=code_iterations)}",
                            }
                        )
                    else:
                        trace_steps.append(
                            TraceStep(step_type="result", content=result_text)
                        )
                        messages.append({"role": "assistant", "content": response})

                        if all_passed:
                            # Ask for verification
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"<result>\n{result_text}\n</result>\n\nExcellent! All tests passed. Now use <verify> to trace through your solution and explain why it works.",
                                }
                            )
                        else:
                            had_errors = True  # Test failures count as errors
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"<result>\n{result_text}\n</result>\n\n{CONTINUE_AFTER_RESULT.format(iteration=code_iterations)}",
                                }
                            )
                else:
                    # No code in response, might just be thinking
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please write your solution in a <code>...</code> block.",
                        }
                    )

            # Calculate score
            if last_total > 0:
                score = -1.0 + (2.0 * last_passed / last_total)
            else:
                score = -1.0

            return GeneratedTrace(
                problem=prompt_text,
                messages=messages,
                code=last_code,
                score=score,
                tests_passed=last_passed,
                tests_total=last_total,
                think_count=think_count,
                code_iterations=code_iterations,
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
                    f"Generating {total} tool-based traces...", total=total
                )

                for problem in problems:
                    for _ in range(num_per_problem):
                        trace = await self.generate_trace(problem)

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
                            f"Generating trace {i * num_per_problem + j + 1}/{total}..."
                        )
                    trace = await self.generate_trace(problem)
                    if only_success and trace.score <= 0:
                        continue
                    traces.append(trace)

        return traces


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Generate tool-based traces for code reasoning (with execution feedback)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="traces_tools.jsonl",
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
        "--training-format",
        action="store_true",
        help="Save in training format (single assistant message with full trace)",
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
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum code execution iterations per problem",
    )
    args = parser.parse_args()

    api_key = os.getenv("OLLAMA_API_KEY", "")

    print("=" * 60)
    print("TOOL-BASED TRACE GENERATOR")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Output: {args.output}")
    print(f"  Traces per problem: {args.num_traces}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Only success: {args.only_success}")
    print(f"  Training format: {args.training_format}")
    print(f"  Problems: {len(BUILTIN_PROBLEMS)}")
    print()
    print("This generator executes code and provides feedback to the model.")
    print("Model can iterate to fix bugs based on test results.")
    print()

    # Initialize generator
    generator = ToolBasedTraceGenerator(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
    )

    # Generate traces
    traces = await generator.generate_traces(
        problems=BUILTIN_PROBLEMS,
        num_per_problem=args.num_traces,
        only_success=args.only_success,
        verbose=True,
    )

    # Save to JSONL
    with open(args.output, "w") as f:
        for trace in traces:
            if args.training_format:
                f.write(json.dumps(trace.to_training_format()) + "\n")
            else:
                f.write(json.dumps(trace.to_dict()) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated {len(traces)} traces")

    if traces:
        scores = [t.score for t in traces]
        success_count = sum(1 for t in traces if t.score > 0)
        avg_thinks = sum(t.think_count for t in traces) / len(traces)
        avg_iterations = sum(t.code_iterations for t in traces) / len(traces)
        error_recovery_count = sum(1 for t in traces if t.had_errors and t.score > 0)
        verify_rate = sum(1 for t in traces if t.has_verify) / len(traces)

        print(f"\nResults:")
        print(
            f"  Success rate: {success_count}/{len(traces)} ({100*success_count/len(traces):.1f}%)"
        )
        print(f"  Avg score: {sum(scores)/len(scores):.2f}")
        print(f"  Avg [THINK] count: {avg_thinks:.1f}")
        print(f"  Avg code iterations: {avg_iterations:.1f}")
        print(f"  [VERIFY] usage: {100*verify_rate:.1f}%")

        print(f"\nTool Usage Metrics:")
        print(
            f"  Traces with error recovery: {error_recovery_count} ({100*error_recovery_count/len(traces):.1f}%)"
        )
        print(
            f"  Multi-iteration traces: {sum(1 for t in traces if t.code_iterations > 1)}"
        )

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
