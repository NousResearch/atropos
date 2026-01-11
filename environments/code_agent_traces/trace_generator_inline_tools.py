#!/usr/bin/env python3
"""
Inline Tool Trace Generator
============================

Generates agent traces with **inline tool calls inside <think> blocks**.
Based on the pattern from tool_use_interleaved_thinking.py.

Key Features:
- Tool calls (`<tool_call>`) are emitted INSIDE an open `<think>` block
- Execution happens, `<tool_response>` is injected, model continues
- Final code is extracted and tested after `</think>` closes
- Multi-turn conversation with real execution feedback

Format:
```
<think>
I need to implement two_sum with a hash map...

<tool_call>{"name": "python", "arguments": {"code": "def two_sum(nums, target):..."}}</tool_call>
<tool_response>{"stdout": "", "passed": 2, "total": 4, "error": null}</tool_response>

The tests show 2/4 passed. Let me check the edge case...

<tool_call>{"name": "python", "arguments": {"code": "def two_sum(nums, target):..."}}</tool_call>
<tool_response>{"stdout": "", "passed": 4, "total": 4, "error": null}</tool_response>

All tests pass now.
</think>

Final solution: [code block]
```

Usage:
    python trace_generator_inline_tools.py --output traces.jsonl --num-traces 10
    python trace_generator_inline_tools.py --output traces.jsonl --only-success
    python trace_generator_inline_tools.py --output traces.jsonl --training-format
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import aiohttp

# Add parent directory to path for local_executor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_executor import execute_code_safe

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Configuration for trace generation."""

    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    )
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "deepseek-v3.2:cloud")
    )
    max_turns: int = 5
    max_tokens_per_turn: int = 1024
    temperature: float = 0.7
    debug: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Python coding AI with access to a code execution tool.

You should reason through problems step by step inside <think> tags, using tool calls to test your code.

Available tool:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "python",
      "description": "Execute Python code and run tests. Returns stdout, test results, and any errors.",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Complete Python code including the function definition"
          }
        },
        "required": ["code"]
      }
    }
  }
]
</tools>

Format for tool calls (INSIDE <think> block):
<tool_call>{"name": "python", "arguments": {"code": "your code here"}}</tool_call>

The system will execute your code and respond with:
<tool_response>{"stdout": "...", "passed": N, "total": M, "error": null}</tool_response>

After your reasoning is complete, close the </think> block and provide the final solution.

IMPORTANT JSON FORMATTING:
- Use SINGLE escape (\\n) for newlines in code strings
- NEVER use double (\\\\n) or quadruple (\\\\\\\\n) escaping
- Keep entire JSON on a single line

CORRECT format:
{"code": "def foo():\\n    return 1"}

INCORRECT formats (DO NOT USE):
❌ {"code": "def foo():\\\\n    return 1"}
❌ {"code": "def foo():\\\\\\\\n    return 1"}

=== Example 1: Simple function ===
<think>
I need to implement a function that adds two numbers.

<tool_call>{"name": "python", "arguments": {"code": "def add(a, b):\\n    return a + b"}}</tool_call>
<tool_response>{"stdout": "", "passed": 3, "total": 3, "error": null}</tool_response>

All tests pass.
</think>

```python
def add(a, b):
    return a + b
```

=== Example 2: Fix failing tests ===
<think>
I need to find the maximum element in a list.

<tool_call>{"name": "python", "arguments": {"code": "def find_max(lst):\\n    return max(lst)"}}</tool_call>
<tool_response>{"stdout": "", "passed": 2, "total": 3, "error": "max() arg is an empty sequence"}</tool_response>

The function fails on empty list. I need to handle that edge case.

<tool_call>{"name": "python", "arguments": {"code": "def find_max(lst):\\n    if not lst:\\n        return None\\n    return max(lst)"}}</tool_call>
<tool_response>{"stdout": "", "passed": 3, "total": 3, "error": null}</tool_response>

Now all tests pass with proper empty list handling.
</think>

```python
def find_max(lst):
    if not lst:
        return None
    return max(lst)
```

=== Example 3: Two sum with hash map ===
<think>
I need to find two indices that sum to target. I'll use a hash map for O(n) time.

<tool_call>{"name": "python", "arguments": {"code": "def two_sum(nums, target):\\n    seen = {}\\n    for i, num in enumerate(nums):\\n        complement = target - num\\n        if complement in seen:\\n            return [seen[complement], i]\\n        seen[num] = i\\n    return []"}}</tool_call>
<tool_response>{"stdout": "", "passed": 3, "total": 3, "error": null}</tool_response>

The hash map approach works correctly.
</think>

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Problems Dataset
# ─────────────────────────────────────────────────────────────────────────────


def get_problems() -> list[dict]:
    """Load problems from HumanEval or use built-in examples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test")
        problems = []
        for item in ds:
            # Parse test cases from prompt
            tests = _extract_tests_from_prompt(
                item["prompt"],
                item["canonical_solution"],
                item["test"],
                item["entry_point"],
            )
            problems.append(
                {
                    "id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "canonical_solution": item["canonical_solution"],
                    "tests": tests,
                }
            )
        return problems
    except Exception as e:
        print(f"[WARN] Could not load HumanEval: {e}")
        return _builtin_problems()


def _extract_tests_from_prompt(
    prompt: str, solution: str, test_code: str, entry_point: str
) -> list[dict]:
    """Extract test cases from HumanEval test code."""
    tests = []

    # Try to parse assert statements from test code
    assert_pattern = re.compile(
        r"assert\s+(\w+)\s*\((.*?)\)\s*==\s*(.*?)(?:\n|$|,)", re.MULTILINE
    )
    for match in assert_pattern.finditer(test_code):
        func_name, args, expected = match.groups()
        if func_name == entry_point:
            tests.append(
                {
                    "input": args.strip(),
                    "expected": expected.strip(),
                }
            )

    # Also try to extract from docstring examples (>>> lines)
    if not tests:
        docstring_pattern = re.compile(
            rf">>>\s*{entry_point}\s*\((.*?)\)\s*\n\s*(.+?)(?:\n|$)", re.MULTILINE
        )
        for match in docstring_pattern.finditer(prompt):
            args, expected = match.groups()
            tests.append(
                {
                    "input": args.strip(),
                    "expected": expected.strip(),
                }
            )

    return tests[:5]  # Limit to 5 tests, return empty if none found


def _builtin_problems() -> list[dict]:
    """Built-in problems if HumanEval is not available."""
    return [
        {
            "id": "two_sum",
            "prompt": '''def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Given an array of integers nums and an integer target,
    return indices of the two numbers such that they add up to target.

    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    >>> two_sum([3, 2, 4], 6)
    [1, 2]
    """
''',
            "entry_point": "two_sum",
            "canonical_solution": """    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""",
            "tests": [
                {"input": "[2, 7, 11, 15], 9", "expected": "[0, 1]"},
                {"input": "[3, 2, 4], 6", "expected": "[1, 2]"},
                {"input": "[3, 3], 6", "expected": "[0, 1]"},
            ],
        },
        {
            "id": "is_palindrome",
            "prompt": '''def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome, considering only alphanumeric characters
    and ignoring cases.

    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    >>> is_palindrome("race a car")
    False
    """
''',
            "entry_point": "is_palindrome",
            "canonical_solution": """    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
""",
            "tests": [
                {"input": '"A man, a plan, a canal: Panama"', "expected": "True"},
                {"input": '"race a car"', "expected": "False"},
                {"input": '""', "expected": "True"},
            ],
        },
        {
            "id": "max_subarray",
            "prompt": '''def max_subarray(nums: list[int]) -> int:
    """
    Find the contiguous subarray with the largest sum.

    >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    6
    >>> max_subarray([1])
    1
    >>> max_subarray([5, 4, -1, 7, 8])
    23
    """
''',
            "entry_point": "max_subarray",
            "canonical_solution": """    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
""",
            "tests": [
                {"input": "[-2, 1, -3, 4, -1, 2, 1, -5, 4]", "expected": "6"},
                {"input": "[1]", "expected": "1"},
                {"input": "[5, 4, -1, 7, 8]", "expected": "23"},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama API Client
# ─────────────────────────────────────────────────────────────────────────────


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, config: Config):
        self.config = config
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def chat_completion(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Get a chat completion from Ollama."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with.")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens_per_turn,
            },
        }

        if stop:
            payload["options"]["stop"] = stop

        url = f"{self.config.base_url}/api/chat"

        if self.config.debug:
            print(f"\n[API] POST {url}")
            print(f"[API] Model: {self.config.model}")
            print(f"[API] Messages: {len(messages)} messages")
            if messages:
                last_msg = messages[-1]
                print(f"[API] Last message role: {last_msg.get('role')}")
                content = last_msg.get("content", "")[:200]
                print(f"[API] Last message preview: {content}...")

        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Ollama API error {resp.status}: {text}")
            data = await resp.json()
            content = data.get("message", {}).get("content", "")

            if self.config.debug:
                print(f"[API] Response length: {len(content)} chars")
                print(f"[API] Response preview: {content[:300]}...")

            return content


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────────────────────────────────────


def execute_tool(call_json: dict, problem: dict) -> dict:
    """Execute a tool call and return the result."""
    name = call_json.get("name", "")
    args = call_json.get("arguments", {})

    if name == "python":
        code = args.get("code", "")

        # Prepare test cases for execute_code_safe
        test_inputs = []
        test_outputs = []
        for test in problem["tests"]:
            try:
                # Evaluate input and expected values
                input_val = eval(test["input"])
                expected_val = eval(test["expected"])
                # Handle tuple inputs properly
                if isinstance(input_val, tuple):
                    test_inputs.append(list(input_val))
                else:
                    test_inputs.append([input_val])
                test_outputs.append(expected_val)
            except Exception:
                # Skip malformed tests
                pass

        test_cases = {
            "fn_name": problem["entry_point"],
            "inputs": test_inputs,
            "outputs": test_outputs,
        }

        results, metadata = execute_code_safe(code, test_cases, timeout=5.0)

        # Parse results
        stdout = metadata.get("output", "")
        error = metadata.get("error")
        passed = sum(1 for r in results if r)
        total = len(results)

        return {
            "stdout": stdout,
            "passed": passed,
            "total": total,
            "error": error,
        }
    else:
        return {"error": f"Unknown tool: {name}"}


def _extract_code_via_regex(content: str) -> str | None:
    """
    Стратегия 3: Извлечь код напрямую через regex.
    Работает даже с docstrings и сложным форматированием.
    """
    # Паттерн для "code": "..." с возможными переносами
    patterns = [
        # Стандартный формат с кавычками
        r'"code"\s*:\s*"((?:[^"\\]|\\.|[\n\r\t])*)"',
        # Формат с тройными кавычками (иногда модели так делают)
        r'"code"\s*:\s*"""(.*?)"""',
        r'"code"\s*:\s*\'\'\'(.*?)\'\'\'',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code = match.group(1)
            # Декодируем escape sequences
            try:
                # Заменяем escaped newlines на реальные
                code = code.replace("\\n", "\n")
                code = code.replace("\\t", "\t")
                code = code.replace("\\r", "\r")
                code = code.replace('\\"', '"')
                code = code.replace("\\\\", "\\")
            except Exception:
                pass
            return code

    return None


def parse_tool_call(text: str, debug: bool = False) -> dict | None:
    """
    Робастный парсер tool_call с 3 стратегиями fallback.

    Решает проблему: модель генерирует реальные newlines (0x0A) вместо \\n
    """
    # Найти последний <tool_call> тег
    last_pos = text.rfind("<tool_call>")
    if last_pos == -1:
        return None

    json_start = last_pos + len("<tool_call>")

    # Найти </tool_call> если есть
    end_tag_pos = text.find("</tool_call>", json_start)
    if end_tag_pos != -1:
        json_text = text[json_start:end_tag_pos].strip()
    else:
        # Нет закрывающего тега - до конца или следующего тега
        next_tag = text.find("<", json_start)
        if next_tag != -1 and next_tag > json_start:
            json_text = text[json_start:next_tag].strip()
        else:
            json_text = text[json_start:].strip()

    if debug:
        print(f"[DEBUG] Raw JSON ({len(json_text)} chars): {json_text[:200]}...")

    # ═══════════════════════════════════════════════════════════════════════════
    # Стратегия 1: Прямой JSON parse (для корректного JSON)
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        result = json.loads(json_text)
        if debug:
            print("[DEBUG] Strategy 1 (direct JSON) succeeded")
        return result
    except json.JSONDecodeError as e:
        if debug:
            print(f"[DEBUG] Strategy 1 failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Стратегия 2: Fix newlines - заменяем реальные \n на escaped \\n
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        # Простая замена - работает в большинстве случаев
        fixed = json_text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        result = json.loads(fixed)
        if debug:
            print("[DEBUG] Strategy 2 (fix newlines) succeeded")
        return result
    except json.JSONDecodeError as e:
        if debug:
            print(f"[DEBUG] Strategy 2 failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Стратегия 3: REGEX extraction - для сложных случаев с docstrings
    # ═══════════════════════════════════════════════════════════════════════════
    code = _extract_code_via_regex(json_text)
    if code:
        if debug:
            print("[DEBUG] Strategy 3 (regex extraction) succeeded")
        return {"name": "python", "arguments": {"code": code}}

    if debug:
        print("[DEBUG] All strategies failed")

    return None


def has_pending_tool_call(text: str) -> bool:
    """Check if there's an unresponded tool_call in text."""
    pos = text.rfind("<tool_call>")
    if pos == -1:
        return False
    return "</tool_response>" not in text[pos:]


def extract_final_code(text: str, entry_point: str) -> str | None:
    """Extract the final code from after </think> block."""
    think_end = text.find("</think>")
    if think_end == -1:
        return None

    after_think = text[think_end + len("</think>") :]

    # Try to find code block
    code_pattern = re.compile(r"```python\s*(.*?)```", re.DOTALL)
    match = code_pattern.search(after_think)
    if match:
        return match.group(1).strip()

    # Try to find function definition
    func_pattern = re.compile(
        rf"(def\s+{entry_point}\s*\(.*?\):.*?)(?:\n\n|\Z)", re.DOTALL
    )
    match = func_pattern.search(after_think)
    if match:
        return match.group(1).strip()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trace Generation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Trace:
    """A single generation trace."""

    problem_id: str
    problem: str
    messages: list[dict]
    full_response: str
    final_code: str | None
    score: float
    tests_passed: int
    tests_total: int
    think_count: int
    tool_calls: int
    tool_responses: int
    trace_segments: list[dict]
    had_errors: bool
    is_ideal: bool


async def generate_trace(
    client: OllamaClient,
    problem: dict,
    config: Config,
) -> Trace:
    """Generate a single trace for a problem."""

    # Build initial messages
    user_prompt = f"""Solve the following coding problem. Think step by step inside <think> tags, use the python tool to test your code, then provide the final solution after </think>.

Problem:
```python
{problem['prompt']}
```

Test cases:
{chr(10).join(f"- {problem['entry_point']}({t['input']}) == {t['expected']}" for t in problem['tests'])}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Track state - pre-fill with <think> to guide model
    assistant_content = "<think>\n"
    trace_segments = []
    tool_calls = 0
    tool_responses = 0
    had_errors = False
    turn = 0

    while turn < config.max_turns:
        turn += 1

        # Build messages with current assistant content
        current_messages = messages.copy()
        if assistant_content:
            current_messages.append({"role": "assistant", "content": assistant_content})

        if config.debug:
            print(f"\n[Turn {turn}] Generating...")

        # Generate - no stop sequences, let model complete naturally
        # We'll detect tool_call and think end in post-processing
        try:
            response = await client.chat_completion(
                current_messages,
                stop=None,  # Don't cut off, parse the full response
                max_tokens=config.max_tokens_per_turn,
            )
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            had_errors = True
            break

        if not response.strip():
            if config.debug:
                print("[Turn] Empty response, ending")
            break

        # Append to assistant content
        assistant_content += response

        if config.debug:
            print(f"[Turn {turn}] Response: {response[:200]}...")

        # Check if we hit a tool call (either complete or incomplete)
        if "<tool_call>" in response and "</tool_response>" not in response:
            # Check if we need to close the tag
            if "</tool_call>" not in assistant_content.split("<tool_call>")[-1]:
                assistant_content += "</tool_call>\n"

            # Parse and execute tool
            call_json = parse_tool_call(assistant_content, debug=config.debug)
            if call_json:
                tool_calls += 1

                if config.debug:
                    print(f"[Turn {turn}] Executing tool: {call_json.get('name')}")

                result = execute_tool(call_json, problem)
                tool_responses += 1

                if result.get("error"):
                    had_errors = True

                # Add tool response
                response_json = json.dumps(result)
                assistant_content += f"<tool_response>{response_json}</tool_response>\n"

                # Error Recovery: Add continuation hint after errors
                if result.get("error"):
                    error_msg = result["error"]
                    if len(error_msg) > 150:
                        error_msg = error_msg[:150] + "..."
                    assistant_content += f"\nI see there's an error: {error_msg}\nLet me analyze and fix the issue.\n\n"
                elif result.get("passed", 0) < result.get("total", 1):
                    # Partial success - encourage fixing
                    passed = result.get("passed", 0)
                    total = result.get("total", 1)
                    assistant_content += f"\nI got {passed}/{total} tests passing. Let me fix the failing cases.\n\n"

                trace_segments.append(
                    {
                        "type": "tool_call",
                        "content": json.dumps(call_json),
                    }
                )
                trace_segments.append(
                    {
                        "type": "tool_response",
                        "content": response_json,
                    }
                )

                if config.debug:
                    print(
                        f"[Turn {turn}] Tool result: passed={result.get('passed')}/{result.get('total')}"
                    )
            else:
                if config.debug:
                    print("[Turn] Failed to parse tool call")
                break

        # Check if </think> was reached
        elif "</think>" in response:
            # Complete the response
            assistant_content += "</think>\n\n"

            # Generate final code block
            try:
                final_response = await client.chat_completion(
                    messages + [{"role": "assistant", "content": assistant_content}],
                    max_tokens=512,
                )
                assistant_content += final_response
            except Exception as e:
                print(f"[ERROR] Final generation failed: {e}")

            break

        # Check if we got </think> pattern
        elif assistant_content.rstrip().endswith("</think>"):
            assistant_content += "\n\n"

            # Generate final code block
            try:
                final_response = await client.chat_completion(
                    messages + [{"role": "assistant", "content": assistant_content}],
                    max_tokens=512,
                )
                assistant_content += final_response
            except Exception as e:
                print(f"[ERROR] Final generation failed: {e}")

            break

    # Extract think segments
    think_pattern = re.compile(r"<think>(.*?)(?:</think>|<tool_call>)", re.DOTALL)
    think_matches = think_pattern.findall(assistant_content)
    think_count = len(think_matches)

    for i, content in enumerate(think_matches):
        trace_segments.insert(i * 2, {"type": "think", "content": content.strip()})

    # Extract final code
    final_code = extract_final_code(assistant_content, problem["entry_point"])

    # If no final code, try to extract from last tool call
    if not final_code:
        last_call = parse_tool_call(assistant_content, debug=config.debug)
        if last_call and last_call.get("arguments", {}).get("code"):
            final_code = last_call["arguments"]["code"]

    # Score the final code
    score = 0.0
    tests_passed = 0
    tests_total = len(problem["tests"])

    if final_code:
        # Prepare test cases for execute_code_safe
        test_inputs = []
        test_outputs = []
        for test in problem["tests"]:
            try:
                input_val = eval(test["input"])
                expected_val = eval(test["expected"])
                if isinstance(input_val, tuple):
                    test_inputs.append(list(input_val))
                else:
                    test_inputs.append([input_val])
                test_outputs.append(expected_val)
            except Exception:
                pass

        test_cases = {
            "fn_name": problem["entry_point"],
            "inputs": test_inputs,
            "outputs": test_outputs,
        }

        results, metadata = execute_code_safe(final_code, test_cases, timeout=5.0)
        tests_passed = sum(1 for r in results if r)

        if tests_passed == tests_total:
            score = 1.0
        else:
            score = -1.0 + 2.0 * (tests_passed / max(tests_total, 1))

    # Determine if trace is "ideal"
    is_ideal = (
        score >= 1.0
        and think_count >= 2
        and tool_calls >= 1
        and tool_responses >= 1
        and not had_errors
    )

    return Trace(
        problem_id=problem["id"],
        problem=problem["prompt"],
        messages=messages + [{"role": "assistant", "content": assistant_content}],
        full_response=assistant_content,
        final_code=final_code,
        score=score,
        tests_passed=tests_passed,
        tests_total=tests_total,
        think_count=think_count,
        tool_calls=tool_calls,
        tool_responses=tool_responses,
        trace_segments=trace_segments,
        had_errors=had_errors,
        is_ideal=is_ideal,
    )


def trace_to_dict(trace: Trace, training_format: bool = False) -> dict:
    """Convert trace to dictionary for JSONL output."""
    if training_format:
        # Single assistant message with full trace
        return {
            "messages": trace.messages,
            "score": trace.score,
        }

    return {
        "problem_id": trace.problem_id,
        "problem": trace.problem,
        "messages": trace.messages,
        "full_response": trace.full_response,
        "code": trace.final_code,
        "score": trace.score,
        "tests_passed": trace.tests_passed,
        "tests_total": trace.tests_total,
        "think_count": trace.think_count,
        "tool_calls": trace.tool_calls,
        "tool_responses": trace.tool_responses,
        "trace": trace.trace_segments,
        "had_errors": trace.had_errors,
        "is_ideal": trace.is_ideal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Generate inline tool traces")
    parser.add_argument(
        "--output", "-o", default="traces_inline.jsonl", help="Output JSONL file"
    )
    parser.add_argument(
        "--num-traces", "-n", type=int, default=10, help="Number of traces to generate"
    )
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Only save successful traces (score > 0)",
    )
    parser.add_argument(
        "--only-ideal",
        action="store_true",
        help="Only save ideal traces (interleaved + tools + success)",
    )
    parser.add_argument(
        "--training-format",
        action="store_true",
        help="Output in training format (messages + score only)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="Model name (default: deepseek-v3.2:cloud)"
    )
    args = parser.parse_args()

    config = Config(
        debug=True,  # Always enable debug for now
        temperature=args.temperature,
    )

    # Override model if specified
    if args.model:
        config.model = args.model

    print(f"Loading problems...")
    problems = get_problems()
    print(f"Loaded {len(problems)} problems")

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Max turns: {config.max_turns}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Output: {args.output}")

    traces_saved = 0
    traces_generated = 0

    async with OllamaClient(config) as client:
        with open(args.output, "w") as f:
            for i in range(args.num_traces):
                problem = problems[i % len(problems)]

                print(f"\n{'='*60}")
                print(f"[{i+1}/{args.num_traces}] Problem: {problem['id']}")
                print(f"{'='*60}")

                try:
                    trace = await generate_trace(client, problem, config)
                    traces_generated += 1

                    # Print summary
                    print(f"\nResult:")
                    print(f"  Score: {trace.score:.2f}")
                    print(f"  Tests: {trace.tests_passed}/{trace.tests_total}")
                    print(f"  Think blocks: {trace.think_count}")
                    print(f"  Tool calls: {trace.tool_calls}")
                    print(f"  Is ideal: {trace.is_ideal}")

                    # Filter based on options
                    if args.only_ideal and not trace.is_ideal:
                        print(f"  [SKIP] Not ideal")
                        continue

                    if args.only_success and trace.score <= 0:
                        print(f"  [SKIP] Not successful")
                        continue

                    # Save trace
                    trace_dict = trace_to_dict(trace, args.training_format)
                    f.write(json.dumps(trace_dict) + "\n")
                    f.flush()
                    traces_saved += 1
                    print(f"  [SAVED]")

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    import traceback

                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Generated: {traces_generated}")
    print(f"  Saved: {traces_saved}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
