"""
Local Python code executor with sandboxing.

Executes generated code locally with timeout and safety restrictions.
Used when Modal is not available.
"""

import ast
import io
import multiprocessing
import signal
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class ExecutionResult:
    """Result of code execution."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        return_value: Any = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.return_value = return_value

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_value": repr(self.return_value) if self.return_value else None,
        }


def _execute_code_worker(code: str, test_inputs: List, fn_name: str, queue: multiprocessing.Queue):
    """Worker function that runs in a separate process."""
    try:
        # Restrict builtins for safety
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
            'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
            'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
            'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
            'ord': ord, 'pow': pow, 'print': print, 'range': range, 'repr': repr,
            'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
            'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
            'zip': zip, 'True': True, 'False': False, 'None': None,
            '__import__': __import__,  # Needed for some imports
        }

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        exec_globals = {"__builtins__": safe_builtins}

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        # Get the function
        if fn_name and fn_name != "none" and fn_name in exec_globals:
            func = exec_globals[fn_name]
            results = []
            for test_input in test_inputs:
                if isinstance(test_input, (list, tuple)):
                    result = func(*test_input)
                else:
                    result = func(test_input)
                results.append(result)
            queue.put(("success", results, stdout_capture.getvalue()))
        else:
            # For stdin/stdout problems, just check if it runs
            queue.put(("success", [], stdout_capture.getvalue()))

    except Exception as e:
        queue.put(("error", str(e), traceback.format_exc()))


def execute_code_safe(
    code: str,
    test_cases: Optional[Dict] = None,
    timeout: float = 10.0,
) -> Tuple[List[bool], Dict]:
    """
    Execute code safely in a separate process with timeout.

    Args:
        code: Python code to execute
        test_cases: Dict with 'inputs', 'outputs', and optional 'fn_name'
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (list of test results, metadata dict)
    """
    if code is None:
        return [False], {"error": "No code provided"}

    # Parse test cases
    if test_cases is None:
        test_cases = {}

    inputs = test_cases.get("inputs", test_cases.get("input", []))
    expected_outputs = test_cases.get("outputs", test_cases.get("output", []))
    fn_name = test_cases.get("fn_name", "none")

    # Handle string inputs (JSON)
    if isinstance(inputs, str):
        import json
        inputs = json.loads(inputs)
    if isinstance(expected_outputs, str):
        import json
        expected_outputs = json.loads(expected_outputs)

    # Create queue for results
    queue = multiprocessing.Queue()

    # Start worker process
    process = multiprocessing.Process(
        target=_execute_code_worker,
        args=(code, inputs, fn_name, queue),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return [False], {"error": "Timeout", "timeout": timeout}

    # Get results from queue
    try:
        status, data, extra = queue.get_nowait()
    except:
        return [False], {"error": "No result from worker"}

    if status == "error":
        return [False], {"error": data, "traceback": extra}

    # Compare results with expected outputs
    results = data
    output = extra

    if not expected_outputs:
        # No expected outputs, just check if it ran
        return [True], {"output": output}

    test_results = []
    for i, (result, expected) in enumerate(zip(results, expected_outputs)):
        passed = result == expected
        test_results.append(passed)

    # Pad with False if not enough results
    while len(test_results) < len(expected_outputs):
        test_results.append(False)

    return test_results, {
        "output": output,
        "results": results,
        "expected": expected_outputs,
        "passed": sum(test_results),
        "total": len(test_results),
    }


def execute_function_tests(
    code: str,
    fn_name: str,
    test_inputs: List,
    expected_outputs: List,
    timeout: float = 10.0,
) -> Tuple[float, Dict]:
    """
    Execute function-based tests and return a score.

    Returns:
        Tuple of (score from 0.0 to 1.0, metadata dict)
    """
    test_cases = {
        "inputs": test_inputs,
        "outputs": expected_outputs,
        "fn_name": fn_name,
    }

    results, metadata = execute_code_safe(code, test_cases, timeout)

    if not results or all(not r for r in results):
        return -1.0, metadata

    # Calculate score based on passed tests
    passed = sum(1 for r in results if r)
    total = len(results)

    if passed == total:
        return 1.0, metadata
    else:
        # Partial credit
        return -1.0 + (2.0 * passed / total), metadata


class LocalCodeExecutor:
    """
    Local code executor for the agent trace pipeline.

    Drop-in replacement for Modal-based execution.
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    async def run_test(
        self, test_cases: Dict, code: str
    ) -> Tuple[List[bool], Dict]:
        """
        Run tests on code (async interface for compatibility).

        Args:
            test_cases: Dict with 'tests' key containing test data
            code: Python code to execute

        Returns:
            Tuple of (list of bool results, metadata dict)
        """
        import asyncio

        tests = test_cases.get("tests", test_cases)

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        results, metadata = await loop.run_in_executor(
            None,
            execute_code_safe,
            code,
            tests,
            self.timeout,
        )

        return results, metadata


# Global executor instance
_executor = LocalCodeExecutor()


async def run_test_local(test_cases: Dict, code: str) -> Tuple[List[bool], Dict]:
    """
    Async function to run tests locally.

    Compatible interface with Modal's run_test.remote.aio()
    """
    return await _executor.run_test(test_cases, code)


if __name__ == "__main__":
    # Test the executor
    import asyncio

    test_code = '''
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
'''

    test_cases = {
        "fn_name": "two_sum",
        "inputs": [
            [[2, 7, 11, 15], 9],
            [[3, 2, 4], 6],
            [[3, 3], 6],
        ],
        "outputs": [
            [0, 1],
            [1, 2],
            [0, 1],
        ],
    }

    async def test():
        results, metadata = await run_test_local({"tests": test_cases}, test_code)
        print(f"Results: {results}")
        print(f"Metadata: {metadata}")
        print(f"All passed: {all(results)}")

    asyncio.run(test())
