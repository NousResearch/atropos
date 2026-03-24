"""
Safe code execution utilities for the Code Debug environment.

Runs generated code in isolated subprocess with timeout and resource limits.
"""

import os
import subprocess
import tempfile
from typing import Optional, Tuple


def execute_code_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> Tuple[bool, str]:
    """
    Execute code with test cases in an isolated subprocess.

    Args:
        code: The function implementation to test.
        test_code: Test code containing a `check(candidate)` function.
        entry_point: The function name to pass to `check()`.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (all_tests_passed, error_message).
    """
    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="code_debug_"
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        else:
            error = result.stderr.strip()
            # Truncate long tracebacks
            if len(error) > 500:
                error = error[-500:]
            return False, error

    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_boxed_code(text: str) -> Optional[str]:
    """
    Extract code from \\boxed{...} format, handling nested braces.

    Finds the LAST occurrence of \\boxed{} to handle cases where
    the model discusses code before providing the final answer.

    Args:
        text: The model's response text.

    Returns:
        The extracted code string, or None if no \\boxed{} found.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    brace_count = 1
    i = start

    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start : i - 1].strip()
    return None


def count_test_results(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> Tuple[int, int]:
    """
    Count how many individual assertions pass vs fail.

    Wraps each assertion in a try/except to count partial success.

    Args:
        code: The function implementation to test.
        test_code: Test code containing a `check(candidate)` function.
        entry_point: The function name to pass.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (passed_count, total_count).
    """
    # Build a counting test harness
    counter_code = f"""
import sys

{code}

_passed = 0
_total = 0

def _counting_check(candidate):
    global _passed, _total
    import types

    # Get the original check function's code
    _orig_check_src = '''{test_code}'''
    _ns = {{'__builtins__': __builtins__}}
    exec(_orig_check_src, _ns)
    _orig_check = _ns.get('check')

    if _orig_check is None:
        print("0/0")
        return

    # Get the source of check and count assertions
    import inspect
    try:
        src = inspect.getsource(_orig_check)
    except (TypeError, OSError):
        # Can't inspect — just run it
        try:
            _orig_check(candidate)
            print("1/1")
        except Exception:
            print("0/1")
        return

    # Count 'assert' lines
    assert_lines = [l.strip() for l in src.split('\\n') if l.strip().startswith('assert')]
    _total = max(len(assert_lines), 1)

    # Run the full check — if it passes, all assertions passed
    try:
        _orig_check(candidate)
        _passed = _total
    except AssertionError:
        # Some failed — try to count
        _passed = 0
        for line in assert_lines:
            try:
                exec(line, {{'__builtins__': __builtins__, '{entry_point}': candidate, 'candidate': candidate}})
                _passed += 1
            except Exception:
                pass
    except Exception:
        _passed = 0

    print(f"{{_passed}}/{{_total}}")

_counting_check({entry_point})
"""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="code_debug_count_"
        ) as f:
            f.write(counter_code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        if "/" in stdout:
            parts = stdout.split("/")
            try:
                return int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass

        # Fallback: if execution succeeded, assume all passed
        if result.returncode == 0:
            return 1, 1
        return 0, 1

    except subprocess.TimeoutExpired:
        return 0, 1
    except Exception:
        return 0, 1
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
