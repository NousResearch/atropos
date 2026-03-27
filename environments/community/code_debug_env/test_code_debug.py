"""
Unit tests for the Code Debug environment.

Tests the code execution, scoring, and extraction utilities
without requiring a running inference server.
"""

import pytest
from code_executor import (
    count_test_results,
    execute_code_with_tests,
    extract_boxed_code,
)

# ============================================================================
# Tests for extract_boxed_code
# ============================================================================


class TestExtractBoxedCode:
    def test_simple_extraction(self):
        text = r"""Here is the fix:
\boxed{def add(a, b):
    return a + b
}"""
        result = extract_boxed_code(text)
        assert result is not None
        assert "def add(a, b):" in result
        assert "return a + b" in result

    def test_nested_braces(self):
        text = r"""\boxed{def foo(x):
    if x > 0:
        return {x: x**2}
    return {}
}"""
        result = extract_boxed_code(text)
        assert result is not None
        assert "return {x: x**2}" in result
        assert "return {}" in result

    def test_no_boxed(self):
        text = "Just some text without any boxed content"
        result = extract_boxed_code(text)
        assert result is None

    def test_last_boxed_used(self):
        text = r"""First attempt: \boxed{def bad(): pass}
Actually, let me reconsider: \boxed{def good():
    return 42
}"""
        result = extract_boxed_code(text)
        assert result is not None
        assert "def good():" in result
        assert "return 42" in result

    def test_empty_boxed(self):
        text = r"\boxed{}"
        result = extract_boxed_code(text)
        assert result == ""


# ============================================================================
# Tests for execute_code_with_tests
# ============================================================================


class TestExecuteCodeWithTests:
    def test_correct_code_passes(self):
        code = "def add(a, b):\n    return a + b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
    assert candidate(0, 0) == 0
"""
        passed, error = execute_code_with_tests(code, test_code, "add")
        assert passed is True
        assert error == ""

    def test_buggy_code_fails(self):
        code = "def add(a, b):\n    return a - b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
"""
        passed, error = execute_code_with_tests(code, test_code, "add")
        assert passed is False
        assert error != ""

    def test_syntax_error(self):
        code = "def add(a, b)\n    return a + b\n"  # Missing colon
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
"""
        passed, error = execute_code_with_tests(code, test_code, "add")
        assert passed is False

    def test_infinite_loop_timeout(self):
        code = "def loop(x):\n    while True: pass\n"
        test_code = """def check(candidate):
    assert candidate(1) is None
"""
        passed, error = execute_code_with_tests(code, test_code, "loop", timeout=2)
        assert passed is False
        assert "timed out" in error.lower() or "Timeout" in error

    def test_runtime_error(self):
        code = "def divide(a, b):\n    return a / b\n"
        test_code = """def check(candidate):
    assert candidate(1, 0) == 0
"""
        passed, error = execute_code_with_tests(code, test_code, "divide")
        assert passed is False


# ============================================================================
# Tests for count_test_results
# ============================================================================


class TestCountTestResults:
    def test_all_pass(self):
        code = "def add(a, b):\n    return a + b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
"""
        passed, total = count_test_results(code, test_code, "add")
        assert passed > 0
        assert passed == total

    def test_none_pass(self):
        code = "def add(a, b):\n    return 0\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(5, 5) == 10
"""
        passed, total = count_test_results(code, test_code, "add")
        assert passed == 0

    def test_syntax_error_returns_zero(self):
        code = "def add(a, b)\n    return a + b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
"""
        passed, total = count_test_results(code, test_code, "add")
        assert passed == 0


# ============================================================================
# Tests for scoring logic
# ============================================================================


class TestScoringLogic:
    """Test the scoring logic that will be used in CodeDebugEnv._score_fix."""

    def test_perfect_fix_scores_one(self):
        """A fix that passes all tests should score 1.0."""
        code = "def add(a, b):\n    return a + b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
"""
        passed, error = execute_code_with_tests(code, test_code, "add")
        assert passed is True
        # Score would be 1.0

    def test_no_fix_scores_negative(self):
        """A fix that doesn't improve should score negatively."""
        buggy = "def add(a, b):\n    return a - b\n"
        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
"""
        passed, error = execute_code_with_tests(buggy, test_code, "add")
        assert passed is False
        # Score would be -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
