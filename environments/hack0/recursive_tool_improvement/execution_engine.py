"""
Execution Engine for Recursive Tool Improvement Environment

This module provides a sandboxed execution environment for safely running
tool compositions created by language models. It handles parsing of tool
compositions, executing them with proper safety constraints, and capturing
execution results.
"""

import ast
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import multiprocessing
import signal
import contextlib
import io
import builtins

from environments.hack0.recursive_tool_improvement.tool_registry import Tool, ToolRegistry, default_registry


class TimeoutError(Exception):
    """Exception raised when execution exceeds the timeout limit."""
    pass


class MemoryLimitError(Exception):
    """Exception raised when execution exceeds the memory limit."""
    pass


class RestrictedImportError(Exception):
    """Exception raised when an attempt is made to import a restricted module."""
    pass


class FunctionCallLimitError(Exception):
    """Exception raised when execution exceeds the function call limit."""
    pass


class ExecutionResult:
    """
    Represents the result of executing a tool composition.

    Attributes:
        success: Whether the execution completed successfully
        result: The value returned by the composition (if successful)
        error: Error message (if not successful)
        execution_time: Time taken for execution (in seconds)
        tool_calls: List of tools called during execution
        stdout: Captured standard output during execution
        stderr: Captured standard error during execution
    """

    def __init__(
        self,
        success: bool = False,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        stdout: str = "",
        stderr: str = "",
    ):
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.tool_calls = tool_calls or []
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution result to a dictionary representation"""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_calls": self.tool_calls,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    def __str__(self) -> str:
        """String representation of the execution result"""
        if self.success:
            return f"Success: {str(self.result)} (Time: {self.execution_time:.3f}s, Tool calls: {len(self.tool_calls)})"
        else:
            return f"Error: {self.error} (Time: {self.execution_time:.3f}s)"


class CodeParser:
    """
    Parses and extracts tool compositions from LLM outputs.

    This class handles the extraction and validation of function definitions
    from the model's response.
    """

    @staticmethod
    def extract_function(text: str, fn_name: str = "solve") -> Optional[str]:
        """
        Extract the function definition from text, looking for a specific function name.

        Args:
            text: The text containing the function definition
            fn_name: The name of the function to extract (default: "solve")

        Returns:
            The extracted function code as a string, or None if not found
        """
        # Look for function inside <composition> tags first
        composition_pattern = r"<composition>(.*?)</composition>"
        composition_match = re.search(composition_pattern, text, re.DOTALL)

        if composition_match:
            code = composition_match.group(1).strip()
            # Check if this is a valid function definition
            try:
                parsed = ast.parse(code)
                for node in ast.walk(parsed):
                    if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                        return code
            except SyntaxError:
                pass

        # If not found or invalid, look for any function definition with the specified name
        function_pattern = rf"def\s+{fn_name}\s*\(.*?\).*?:"
        match = re.search(function_pattern, text, re.DOTALL)

        if not match:
            return None

        # Get the starting position of the function definition
        start_pos = match.start()

        # Extract the function body (handling indentation)
        lines = text[start_pos:].split('\n')
        function_lines = [lines[0]]

        # Find the indentation of the first line of the function body
        for i in range(1, len(lines)):
            if re.match(r'^\s*$', lines[i]):  # Skip empty lines
                function_lines.append(lines[i])
                continue

            if i == 1:  # First non-empty line after function definition
                indent_match = re.match(r'^(\s+)', lines[i])
                if not indent_match:  # No indentation, not a valid function
                    return None
                base_indent = indent_match.group(1)
                function_lines.append(lines[i])
            elif lines[i].startswith(base_indent):  # Line has base indentation
                function_lines.append(lines[i])
            elif not re.match(r'^\s', lines[i]):  # New unindented line - end of function
                break
            else:  # Line has different indentation - likely part of nested block
                indent_match = re.match(r'^(\s+)', lines[i])
                if indent_match and len(indent_match.group(1)) > len(base_indent):
                    function_lines.append(lines[i])
                else:
                    break

        return '\n'.join(function_lines)

    @staticmethod
    def is_valid_function(code: str, fn_name: str = "solve") -> bool:
        """
        Check if the extracted code is a valid function with the expected name.

        Args:
            code: The code to validate
            fn_name: The expected function name

        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = ast.parse(code)

            # Check if there is exactly one function definition with the correct name
            functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]

            if len(functions) != 1:
                return False

            return functions[0].name == fn_name

        except SyntaxError:
            return False


class ExecutionEngine:
    """
    Safely executes tool compositions with appropriate security constraints.

    This class provides controlled execution of code generated by language models,
    enforcing timeouts, memory limits, and access restrictions.
    """

    def __init__(
        self,
        registry: ToolRegistry = default_registry,
        timeout: int = 5,
        memory_limit: int = 100 * 1024 * 1024,  # 100 MB
        max_function_calls: int = 100,
        restricted_modules: Optional[Set[str]] = None,
        allowed_builtins: Optional[Set[str]] = None,
    ):
        """
        Initialize the ExecutionEngine with security constraints.

        Args:
            registry: The tool registry containing available tools
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory usage in bytes
            max_function_calls: Maximum number of function calls allowed
            restricted_modules: Set of module names that are not allowed to be imported
            allowed_builtins: Set of Python built-in functions that are allowed
        """
        self.registry = registry
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.max_function_calls = max_function_calls

        # Default restricted modules if none provided
        self.restricted_modules = restricted_modules or {
            'os', 'subprocess', 'sys', 'socket', 'requests', 'urllib',
            'http', 'ftplib', 'telnetlib', 'smtplib', 'popen2', 'multiprocessing',
            'ctypes', 'importlib', 'marshal'
        }

        # Default allowed builtins if none provided
        self.allowed_builtins = allowed_builtins or {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytes', 'chr', 'complex',
            'dict', 'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
            'frozenset', 'hash', 'hex', 'int', 'isinstance', 'issubclass', 'iter',
            'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow',
            'print', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
            'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
        }

        self.function_call_count = 0
        self.start_time = 0

    def _secure_exec(self, code: str, input_data: Any = None) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute the given code in a secure environment with constraints.

        Args:
            code: Python code to execute
            input_data: Data to pass to the function as input

        Returns:
            Tuple containing:
                - Success flag (True/False)
                - Result of execution (or None if error)
                - Error message (or None if successful)
        """
        self.function_call_count = 0
        self.start_time = time.time()

        # Capture original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create string buffers for stdout/stderr capture
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Swap stdout and stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        # Define the secure execution environment
        exec_globals = {
            "__builtins__": {name: getattr(builtins, name) for name in self.allowed_builtins},
        }

        # Add registry tools to the execution environment
        for tool_name, tool in self.registry.tools.items():
            exec_globals[tool_name] = self._wrap_tool_for_tracking(tool)

        # Create a secure import function that filters out restricted modules
        def secure_import(name, *args, **kwargs):
            if name in self.restricted_modules:
                raise RestrictedImportError(f"Import of module '{name}' is not allowed")
            return __import__(name, *args, **kwargs)

        exec_globals["__import__"] = secure_import

        # Function to check timeout
        def check_timeout():
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError(f"Execution timed out after {self.timeout} seconds")

        # Add the timeout check to exec_globals
        exec_globals["check_timeout"] = check_timeout

        # Track tool calls
        self.tool_calls = []

        try:
            # First compile the code to catch syntax errors
            compiled_code = compile(code, "<string>", "exec")

            # Execute the code
            exec(compiled_code, exec_globals)

            # Extract the solve function
            solve_function = exec_globals.get("solve")
            if not solve_function:
                return False, None, "No 'solve' function found in the code"

            # Call the solve function with the provided input data
            result = solve_function(input_data)
            return True, result, None

        except TimeoutError as e:
            return False, None, str(e)
        except MemoryLimitError as e:
            return False, None, str(e)
        except RestrictedImportError as e:
            return False, None, str(e)
        except FunctionCallLimitError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Get captured output
            stdout_value = stdout_buffer.getvalue()
            stderr_value = stderr_buffer.getvalue()

            # Close the buffers
            stdout_buffer.close()
            stderr_buffer.close()

    def _wrap_tool_for_tracking(self, tool: Tool) -> callable:
        """
        Wrap a tool function to track calls and enforce limits.

        Args:
            tool: The Tool object to wrap

        Returns:
            A wrapped function that tracks calls and enforces limits
        """
        def wrapped_tool(*args, **kwargs):
            # Check if execution time limit exceeded
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError(f"Execution timed out after {self.timeout} seconds")

            # Check if function call limit exceeded
            self.function_call_count += 1
            if self.function_call_count > self.max_function_calls:
                raise FunctionCallLimitError(f"Maximum function call limit ({self.max_function_calls}) exceeded")

            # Track this tool call
            self.tool_calls.append({
                "tool": tool.name,
                "args": args,
                "kwargs": kwargs,
                "timestamp": time.time() - self.start_time
            })

            # Execute the actual tool function
            return tool.function(*args, **kwargs)

        return wrapped_tool

    def _process_executor(self, code: str, input_data: Any) -> Tuple[bool, Any, Optional[str], List[Dict[str, Any]], str, str]:
        """
        Execute code in a separate process for isolation.

        Args:
            code: Python code to execute
            input_data: Data to pass to the function

        Returns:
            Tuple containing execution results and metadata
        """
        # Create string buffers for stdout/stderr capture
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            success, result, error = self._secure_exec(code, input_data)

        return (
            success,
            result,
            error,
            self.tool_calls,
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue()
        )

    def execute(self, code: str, input_data: Any = None) -> ExecutionResult:
        """
        Execute a tool composition safely in the current process.

        Simplified implementation that doesn't use multiprocessing to avoid
        pickling errors and other complexities.

        Args:
            code: The Python code of the tool composition
            input_data: The input data to pass to the composition

        Returns:
            ExecutionResult object containing the results and metadata
        """
        if not code or not code.strip():
            return ExecutionResult(
                success=False,
                error="Empty code provided for execution",
                execution_time=0.0
            )

        # Validate input code for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=f"Syntax error in code: {str(e)}",
                execution_time=0.0,
                stderr=str(e)
            )

        start_time = time.time()

        # Create string buffers for stdout/stderr capture
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Redirect stdout and stderr
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Execute the code directly
                success, result, error = self._secure_exec(code, input_data)
                execution_time = time.time() - start_time

                return ExecutionResult(
                    success=success,
                    result=result,
                    error=error,
                    execution_time=execution_time,
                    tool_calls=self.tool_calls,
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue()
                )
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Error during execution: {type(e).__name__}: {str(e)}"
                return ExecutionResult(
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    stderr=f"{error_msg}\n{traceback.format_exc()}"
                )

    def execute_from_text(self, text: str, input_data: Any = None, fn_name: str = "solve") -> ExecutionResult:
        """
        Extract a function from text and execute it.

        This method handles the extraction of code from unstructured text (like LLM responses),
        validates it, and then executes it safely.

        Args:
            text: The text containing the function definition
            input_data: The input data to pass to the function
            fn_name: The name of the function to extract (default: "solve")

        Returns:
            ExecutionResult object containing the results and metadata
        """
        if not text:
            return ExecutionResult(
                success=False,
                error="Empty text provided for function extraction",
                execution_time=0.0
            )

        # Try to extract the function code
        try:
            code = CodeParser.extract_function(text, fn_name)
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Error extracting function from text: {str(e)}",
                execution_time=0.0,
                stderr=f"Function extraction error: {str(e)}\n{traceback.format_exc()}"
            )

        if not code:
            # Provide a more detailed diagnostic when no function is found
            if "<composition>" in text and "</composition>" in text:
                composition_content = re.search(r"<composition>(.*?)</composition>", text, re.DOTALL)
                if composition_content:
                    content = composition_content.group(1).strip()
                    if not content:
                        error_message = f"Empty <composition> tags found, but no valid '{fn_name}' function detected"
                    else:
                        error_message = f"<composition> tags found, but they don't contain a valid '{fn_name}' function"
                else:
                    error_message = f"<composition> tags found, but couldn't extract content between them"
            elif f"def {fn_name}" in text:
                error_message = f"Found 'def {fn_name}', but couldn't extract a complete function definition"
            else:
                error_message = f"No '{fn_name}' function definition found in the text"

            return ExecutionResult(
                success=False,
                error=error_message,
                execution_time=0.0,
                stderr=f"ERROR: {error_message}\n\nText snippet: {text[:200]}..."
            )

        # Check if it's a valid function
        if not CodeParser.is_valid_function(code, fn_name):
            return ExecutionResult(
                success=False,
                error=f"The extracted code is not a valid '{fn_name}' function",
                execution_time=0.0
            )

        # Execute the function
        return self.execute(code, input_data)