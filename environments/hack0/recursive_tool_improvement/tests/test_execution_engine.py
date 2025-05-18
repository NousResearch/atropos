"""
Unit tests for the execution engine component.
"""

import unittest
from ..execution_engine import CodeParser, ExecutionEngine, ExecutionResult
from ..tool_registry import ToolRegistry


class TestCodeParser(unittest.TestCase):
    """Tests for the CodeParser class."""
    
    def test_extract_function_from_composition_tags(self):
        """Test extracting a function from composition tags."""
        text = """
        <composition>
        def solve(input_data):
            return input_data.upper()
        </composition>
        """
        
        code = CodeParser.extract_function(text)
        self.assertIsNotNone(code)
        self.assertIn("def solve(input_data):", code)
        self.assertIn("return input_data.upper()", code)
    
    def test_extract_function_without_tags(self):
        """Test extracting a function without tags."""
        text = """
        Here's my solution:
        
        def solve(input_data):
            return input_data.upper()
            
        I think this works well.
        """
        
        code = CodeParser.extract_function(text)
        self.assertIsNotNone(code)
        self.assertIn("def solve(input_data):", code)
        self.assertIn("return input_data.upper()", code)
    
    def test_extract_function_with_indentation(self):
        """Test extracting a function with proper indentation handling."""
        text = """
        def solve(input_data):
            # Convert to uppercase
            result = input_data.upper()
            
            # Split by spaces
            words = result.split(' ')
            
            # Join with dashes
            return '-'.join(words)
        """
        
        code = CodeParser.extract_function(text)
        self.assertIsNotNone(code)
        self.assertEqual(code.strip(), text.strip())
    
    def test_extract_non_existent_function(self):
        """Test trying to extract a function that doesn't exist."""
        text = "There is no function here, just some text."
        
        code = CodeParser.extract_function(text)
        self.assertIsNone(code)
    
    def test_is_valid_function(self):
        """Test function validation."""
        # Valid function
        valid_code = "def solve(input_data):\n    return input_data.upper()"
        self.assertTrue(CodeParser.is_valid_function(valid_code))
        
        # Invalid function (syntax error)
        invalid_code = "def solve(input_data):\n    return input_data.upper("
        self.assertFalse(CodeParser.is_valid_function(invalid_code))
        
        # Not a function
        not_a_function = "x = 5"
        self.assertFalse(CodeParser.is_valid_function(not_a_function))
        
        # Wrong function name
        wrong_name = "def process(input_data):\n    return input_data.upper()"
        self.assertFalse(CodeParser.is_valid_function(wrong_name))


class TestExecutionEngine(unittest.TestCase):
    """Tests for the ExecutionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test registry with a simple tool
        self.registry = ToolRegistry()
        
        # Add a simple uppercase tool to the registry
        from ..tool_registry import Tool
        
        def to_uppercase(text):
            return text.upper()
        
        self.registry.register(Tool(
            name="to_uppercase",
            description="Convert text to uppercase",
            parameters={
                "text": {"type": "string", "description": "Text to convert"}
            },
            function=to_uppercase
        ))
        
        # Create the execution engine with a short timeout
        self.engine = ExecutionEngine(
            registry=self.registry,
            timeout=2,
            max_function_calls=10
        )
    
    def test_execute_simple_code(self):
        """Test executing simple code."""
        code = """
        def solve(input_data):
            return input_data * 2
        """
        
        result = self.engine.execute(code, "test")
        
        self.assertTrue(result.success)
        self.assertEqual(result.result, "testtest")
        self.assertIsNone(result.error)
    
    def test_execute_with_tool(self):
        """Test executing code that uses a registered tool."""
        code = """
        def solve(input_data):
            return to_uppercase(input_data)
        """
        
        result = self.engine.execute(code, "test")
        
        self.assertTrue(result.success)
        self.assertEqual(result.result, "TEST")
        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["tool"], "to_uppercase")
    
    def test_execute_syntax_error(self):
        """Test executing code with syntax errors."""
        code = """
        def solve(input_data)
            return input_data
        """
        
        result = self.engine.execute(code, "test")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("Syntax error", result.error)
    
    def test_execute_runtime_error(self):
        """Test executing code with runtime errors."""
        code = """
        def solve(input_data):
            return input_data / 0
        """
        
        result = self.engine.execute(code, "test")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("ZeroDivisionError", result.error)
    
    def test_execute_timeout(self):
        """Test executing code that exceeds the timeout."""
        code = """
        def solve(input_data):
            import time
            time.sleep(5)  # This should exceed our 2-second timeout
            return input_data
        """
        
        result = self.engine.execute(code, "test")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("timed out", result.error.lower())
    
    def test_execution_result_formatting(self):
        """Test formatting of ExecutionResult."""
        result = ExecutionResult(
            success=True,
            result="TEST",
            execution_time=1.5,
            tool_calls=[{"tool": "to_uppercase", "args": ["test"], "kwargs": {}}]
        )
        
        # Test to_dict method
        result_dict = result.to_dict()
        self.assertEqual(result_dict["success"], True)
        self.assertEqual(result_dict["result"], "TEST")
        self.assertEqual(result_dict["execution_time"], 1.5)
        self.assertEqual(len(result_dict["tool_calls"]), 1)
        
        # Test string representation
        result_str = str(result)
        self.assertIn("Success", result_str)
        self.assertIn("TEST", result_str)
        self.assertIn("1.5", result_str)
    
    def test_execute_from_text(self):
        """Test extracting and executing a function from text."""
        text = """
        I think this solution should work:
        
        <composition>
        def solve(input_data):
            return to_uppercase(input_data)
        </composition>
        """
        
        result = self.engine.execute_from_text(text, "test")
        
        self.assertTrue(result.success)
        self.assertEqual(result.result, "TEST")
        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 1)


if __name__ == "__main__":
    unittest.main()