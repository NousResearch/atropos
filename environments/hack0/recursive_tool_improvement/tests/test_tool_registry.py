"""
Unit tests for the tool registry component.
"""

import unittest
from ..tool_registry import Tool, ToolRegistry, default_registry


class TestToolRegistry(unittest.TestCase):
    """Tests for the Tool and ToolRegistry classes."""
    
    def test_tool_creation(self):
        """Test creating a Tool object."""
        # Define a simple test function
        def add(a, b):
            return a + b
        
        # Create a tool
        tool = Tool(
            name="add",
            description="Add two numbers together",
            parameters={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            function=add,
            examples=["add(1, 2)"]
        )
        
        # Test attributes
        self.assertEqual(tool.name, "add")
        self.assertEqual(tool.description, "Add two numbers together")
        self.assertEqual(len(tool.parameters), 2)
        self.assertEqual(tool.examples, ["add(1, 2)"])
        
        # Test calling the tool
        self.assertEqual(tool(1, 2), 3)
        
        # Test to_dict method
        tool_dict = tool.to_dict()
        self.assertEqual(tool_dict["name"], "add")
        self.assertEqual(tool_dict["description"], "Add two numbers together")
        self.assertEqual(len(tool_dict["parameters"]), 2)
        self.assertEqual(tool_dict["examples"], ["add(1, 2)"])
        self.assertNotIn("function", tool_dict)  # function should not be included
    
    def test_registry_operations(self):
        """Test ToolRegistry operations."""
        # Create a registry
        registry = ToolRegistry()
        
        # Define a couple of test functions
        def add(a, b):
            return a + b
        
        def multiply(a, b):
            return a * b
        
        # Create tools
        add_tool = Tool(
            name="add",
            description="Add two numbers together",
            parameters={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            function=add
        )
        
        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers together",
            parameters={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            function=multiply
        )
        
        # Register tools
        registry.register(add_tool)
        registry.register(multiply_tool)
        
        # Test registry size
        self.assertEqual(len(registry.tools), 2)
        
        # Test get_tool
        self.assertEqual(registry.get_tool("add"), add_tool)
        self.assertEqual(registry.get_tool("multiply"), multiply_tool)
        self.assertIsNone(registry.get_tool("non_existent"))
        
        # Test get_all_tools
        all_tools = registry.get_all_tools()
        self.assertEqual(len(all_tools), 2)
        self.assertIn(add_tool, all_tools)
        self.assertIn(multiply_tool, all_tools)
        
        # Test get_tools_as_dicts
        all_tools_dicts = registry.get_tools_as_dicts()
        self.assertEqual(len(all_tools_dicts), 2)
        self.assertEqual(all_tools_dicts[0]["name"], "add")
        self.assertEqual(all_tools_dicts[1]["name"], "multiply")
        
        # Test format_for_prompt
        formatted = registry.format_for_prompt()
        self.assertIn("## add", formatted)
        self.assertIn("## multiply", formatted)
        self.assertIn("Add two numbers together", formatted)
        self.assertIn("Multiply two numbers together", formatted)
    
    def test_default_registry(self):
        """Test the default tool registry."""
        # The default registry should have some predefined tools
        self.assertGreater(len(default_registry.tools), 0)
        
        # Test some specific tools that should be present
        self.assertIsNotNone(default_registry.get_tool("split_text"))
        self.assertIsNotNone(default_registry.get_tool("join_text"))
        self.assertIsNotNone(default_registry.get_tool("to_lowercase"))
        
        # Test that tools are callable
        result = default_registry.get_tool("to_lowercase")("TEST")
        self.assertEqual(result, "test")


if __name__ == "__main__":
    unittest.main()