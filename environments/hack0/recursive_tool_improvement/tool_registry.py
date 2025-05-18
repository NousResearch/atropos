"""
Tool Registry for Recursive Tool Improvement Environment

This module provides a registry of tools that can be composed by language models
to solve problems. Each tool has a well-defined interface with documented parameters
and examples of usage.
"""

import re
import json
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

class Tool:
    """
    Represents a single tool that can be used in tool compositions.
    
    Attributes:
        name: The name of the tool (used for calling in compositions)
        description: A human-readable description of what the tool does
        parameters: Dict mapping parameter names to their type and description
        examples: List of example usages of the tool
        function: The actual function that implements the tool behavior
    """
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Dict[str, str]], 
        function: Callable,
        examples: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.examples = examples or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool (without the function)"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples
        }
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying function"""
        return self.function(*args, **kwargs)


class ToolRegistry:
    """
    Registry of available tools for the Recursive Tool Improvement Environment.
    
    Acts as a central registry for all tools that can be composed together
    to solve problems.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool with the registry"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tools_as_dicts(self) -> List[Dict[str, Any]]:
        """Get all tools as dictionaries (for serialization)"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def format_for_prompt(self) -> str:
        """Format all tools for inclusion in a prompt"""
        result = []
        
        for tool in self.tools.values():
            tool_str = f"## {tool.name}\n\n{tool.description}\n\n"
            
            tool_str += "Parameters:\n"
            for param_name, param_info in tool.parameters.items():
                tool_str += f"- {param_name}: {param_info['type']} - {param_info['description']}\n"
            
            if tool.examples:
                tool_str += "\nExamples:\n"
                for example in tool.examples:
                    tool_str += f"- `{example}`\n"
            
            result.append(tool_str)
        
        return "\n\n".join(result)


# Define basic text processing functions

def split_text(text: str, delimiter: str = " ") -> List[str]:
    """Split text by delimiter"""
    return text.split(delimiter)

def join_text(parts: List[str], delimiter: str = " ") -> str:
    """Join text parts with delimiter"""
    return delimiter.join(parts)

def extract_regex(text: str, pattern: str) -> List[str]:
    """Extract text matching a regex pattern"""
    return re.findall(pattern, text)

def replace_text(text: str, old: str, new: str) -> str:
    """Replace occurrences of old text with new text"""
    return text.replace(old, new)

def regex_replace(text: str, pattern: str, replacement: str) -> str:
    """Replace text matching a regex pattern"""
    return re.sub(pattern, replacement, text)

def to_lowercase(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

def to_uppercase(text: str) -> str:
    """Convert text to uppercase"""
    return text.upper()

def count_occurrences(text: str, substring: str) -> int:
    """Count occurrences of substring in text"""
    return text.count(substring)

def parse_json(text: str) -> Any:
    """Parse JSON string into Python object"""
    return json.loads(text)

def format_json(obj: Any, indent: int = 2) -> str:
    """Format Python object as a JSON string"""
    return json.dumps(obj, indent=indent)

def extract_between(text: str, start: str, end: str) -> List[str]:
    """Extract text between start and end markers"""
    pattern = f"{re.escape(start)}(.*?){re.escape(end)}"
    return re.findall(pattern, text, re.DOTALL)

def filter_list(items: List[str], contains: str = "") -> List[str]:
    """Filter list to items containing a substring"""
    return [item for item in items if contains in item]

def sort_list(items: List[str], reverse: bool = False) -> List[str]:
    """Sort a list of strings"""
    return sorted(items, reverse=reverse)

def remove_duplicates(items: List[str]) -> List[str]:
    """Remove duplicate items from a list while preserving order"""
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]

def trim_whitespace(text: str) -> str:
    """Remove leading and trailing whitespace"""
    return text.strip()


# Create the default tool registry with basic text processing tools
def create_default_registry() -> ToolRegistry:
    """Create and populate a default tool registry with basic text processing tools"""
    registry = ToolRegistry()
    
    # Register text processing tools
    registry.register(Tool(
        name="split_text",
        description="Splits text into parts using a delimiter",
        parameters={
            "text": {"type": "string", "description": "The text to split"},
            "delimiter": {"type": "string", "description": "The delimiter to split on (default is space)"}
        },
        function=split_text,
        examples=["split_text(\"hello world\", \" \")", "split_text(\"a,b,c\", \",\")"]
    ))
    
    registry.register(Tool(
        name="join_text",
        description="Joins a list of text parts with a delimiter",
        parameters={
            "parts": {"type": "list[string]", "description": "List of text parts to join"},
            "delimiter": {"type": "string", "description": "The delimiter to join with (default is space)"}
        },
        function=join_text,
        examples=["join_text([\"hello\", \"world\"], \" \")", "join_text([\"a\", \"b\", \"c\"], \",\")"]
    ))
    
    registry.register(Tool(
        name="extract_regex",
        description="Extracts all text matching a regular expression pattern",
        parameters={
            "text": {"type": "string", "description": "The text to search in"},
            "pattern": {"type": "string", "description": "Regular expression pattern to match"}
        },
        function=extract_regex,
        examples=[
            "extract_regex(\"Email me at john@example.com\", \"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\")",
            "extract_regex(\"Prices: $10, $25, $42\", \"\\$\\d+\")"
        ]
    ))
    
    registry.register(Tool(
        name="replace_text",
        description="Replaces occurrences of text with new text",
        parameters={
            "text": {"type": "string", "description": "The text to modify"},
            "old": {"type": "string", "description": "Text to be replaced"},
            "new": {"type": "string", "description": "Replacement text"}
        },
        function=replace_text,
        examples=["replace_text(\"Hello world\", \"world\", \"universe\")"]
    ))
    
    registry.register(Tool(
        name="regex_replace",
        description="Replaces text matching a regular expression pattern",
        parameters={
            "text": {"type": "string", "description": "The text to modify"},
            "pattern": {"type": "string", "description": "Regular expression pattern to match"},
            "replacement": {"type": "string", "description": "Replacement text (can use \\1, \\2, etc. for capture groups)"}
        },
        function=regex_replace,
        examples=["regex_replace(\"hello 123 world\", \"\\d+\", \"[number]\")", "regex_replace(\"06/15/2023\", \"(\\d{2})/(\\d{2})/(\\d{4})\", \"\\3-\\1-\\2\")"]
    ))
    
    registry.register(Tool(
        name="to_lowercase",
        description="Converts text to lowercase",
        parameters={
            "text": {"type": "string", "description": "The text to convert"}
        },
        function=to_lowercase,
        examples=["to_lowercase(\"Hello World\")"]
    ))
    
    registry.register(Tool(
        name="to_uppercase",
        description="Converts text to uppercase",
        parameters={
            "text": {"type": "string", "description": "The text to convert"}
        },
        function=to_uppercase,
        examples=["to_uppercase(\"Hello World\")"]
    ))
    
    registry.register(Tool(
        name="count_occurrences",
        description="Counts occurrences of a substring in text",
        parameters={
            "text": {"type": "string", "description": "The text to search in"},
            "substring": {"type": "string", "description": "The substring to count"}
        },
        function=count_occurrences,
        examples=["count_occurrences(\"hello hello world\", \"hello\")"]
    ))
    
    registry.register(Tool(
        name="parse_json",
        description="Parses a JSON string into a Python object",
        parameters={
            "text": {"type": "string", "description": "JSON string to parse"}
        },
        function=parse_json,
        examples=["parse_json(\"{\\\"name\\\": \\\"John\\\", \\\"age\\\": 30}\")"]
    ))
    
    registry.register(Tool(
        name="format_json",
        description="Formats a Python object as a JSON string",
        parameters={
            "obj": {"type": "any", "description": "Object to format as JSON"},
            "indent": {"type": "integer", "description": "Number of spaces for indentation (default is 2)"}
        },
        function=format_json,
        examples=["format_json({\"name\": \"John\", \"age\": 30})"]
    ))
    
    registry.register(Tool(
        name="extract_between",
        description="Extracts all text between start and end markers",
        parameters={
            "text": {"type": "string", "description": "The text to search in"},
            "start": {"type": "string", "description": "Start marker"},
            "end": {"type": "string", "description": "End marker"}
        },
        function=extract_between,
        examples=["extract_between(\"<h1>Title</h1> <p>Content</p>\", \"<p>\", \"</p>\")"]
    ))
    
    registry.register(Tool(
        name="filter_list",
        description="Filters a list to items containing a substring",
        parameters={
            "items": {"type": "list[string]", "description": "List of items to filter"},
            "contains": {"type": "string", "description": "Substring that items must contain"}
        },
        function=filter_list,
        examples=["filter_list([\"apple\", \"banana\", \"cherry\"], \"a\")"]
    ))
    
    registry.register(Tool(
        name="sort_list",
        description="Sorts a list of strings",
        parameters={
            "items": {"type": "list[string]", "description": "List of items to sort"},
            "reverse": {"type": "boolean", "description": "Whether to sort in descending order (default is False)"}
        },
        function=sort_list,
        examples=["sort_list([\"c\", \"a\", \"b\"])", "sort_list([\"c\", \"a\", \"b\"], reverse=True)"]
    ))
    
    registry.register(Tool(
        name="remove_duplicates",
        description="Removes duplicate items from a list while preserving order",
        parameters={
            "items": {"type": "list[string]", "description": "List of items to deduplicate"}
        },
        function=remove_duplicates,
        examples=["remove_duplicates([\"a\", \"b\", \"a\", \"c\", \"b\"])"]
    ))
    
    registry.register(Tool(
        name="trim_whitespace",
        description="Removes leading and trailing whitespace from text",
        parameters={
            "text": {"type": "string", "description": "The text to trim"}
        },
        function=trim_whitespace,
        examples=["trim_whitespace(\"  hello world  \")"]
    ))
    
    return registry


# Default singleton instance
default_registry = create_default_registry()