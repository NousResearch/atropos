"""
FLE Tool Discovery Utility

This module provides functionality to dynamically discover and extract signatures
from Factorio Learning Environment (FLE) tools. It can be used by any FLE-based
environment to get accurate tool information for LLM prompting.
"""

import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def discover_fle_tools(
    fle_path: Optional[Path] = None,
    tool_categories: List[str] = ["agent"]
) -> Dict[str, Dict[str, Any]]:
    """
    Discover all FLE tools and extract their signatures.
    
    Args:
        fle_path: Path to FLE installation. If None, tries to find it automatically.
        tool_categories: List of tool categories to discover (e.g., ["agent", "admin"])
    
    Returns:
        Dictionary mapping tool names to their metadata including signatures
    """
    if fle_path is None:
        # Try to find FLE path automatically
        fle_path = Path(__file__).parent / "fle" / "fle" / "env" / "tools"
    else:
        fle_path = Path(fle_path) / "tools"
    
    if not fle_path.exists():
        logger.error(f"FLE tools path not found: {fle_path}")
        return {}
    
    tools = {}
    
    for category in tool_categories:
        category_path = fle_path / category
        if not category_path.exists():
            logger.warning(f"Tool category path not found: {category_path}")
            continue
        
        # Scan all subdirectories in the category
        for tool_dir in category_path.iterdir():
            if not tool_dir.is_dir():
                continue
            
            client_file = tool_dir / "client.py"
            if not client_file.exists():
                continue
            
            tool_name = tool_dir.name
            
            try:
                tool_info = extract_tool_from_file(client_file, tool_name, category)
                if tool_info:
                    tools[tool_name] = tool_info
            except Exception as e:
                logger.error(f"Failed to extract tool {tool_name}: {e}")
    
    return tools


def extract_tool_from_file(
    client_file: Path, 
    tool_name: str,
    category: str
) -> Optional[Dict[str, Any]]:
    """
    Extract tool information from a client.py file.
    
    Args:
        client_file: Path to the client.py file
        tool_name: Name of the tool (directory name)
        category: Category of the tool (agent/admin)
    
    Returns:
        Dictionary with tool metadata or None if extraction fails
    """
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(
            f"fle_tools.{category}.{tool_name}",
            client_file
        )
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Find the tool class - look for classes that inherit from Tool
        tool_class = None
        
        # First, try to import the Tool base class to check inheritance
        try:
            from fle.env.tools import Tool as FLETool
            tool_base_class = FLETool
        except ImportError:
            tool_base_class = None
        
        # Find classes that are likely to be the tool
        candidates = []
        for name, obj in inspect.getmembers(module):
            if not inspect.isclass(obj):
                continue
            
            # Skip imported classes and base classes
            if obj.__module__ != module.__name__:
                continue
            
            # Skip known base classes and enums
            if name in ['Tool', 'Controller', 'Direction', 'Prototype', 'Resource', 'Position']:
                continue
            
            # Check if it has __call__ method defined in this class (not inherited from object)
            if hasattr(obj, '__call__') and '__call__' in obj.__dict__:
                # Prefer classes that inherit from Tool if we have it
                if tool_base_class and issubclass(obj, tool_base_class):
                    tool_class = obj
                    break
                candidates.append(obj)
        
        # If no Tool subclass found, use the first candidate
        if tool_class is None and candidates:
            tool_class = candidates[0]
        
        if tool_class is None:
            logger.warning(f"No tool class found in {client_file}")
            return None
        
        # Extract signature from __call__ method
        call_method = getattr(tool_class, '__call__')
        signature = inspect.signature(call_method)
        
        # Extract parameters
        parameters = []
        for param_name, param in signature.parameters.items():
            if param_name in ['self', 'args', 'kwargs']:
                continue
            
            param_info = {
                "name": param_name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": None if param.default == inspect.Parameter.empty else repr(param.default),
                "required": param.default == inspect.Parameter.empty
            }
            
            # Clean up type annotations
            param_info["type"] = clean_type_annotation(param_info["type"])
            
            parameters.append(param_info)
        
        # Extract docstring
        docstring = inspect.getdoc(call_method) or ""
        description = docstring.split('\n')[0] if docstring else f"Execute {tool_name} action"
        
        # Extract return type
        return_type = "Any"
        if signature.return_annotation != inspect.Parameter.empty:
            return_type = clean_type_annotation(str(signature.return_annotation))
        
        return {
            "name": tool_name,
            "class_name": tool_class.__name__,
            "category": category,
            "parameters": parameters,
            "description": description,
            "returns": return_type,
            "docstring": docstring
        }
        
    except Exception as e:
        logger.error(f"Error extracting tool from {client_file}: {e}")
        return None


def clean_type_annotation(type_str: str) -> str:
    """
    Clean up type annotation strings for better readability.
    
    Args:
        type_str: Raw type annotation string
    
    Returns:
        Cleaned type annotation
    """
    # Remove module prefixes for common types
    replacements = {
        "<class '": "",
        "'>": "",
        "typing.": "",
        "fle.env.entities.": "",
        "fle.env.game_types.": "",
        "NoneType": "None",
    }
    
    for old, new in replacements.items():
        type_str = type_str.replace(old, new)
    
    # Handle Union types more cleanly
    if "Union[" in type_str or "Optional[" in type_str:
        type_str = type_str.replace("Union[", "Union[")
        type_str = type_str.replace("Optional[", "Optional[")
    
    return type_str


def format_tool_for_prompt(tool_info: Dict[str, Any]) -> str:
    """
    Format tool information for inclusion in an LLM prompt.
    
    Args:
        tool_info: Tool metadata dictionary
    
    Returns:
        Formatted string describing the tool
    """
    # Build parameter string
    params = []
    for param in tool_info["parameters"]:
        param_str = f"{param['name']}: {param['type']}"
        if param["default"] is not None:
            param_str += f" = {param['default']}"
        params.append(param_str)
    
    param_string = ", ".join(params)
    
    return {
        "name": tool_info["name"],
        "description": tool_info["description"],
        "arguments": {
            param["name"]: f"{param['type']}" + (f" (default: {param['default']})" if param["default"] else " (required)")
            for param in tool_info["parameters"]
        }
    }


def get_tool_signatures_for_prompt(
    tools: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[str]:
    """
    Get tool signatures formatted for an LLM system prompt.
    
    Args:
        tools: Tool dictionary. If None, discovers tools automatically.
    
    Returns:
        List of formatted tool signature strings
    """
    if tools is None:
        tools = discover_fle_tools()
    
    signatures = []
    for tool_name, tool_info in sorted(tools.items()):
        formatted = format_tool_for_prompt(tool_info)
        signatures.append(f"- {formatted}")
    
    return signatures


if __name__ == "__main__":
    # Quick test when run directly
    logging.basicConfig(level=logging.INFO)
    tools = discover_fle_tools()
    print(f"Discovered {len(tools)} tools:")
    for name, info in sorted(tools.items()):
        print(f"  - {name}: {info['description']}")