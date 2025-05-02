"""
Tool call parser helper for extracting and validating tool calls from LLM responses.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_tool_call(text: str, preferred_tags: List[str] = None) -> Optional[str]:
    """
    Extract the content within tool call tags.

    Args:
        text: The text to extract tool call from
        preferred_tags: The tag names to look for (default: ['tool_call'])

    Returns:
        The extracted content or None if no tool call found
    """
    preferred_tags = preferred_tags or ["tool_call"]
    for tag in preferred_tags:
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    return None


def parse_tool_call(
    response: str, available_tools: List[Dict] = None, preferred_tags: List[str] = None
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Parse a tool call from an LLM response.

    Args:
        response: The LLM response text to parse
        available_tools: Optional list of available tools for validation
        preferred_tags: The tag names to look for (default: ['tool_call'])

    Returns:
        Tuple of (tool_name, arguments, is_error)
        - tool_name: Name of the called tool or "-ERROR-" if invalid
        - arguments: Dictionary of arguments provided to the tool
        - is_error: Boolean indicating if there was an error parsing
    """
    # Extract content from tags
    tool_call_content = extract_tool_call(response, preferred_tags)

    if not tool_call_content:
        return "-ERROR-", {}, True

    # Parse JSON
    try:
        # Handle potential single quotes
        tool_call_content = tool_call_content.replace("'", '"')

        tool_call = json.loads(tool_call_content)

        # Extract tool name and arguments
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Validate tool existence if tools are provided
        if available_tools:
            valid_tool_names = set()
            for tool in available_tools:
                if isinstance(tool, dict):
                    if "name" in tool:
                        valid_tool_names.add(tool["name"])
                    elif "function" in tool and "name" in tool["function"]:
                        valid_tool_names.add(tool["function"]["name"])

            if not tool_name or tool_name not in valid_tool_names:
                return "-ERROR-", arguments, True

        return tool_name, arguments, False

    except (json.JSONDecodeError, Exception) as json_error:
        logger.error(f"Failed to parse tool call: {json_error}", exc_info=True)
        return "-ERROR-", {}, True


def format_tool_call_for_hangman(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Format a parsed tool call into the format expected by Hangman.

    Args:
        tool_name: The name of the tool to call
        arguments: Dictionary of arguments

    Returns:
        The formatted tool call string (e.g., "[letter]" or "[word]")
    """
    if tool_name == "guess_letter" and "letter" in arguments:
        return f"[{arguments['letter']}]"
    elif tool_name == "guess_word" and "word" in arguments:
        return f"[{arguments['word']}]"
    else:
        return "-ERROR-"
