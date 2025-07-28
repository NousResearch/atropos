"""Utility functions for parsing memory blocks from agent responses."""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def extract_memory_block(response_text: str) -> Optional[str]:
    """
    Extract memory content from <memory> tags in agent response.

    Args:
        response_text: The full response text from the agent

    Returns:
        The extracted memory content if found, None otherwise
    """
    if not response_text:
        return None

    # Pattern to match <memory> content </memory>
    memory_pattern = r"<memory>\s*(.*?)\s*</memory>"

    match = re.search(memory_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        memory_content = match.group(1).strip()
        logger.debug(f"Extracted memory: {memory_content[:100]}...")
        return memory_content
    else:
        logger.debug("No memory block found in response")
        return None


def extract_thinking_and_memory(
    response_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract both thinking and memory blocks from agent response.

    Args:
        response_text: The full response text from the agent

    Returns:
        Tuple of (thinking_content, memory_content), either can be None
    """
    thinking_pattern = r"<think>\s*(.*?)\s*</think>"
    memory_pattern = r"<memory>\s*(.*?)\s*</memory>"

    thinking_match = re.search(
        thinking_pattern, response_text, re.DOTALL | re.IGNORECASE
    )
    memory_match = re.search(memory_pattern, response_text, re.DOTALL | re.IGNORECASE)

    thinking_content = thinking_match.group(1).strip() if thinking_match else None
    memory_content = memory_match.group(1).strip() if memory_match else None

    return thinking_content, memory_content


def validate_memory_content(memory_content: str) -> bool:
    """
    Validate that memory content meets basic requirements.

    Args:
        memory_content: The extracted memory content

    Returns:
        True if memory is valid, False otherwise
    """
    if not memory_content:
        return False

    # Check minimum length (at least 10 characters)
    if len(memory_content) < 10:
        logger.warning(f"Memory too short: {len(memory_content)} characters")
        return False

    # Check maximum length (prevent overly long memories)
    if len(memory_content) > 500:
        logger.warning(f"Memory too long: {len(memory_content)} characters")
        return False

    # Check that it's not just whitespace or punctuation
    if not any(c.isalnum() for c in memory_content):
        logger.warning("Memory contains no alphanumeric characters")
        return False

    return True
