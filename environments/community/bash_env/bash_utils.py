"""
Bash Command Utilities

Provides utilities for processing and comparing Bash commands.
Used by the NL2Bash Environment for reward verification.
"""

import re
import shlex
from typing import Optional


def normalize_bash(cmd: str) -> str:
    """
    Normalize a bash command for comparison.

    Normalizations applied:
    - Strip leading/trailing whitespace
    - Normalize internal whitespace (collapse multiple spaces)
    - Handle common quoting variations

    Args:
        cmd: Raw bash command string

    Returns:
        Normalized command string
    """
    if not cmd:
        return ""

    # Strip whitespace
    cmd = cmd.strip()

    # Normalize internal whitespace
    cmd = re.sub(r"\s+", " ", cmd)

    return cmd


def extract_boxed_bash(text: str) -> Optional[str]:
    """
    Extract Bash command from \\boxed{} format in LLM response.

    Args:
        text: LLM response text

    Returns:
        Extracted Bash command string, or None if not found
    """
    if not text:
        return None

    # Try to find \boxed{...} pattern
    # Handle both \\boxed{} and \boxed{} formats
    patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Handles nested braces
        r"\\boxed\{(.+?)\}",  # Simple pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            bash_cmd = match.group(1).strip()
            if bash_cmd:
                return bash_cmd

    return None


def commands_match(
    generated: str,
    gold: str,
    alt_gold: Optional[str] = None,
) -> bool:
    """
    Check if generated command matches gold or alternative.

    Comparison strategy:
    1. Exact match against gold or alt_gold
    2. Normalized match (whitespace, etc.)

    Args:
        generated: Generated bash command
        gold: Primary gold bash command
        alt_gold: Optional alternative gold command (from bash2 field)

    Returns:
        True if commands match, False otherwise
    """
    if not generated:
        return False

    # Normalize all commands
    gen_norm = normalize_bash(generated)
    gold_norm = normalize_bash(gold)

    # Check primary gold
    if gen_norm == gold_norm:
        return True

    # Check exact match (in case normalization removes something)
    if generated.strip() == gold.strip():
        return True

    # Check alternative gold if provided
    if alt_gold:
        alt_norm = normalize_bash(alt_gold)
        if gen_norm == alt_norm:
            return True
        if generated.strip() == alt_gold.strip():
            return True

    return False


def is_valid_bash_syntax(cmd: str) -> bool:
    """
    Perform a basic syntax check on a bash command.

    This is a lightweight check that catches obvious issues
    without actually executing the command.

    Args:
        cmd: Bash command to check

    Returns:
        True if command appears syntactically valid
    """
    if not cmd or not cmd.strip():
        return False

    # Check for unclosed quotes
    try:
        shlex.split(cmd)
    except ValueError:
        return False

    # Check for obviously incomplete commands
    cmd_stripped = cmd.strip()
    if cmd_stripped.endswith(("&&", "||", "|", ";")):
        return False

    return True
