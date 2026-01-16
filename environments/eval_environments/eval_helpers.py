"""
Shared helper functions for evaluation environments.

This module contains common utilities used across multiple eval environments,
making it easier to maintain consistent behavior and update logic in one place.

Includes:
- MCQA answer extraction (letter-based)
- Numbered choice extraction
- Freeform answer extraction
- Thinking mode validation
- Math answer verification (using math_verify library)
- System prompt creation
- Results saving utilities
- Reasoning content extraction from various API response formats
"""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# REASONING/THINKING PROMPTS
# =============================================================================
# Standard prompts for triggering reasoning mode in various models.
# These are NOT automatically injected - use explicitly when desired.

HERMES_REASONING_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
    "consider the problem and deliberate with yourself via systematic reasoning processes "
    "to help come to a correct solution prior to answering. You should enclose your "
    "thoughts and internal monologue inside <think> </think> tags, and then provide your "
    "solution or response to the problem."
)
"""
Standard reasoning prompt for Hermes models.

This prompt triggers the model to use extended chain-of-thought reasoning
with explicit <think></think> tags. Use this when you want visible reasoning
in the response content.

Example usage:
    from eval_helpers import HERMES_REASONING_PROMPT

    messages = [
        {"role": "system", "content": HERMES_REASONING_PROMPT},
        {"role": "user", "content": question},
    ]
"""

HERMES_REASONING_PROMPT_WITH_ANSWER = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
    "consider the problem and deliberate with yourself via systematic reasoning processes "
    "to help come to a correct solution prior to answering. You should enclose your "
    "thoughts and internal monologue inside <think> </think> tags, and then provide your "
    "solution or response to the problem. After your thinking, provide your final answer "
    "inside <answer></answer> tags."
)
"""
Standard reasoning prompt for Hermes models with explicit answer tag instruction.

Use this when you want the model to clearly separate reasoning from the final answer.
"""

# Try to import math_verify libraries (optional dependency for math evals)
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    NormalizationConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None
    TimeoutException = Exception


# Pre-compiled regex for answer tag extraction
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# Pre-compiled regex for thinking mode
THINK_OPEN_PATTERN = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)
THINK_CONTENT_AFTER_PATTERN = re.compile(r"</think>\s*(.*)", re.DOTALL | re.IGNORECASE)
THINK_CONTENT_INSIDE_PATTERN = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)

# Pre-compiled regex for scratchpad mode (alternative reasoning format)
SCRATCHPAD_OPEN_PATTERN = re.compile(r"<\|start_of_scratchpad\|>")
SCRATCHPAD_CLOSE_PATTERN = re.compile(r"<\|end_of_scratchpad\|>")
SCRATCHPAD_CONTENT_AFTER_PATTERN = re.compile(
    r"<\|end_of_scratchpad\|>\s*(.*)", re.DOTALL
)
SCRATCHPAD_CONTENT_INSIDE_PATTERN = re.compile(
    r"<\|start_of_scratchpad\|>(.*?)<\|end_of_scratchpad\|>", re.DOTALL
)


# Common prefixes that models use before stating their answer
# These will be stripped to help isolate the actual answer
ANSWER_PREFIXES = [
    # "Final Answer" variants
    r"(?:the\s+)?final\s+answer\s+is\s*:?\s*",
    r"(?:my\s+)?final\s+answer\s*:?\s*",
    # "Answer" variants
    r"(?:the\s+)?answer\s+is\s*:?\s*",
    r"(?:the\s+)?correct\s+answer\s+is\s*:?\s*",
    r"(?:my\s+)?answer\s*:?\s*",
    r"answer\s*:\s*",
    # "Choice/Option" variants
    r"(?:the\s+)?(?:correct\s+)?(?:choice|option)\s+is\s*:?\s*",
    r"(?:i\s+)?(?:choose|select|pick)\s*:?\s*",
    # "It is/It's" variants
    r"(?:it\s+is|it's)\s*:?\s*",
    r"(?:that\s+would\s+be|that's)\s*:?\s*",
    # "I think/believe" variants
    r"(?:i\s+)?(?:think|believe|would\s+say)\s+(?:it\s+is|it's|the\s+answer\s+is)\s*:?\s*",
]

# Compile the prefix patterns
ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:" + "|".join(ANSWER_PREFIXES) + r")", re.IGNORECASE
)


def extract_letter_from_answer_tag(
    response: str,
    valid_letters: Set[str],
    debug: bool = False,
    choices: Optional[List[str]] = None,
) -> Tuple[Optional[str], str]:
    """
    Extract a single letter answer from <answer></answer> tags.

    Uses multiple strategies:
    1. Exact choice text matching (if choices provided)
    2. Word boundary matching for single letter

    Handles many common variations:
    - <answer>A</answer>                    -> A (single letter)
    - <answer>A.</answer>                   -> A (letter with punctuation)
    - <answer>(A)</answer>                  -> A (letter in parentheses)
    - <answer>Final Answer: B</answer>      -> B (common prefix stripped)
    - <answer>The answer is C</answer>      -> C (common prefix stripped)
    - <answer>Tom Holland</answer>          -> A (if "Tom Holland" is choice A)
    - <answer>A) Tom Holland</answer>       -> A (choice text with letter prefix)
    - <answer>A - Tom Holland</answer>      -> A (choice text with letter prefix)

    Rejects ambiguous cases like:
    - <answer>A is better than B</answer>   -> None (multiple valid letters)
    - <answer>Between A and C</answer>      -> None (multiple valid letters)

    Args:
        response: The model's response (content after </think> in thinking mode)
        valid_letters: Set of valid answer letters (e.g., {'A', 'B', 'C', 'D'})
        debug: Whether to print debug information
        choices: Optional list of choice texts in order (A, B, C, D...)

    Returns:
        Tuple of (extracted_letter or None, extraction_method)
    """
    # Find the first <answer></answer> tag
    answer_tag_match = ANSWER_TAG_PATTERN.search(response)
    if not answer_tag_match:
        return None, "no_answer_tag"

    answer_content = answer_tag_match.group(1).strip()
    if not answer_content:
        return None, "empty_answer_tag"

    # Try stripping common prefixes to isolate the answer
    content_to_check = answer_content
    prefix_match = ANSWER_PREFIX_PATTERN.match(content_to_check)
    if prefix_match:
        content_to_check = content_to_check[prefix_match.end() :].strip()  # noqa: E203
        if debug:
            print(f"    Stripped prefix, remaining content: '{content_to_check}'")

    # STRICT CHECK: If after stripping, we have just a letter (with optional punctuation)
    # This is the ideal case: <answer>B</answer> or <answer>Final Answer: B</answer>
    cleaned = content_to_check.strip(".,)(:;!? \t\n")
    if cleaned.upper() in valid_letters:
        if debug:
            print(
                f"    Extracted '{cleaned.upper()}' using method 'answer_tag' (strict letter match)"
            )
        return cleaned.upper(), "answer_tag"

    # CHOICE TEXT MATCHING: Check if answer matches a choice text
    if choices:
        letter_from_choice = _match_choice_text(
            answer_content, choices, valid_letters, debug
        )
        if letter_from_choice:
            return letter_from_choice, "answer_tag_choice_match"

    # WORD BOUNDARY CHECK: Find all valid letters as standalone words
    letters_pattern = "|".join(sorted(valid_letters))
    word_bounded_pattern = re.compile(rf"\b({letters_pattern})\b", re.IGNORECASE)

    # Find ALL valid letters in the (prefix-stripped) content
    found_letters = word_bounded_pattern.findall(content_to_check.upper())

    # Only accept if EXACTLY ONE valid letter is found
    if len(found_letters) == 1:
        letter = found_letters[0].upper()
        if debug:
            print(
                f"    Extracted '{letter}' using method 'answer_tag' (single word-bounded letter)"
            )
        return letter, "answer_tag"
    elif len(found_letters) > 1:
        # If multiple found after prefix strip, try the original content
        found_in_original = word_bounded_pattern.findall(answer_content.upper())
        if len(found_in_original) == 1:
            letter = found_in_original[0].upper()
            if debug:
                print(
                    f"    Extracted '{letter}' using method 'answer_tag' (from original content)"
                )
            return letter, "answer_tag"
        if debug:
            print(
                f"    Multiple letters found in answer tag: {found_letters} - rejecting"
            )
        return None, "answer_tag_ambiguous"
    else:
        # No letters found after prefix strip, try original content
        found_in_original = word_bounded_pattern.findall(answer_content.upper())
        if len(found_in_original) == 1:
            letter = found_in_original[0].upper()
            if debug:
                print(
                    f"    Extracted '{letter}' using method 'answer_tag' (from original content)"
                )
            return letter, "answer_tag"
        if debug:
            print(
                f"    No valid letters found in answer tag content: '{answer_content}'"
            )
        return None, "answer_tag_no_letter"


def _match_choice_text(
    answer_content: str,
    choices: List[str],
    valid_letters: Set[str],
    debug: bool = False,
) -> Optional[str]:
    """
    Match answer content against choice texts.

    Handles formats like:
    - "Tom Holland" (exact choice text)
    - "A) Tom Holland" (letter prefix + choice text)
    - "A - Tom Holland" (letter prefix + choice text)
    - "A. Tom Holland" (letter prefix + choice text)

    Args:
        answer_content: The content inside <answer> tags
        choices: List of choice texts in order (index 0 = A, 1 = B, etc.)
        valid_letters: Set of valid letters
        debug: Whether to print debug info

    Returns:
        The letter if a match is found, None otherwise
    """
    answer_lower = answer_content.lower().strip()
    answer_normalized = re.sub(r"\s+", " ", answer_lower)  # Normalize whitespace

    for i, choice_text in enumerate(choices):
        if i >= len(valid_letters):
            break
        letter = ascii_uppercase[i]
        if letter not in valid_letters:
            continue

        choice_lower = choice_text.lower().strip()
        choice_normalized = re.sub(r"\s+", " ", choice_lower)

        # Check exact match with choice text
        if answer_normalized == choice_normalized:
            if debug:
                print(
                    f"    Extracted '{letter}' via exact choice text match: '{choice_text}'"
                )
            return letter

        # Check if answer contains choice text (for longer answers)
        if choice_normalized and choice_normalized in answer_normalized:
            # Make sure it's a substantial match (not just a single word)
            if len(choice_normalized) >= 3:
                if debug:
                    print(
                        f"    Extracted '{letter}' via choice text containment: '{choice_text}'"
                    )
                return letter

        # Check for "A) choice text", "A - choice text", "A. choice text" patterns
        prefixed_patterns = [
            rf"^{letter}\s*[\)\-\.:\]]\s*",  # A), A-, A., A:, A]
            rf"^\({letter}\)\s*",  # (A)
        ]
        for pattern in prefixed_patterns:
            stripped = re.sub(pattern, "", answer_content, flags=re.IGNORECASE).strip()
            stripped_normalized = re.sub(r"\s+", " ", stripped.lower())
            if stripped_normalized == choice_normalized:
                if debug:
                    print(
                        f"    Extracted '{letter}' via prefixed choice text match: '{choice_text}'"
                    )
                return letter

    return None


def extract_number_from_answer_tag(
    response: str, num_choices: int, debug: bool = False
) -> Tuple[Optional[int], str]:
    """
    Extract a single number answer from <answer></answer> tags.

    Uses word boundary matching to find valid numbers. Only returns a match
    if EXACTLY ONE valid number (1 to num_choices) is found.

    Handles variations like:
    - <answer>2</answer>                -> 2
    - <answer>2.</answer>               -> 2
    - <answer>Choice 2</answer>         -> 2
    - <answer>The answer is 3</answer>  -> 3
    - <answer>Option 1</answer>         -> 1

    Args:
        response: The model's response (content after </think> in thinking mode)
        num_choices: Number of valid choices (e.g., 5 means valid range is 1-5)
        debug: Whether to print debug information

    Returns:
        Tuple of (extracted_number or None, extraction_method)
    """
    # Find the first <answer></answer> tag
    answer_tag_match = ANSWER_TAG_PATTERN.search(response)
    if not answer_tag_match:
        return None, "no_answer_tag"

    answer_content = answer_tag_match.group(1).strip()
    if not answer_content:
        return None, "empty_answer_tag"

    # Try stripping common prefixes
    content_to_check = answer_content
    prefix_match = ANSWER_PREFIX_PATTERN.match(content_to_check)
    if prefix_match:
        content_to_check = content_to_check[prefix_match.end() :].strip()  # noqa: E203
        if debug:
            print(f"    Stripped prefix, remaining content: '{content_to_check}'")

    # STRICT CHECK: If after stripping, we have just a number
    cleaned = content_to_check.strip(".,)(:;!? \t\n")
    try:
        num = int(cleaned)
        if 1 <= num <= num_choices:
            if debug:
                print(
                    f"    Extracted '{num}' using method 'answer_tag' (strict match after prefix strip)"
                )
            return num, "answer_tag"
    except ValueError:
        pass

    # WORD BOUNDARY CHECK: Find ALL word-bounded numbers
    word_bounded_numbers = re.findall(r"\b(\d+)\b", content_to_check)

    # Filter to valid range
    valid_numbers = []
    for num_str in word_bounded_numbers:
        try:
            num = int(num_str)
            if 1 <= num <= num_choices:
                valid_numbers.append(num)
        except ValueError:
            continue

    # Only accept if EXACTLY ONE valid number is found
    if len(valid_numbers) == 1:
        number = valid_numbers[0]
        if debug:
            print(
                f"    Extracted '{number}' using method 'answer_tag' (single word-bounded number)"
            )
        return number, "answer_tag"
    elif len(valid_numbers) > 1:
        # Try original content
        original_numbers = re.findall(r"\b(\d+)\b", answer_content)
        valid_in_original = [
            int(n) for n in original_numbers if 1 <= int(n) <= num_choices
        ]
        if len(valid_in_original) == 1:
            if debug:
                print(
                    f"    Extracted '{valid_in_original[0]}' using method 'answer_tag' (from original)"
                )
            return valid_in_original[0], "answer_tag"
        if debug:
            print(
                f"    Multiple valid numbers found in answer tag: {valid_numbers} - rejecting"
            )
        return None, "answer_tag_ambiguous"
    else:
        # Try original content
        original_numbers = re.findall(r"\b(\d+)\b", answer_content)
        valid_in_original = [
            int(n) for n in original_numbers if 1 <= int(n) <= num_choices
        ]
        if len(valid_in_original) == 1:
            if debug:
                print(
                    f"    Extracted '{valid_in_original[0]}' using method 'answer_tag' (from original)"
                )
            return valid_in_original[0], "answer_tag"
        if debug:
            print(
                f"    No valid numbers found in answer tag content: '{answer_content}'"
            )
        return None, "answer_tag_no_number"


def extract_freeform_from_answer_tag(
    response: str, debug: bool = False
) -> Tuple[Optional[str], str]:
    """
    Extract freeform text answer from <answer></answer> tags.

    Simply returns the stripped content inside the tags.
    Used for open-ended questions like DROP and SimpleQA.

    Args:
        response: The model's response (content after </think> in thinking mode)
        debug: Whether to print debug information

    Returns:
        Tuple of (extracted_text or None, extraction_method)
    """
    # Find the first <answer></answer> tag
    answer_tag_match = ANSWER_TAG_PATTERN.search(response)
    if not answer_tag_match:
        return None, "no_answer_tag"

    answer_content = answer_tag_match.group(1).strip()
    if not answer_content:
        return None, "empty_answer_tag"

    if debug:
        preview = (
            answer_content[:50] + "..." if len(answer_content) > 50 else answer_content
        )
        print(f"    Extracted '{preview}' using method 'answer_tag'")

    return answer_content, "answer_tag"


def validate_thinking_format(
    response: str, thinking_mode: bool = True
) -> Tuple[bool, str]:
    """
    Validate thinking format and extract content after reasoning tags.

    In thinking mode, we expect exactly one pair of reasoning tags.
    Supports both <think></think> and <|start_of_scratchpad|><|end_of_scratchpad|> formats.
    Returns the content after the closing tag for answer extraction.

    Args:
        response: The model's full response
        thinking_mode: Whether thinking mode is enabled

    Returns:
        Tuple of (is_valid, content_for_extraction)
    """
    if not thinking_mode:
        return True, response

    # Try <think></think> tags first
    think_open_count = len(THINK_OPEN_PATTERN.findall(response))
    think_close_count = len(THINK_CLOSE_PATTERN.findall(response))

    if think_open_count == 1 and think_close_count == 1:
        # Extract content after </think> tags for answer extraction
        match = THINK_CONTENT_AFTER_PATTERN.search(response)
        if match:
            return True, match.group(1).strip()

    # Try <|start_of_scratchpad|><|end_of_scratchpad|> tags
    scratchpad_open_count = len(SCRATCHPAD_OPEN_PATTERN.findall(response))
    scratchpad_close_count = len(SCRATCHPAD_CLOSE_PATTERN.findall(response))

    if scratchpad_open_count == 1 and scratchpad_close_count == 1:
        # Extract content after <|end_of_scratchpad|> tags for answer extraction
        match = SCRATCHPAD_CONTENT_AFTER_PATTERN.search(response)
        if match:
            return True, match.group(1).strip()

    # No valid reasoning format found
    return False, response


def extract_thinking_content(response: str) -> Optional[str]:
    """
    Extract the content inside reasoning tags.

    Supports both <think></think> and <|start_of_scratchpad|><|end_of_scratchpad|> formats.

    Args:
        response: The model's full response

    Returns:
        Content inside reasoning tags, or None if not found
    """
    # Try <think></think> tags first
    match = THINK_CONTENT_INSIDE_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    # Try <|start_of_scratchpad|><|end_of_scratchpad|> tags
    match = SCRATCHPAD_CONTENT_INSIDE_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    return None


def get_default_thinking_prompt(custom_prompt: Optional[str] = None) -> Optional[str]:
    """
    Get the thinking system prompt.

    By default, returns None (no prompt injection). Pass a custom prompt or use
    HERMES_REASONING_PROMPT explicitly if you want reasoning prompt injection.

    Args:
        custom_prompt: Optional custom thinking prompt to use. If None, returns None.
                      Use HERMES_REASONING_PROMPT for the standard Hermes prompt.

    Returns:
        The thinking prompt string, or None if no prompt specified.

    Example:
        # No prompt injection (default):
        prompt = get_default_thinking_prompt()  # Returns None

        # Use Hermes reasoning prompt:
        from eval_helpers import HERMES_REASONING_PROMPT
        prompt = get_default_thinking_prompt(HERMES_REASONING_PROMPT)
    """
    return custom_prompt  # None means no prompt injection


def get_thinking_prompt_or_hermes(custom_prompt: Optional[str] = None) -> str:
    """
    Get thinking prompt, defaulting to HERMES_REASONING_PROMPT if none provided.

    Use this when you want to ensure a thinking prompt is always used.

    Args:
        custom_prompt: Optional custom thinking prompt. If None, uses HERMES_REASONING_PROMPT.

    Returns:
        The thinking prompt string (never None).
    """
    return custom_prompt if custom_prompt else HERMES_REASONING_PROMPT


# =============================================================================
# REASONING CONTENT EXTRACTION
# =============================================================================
# Functions for extracting reasoning content from various API response formats.
# Different providers return reasoning in different ways:
# - OpenRouter/Nebius: reasoning_details[].text or reasoning_content field
# - Some providers: reasoning field in message
# - Hermes/others: <think></think> blocks in message content


def extract_reasoning_from_response(
    response: Any,
    content: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Extract reasoning content from various API response formats.

    This function handles multiple reasoning formats:
    1. reasoning_content field on the message (some providers)
    2. reasoning_details[].text field (OpenRouter style for reasoning models)
    3. reasoning field on the message (some providers)
    4. <think></think> blocks in message content (Hermes style)
    5. <|start_of_scratchpad|><|end_of_scratchpad|> blocks (alternative format)

    Args:
        response: The ChatCompletion response object from the API
        content: Optional message content string. If provided, will check for
                reasoning tag blocks in addition to API fields.

    Returns:
        Tuple of (reasoning_content, source) where:
        - reasoning_content: The extracted reasoning text, or None if not found
        - source: String indicating where reasoning was found:
          "reasoning_content", "reasoning_details", "reasoning", "think_block",
          "scratchpad_block", or "none"

    Example:
        completion = await server.chat_completion(messages=messages)
        message = completion.choices[0].message
        reasoning, source = extract_reasoning_from_response(
            completion.choices[0],
            content=message.content
        )
        if reasoning:
            print(f"Found reasoning via {source}: {len(reasoning)} chars")
    """
    # Try reasoning_content field (some providers like certain OpenAI-compatible APIs)
    if hasattr(response, "reasoning_content") and response.reasoning_content:
        return response.reasoning_content, "reasoning_content"

    # Try message.reasoning_content if response is a Choice
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content, "reasoning_content"
        if hasattr(message, "reasoning") and message.reasoning:
            return message.reasoning, "reasoning"

    # Try reasoning_details field (OpenRouter style)
    if hasattr(response, "reasoning_details") and response.reasoning_details:
        for detail in response.reasoning_details:
            if hasattr(detail, "text") and detail.text:
                return detail.text, "reasoning_details"
            # Some formats use 'content' instead of 'text'
            if isinstance(detail, dict) and detail.get("text"):
                return detail["text"], "reasoning_details"

    # Try message.reasoning_details if response is a Choice
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            for detail in message.reasoning_details:
                if hasattr(detail, "text") and detail.text:
                    return detail.text, "reasoning_details"
                if isinstance(detail, dict) and detail.get("text"):
                    return detail["text"], "reasoning_details"

    # Try reasoning field directly
    if hasattr(response, "reasoning") and response.reasoning:
        return response.reasoning, "reasoning"

    # Try <think> blocks in content (Hermes style)
    if content:
        match = THINK_CONTENT_INSIDE_PATTERN.search(content)
        if match:
            return match.group(1).strip(), "think_block"

    # Try <|start_of_scratchpad|> blocks in content (alternative reasoning format)
    if content:
        match = SCRATCHPAD_CONTENT_INSIDE_PATTERN.search(content)
        if match:
            return match.group(1).strip(), "scratchpad_block"

    return None, "none"


def extract_reasoning_from_completion(
    completion: Any,
    choice_idx: int = 0,
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Extract reasoning from a ChatCompletion object.

    Convenience wrapper around extract_reasoning_from_response that handles
    the common case of extracting from a ChatCompletion.

    Args:
        completion: The ChatCompletion response object
        choice_idx: Index of the choice to extract from (default 0)

    Returns:
        Tuple of (reasoning_content, source, message_content) where:
        - reasoning_content: The extracted reasoning text, or None
        - source: Where reasoning was found (see extract_reasoning_from_response)
        - message_content: The message content (for convenience)

    Example:
        completion = await server.chat_completion(messages=messages)
        reasoning, source, content = extract_reasoning_from_completion(completion)
    """
    if not completion or not completion.choices:
        return None, "none", None

    if choice_idx >= len(completion.choices):
        return None, "none", None

    choice = completion.choices[choice_idx]
    content = None

    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        content = choice.message.content

    reasoning, source = extract_reasoning_from_response(choice, content)
    return reasoning, source, content


def get_reasoning_token_usage(completion: Any) -> Dict[str, Any]:
    """
    Extract reasoning token usage information from a ChatCompletion.

    This extracts token counts from the usage field, including reasoning-specific
    metrics when available (e.g., reasoning_tokens from OpenRouter/OpenAI).

    Works with all known providers:
    - OpenAI: usage.completion_tokens_details.reasoning_tokens
    - OpenRouter (Claude, Hermes, DeepSeek, etc.): Same location + provider/cost fields

    Args:
        completion: The ChatCompletion response object

    Returns:
        Dict with token usage info:
        - model: Model name used
        - completion_tokens: Total completion tokens
        - prompt_tokens: Input tokens
        - total_tokens: Total tokens used
        - reasoning_tokens: Reasoning/thinking tokens (if available)
        - cached_tokens: Cached prompt tokens (if available)
        - cost: API cost (if available, OpenRouter)
        - provider: Provider name (if available, OpenRouter)
        - has_reasoning_content: Whether message contains reasoning field

    Example:
        completion = await server.chat_completion(messages=messages)
        usage = get_reasoning_token_usage(completion)
        if config.full_debug:
            print(f"  Reasoning tokens: {usage.get('reasoning_tokens', 'N/A')}")
    """
    result = {
        "model": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "cost": None,
        "provider": None,
        "has_reasoning_content": False,
    }

    if not completion:
        return result

    # Extract model name
    if hasattr(completion, "model"):
        result["model"] = completion.model

    # Extract provider (OpenRouter includes this)
    if hasattr(completion, "provider"):
        result["provider"] = completion.provider

    # Check if message has reasoning content
    if hasattr(completion, "choices") and completion.choices:
        msg = (
            completion.choices[0].message
            if hasattr(completion.choices[0], "message")
            else None
        )
        if msg:
            # Check for reasoning field (OpenRouter normalized field)
            if hasattr(msg, "reasoning") and msg.reasoning:
                result["has_reasoning_content"] = True
            # Check for reasoning_details (OpenRouter)
            elif hasattr(msg, "reasoning_details") and msg.reasoning_details:
                result["has_reasoning_content"] = True

    # Extract usage info
    if not hasattr(completion, "usage") or not completion.usage:
        return result

    usage = completion.usage

    result["completion_tokens"] = getattr(usage, "completion_tokens", None)
    result["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
    result["total_tokens"] = getattr(usage, "total_tokens", None)

    # Extract cost (OpenRouter includes this)
    if hasattr(usage, "cost"):
        result["cost"] = usage.cost

    # Extract reasoning tokens from completion_tokens_details
    # This works for: OpenAI, OpenRouter (Claude, Hermes, DeepSeek, etc.)
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        if hasattr(details, "reasoning_tokens"):
            result["reasoning_tokens"] = details.reasoning_tokens

    # Extract cached tokens from prompt_tokens_details (OpenRouter/OpenAI)
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        if hasattr(details, "cached_tokens"):
            result["cached_tokens"] = details.cached_tokens

    return result


def format_reasoning_debug_info(
    completion: Any, reasoning_content: Optional[str] = None
) -> str:
    """
    Format reasoning debug information for logging.

    Use this in evals when full_debug is enabled to show reasoning token usage.

    Args:
        completion: The ChatCompletion response object
        reasoning_content: Optional pre-extracted reasoning content

    Returns:
        Formatted string with reasoning debug info

    Example:
        if self.config.full_debug:
            print(format_reasoning_debug_info(completion))
    """
    usage = get_reasoning_token_usage(completion)

    lines = ["  [Reasoning/Token Debug Info]"]

    # Model and provider info
    if usage["model"]:
        lines.append(f"    Model: {usage['model']}")
    if usage["provider"]:
        lines.append(f"    Provider: {usage['provider']}")

    # Token counts
    if usage["prompt_tokens"] is not None:
        prompt_info = f"    Prompt tokens: {usage['prompt_tokens']}"
        if usage["cached_tokens"]:
            prompt_info += f" (cached: {usage['cached_tokens']})"
        lines.append(prompt_info)

    if usage["completion_tokens"] is not None:
        lines.append(f"    Completion tokens: {usage['completion_tokens']}")

    # Reasoning-specific info
    if usage["reasoning_tokens"] is not None:
        lines.append(f"    Reasoning tokens: {usage['reasoning_tokens']}")
        if usage["completion_tokens"] and usage["completion_tokens"] > 0:
            pct = (usage["reasoning_tokens"] / usage["completion_tokens"]) * 100
            lines.append(f"    Reasoning %: {pct:.1f}%")

    if usage["has_reasoning_content"]:
        lines.append("    Has reasoning content: Yes")

    # Cost info
    if usage["cost"] is not None:
        lines.append(f"    Cost: ${usage['cost']:.6f}")

    # Total
    if usage["total_tokens"] is not None:
        lines.append(f"    Total tokens: {usage['total_tokens']}")

    # Reasoning content length if provided
    if reasoning_content:
        lines.append(f"    Reasoning content length: {len(reasoning_content)} chars")

    return "\n".join(lines)


# Fallback regex patterns for MCQA when answer tags don't work
def build_mcqa_fallback_patterns(num_choices: int = 4):
    """
    Build fallback regex patterns for extracting MCQA answers.

    These are used when <answer> tags are not present or ambiguous.
    Patterns are ordered by priority (lower number = higher priority).

    Args:
        num_choices: Number of valid choices (determines valid letters)

    Returns:
        List of (priority, pattern, method_name) tuples
    """
    letters = ascii_uppercase[:num_choices]
    letter_pattern = rf"([{letters}]|\([{letters}]\))"

    patterns = [
        # Priority 0: "final answer is: X" with "I hope"
        (
            0,
            re.compile(
                rf"(?i:final\s+answer\s+is)\s*:?\s*{letter_pattern}\.?\s*I\s*hope",
                re.IGNORECASE,
            ),
            "final_answer_hope",
        ),
        # Priority 50: "final answer ... is X"
        (
            50,
            re.compile(
                rf"(?i:final\s+answer).{{0,100}}?\s+is\s*:?\s*{letter_pattern}",
                re.IGNORECASE | re.DOTALL,
            ),
            "final_answer_is",
        ),
        # Priority 75: "the answer is X"
        (
            75,
            re.compile(
                rf"(?i:the\s+answer\s+is)\s*:?\s*{letter_pattern}", re.IGNORECASE
            ),
            "the_answer_is",
        ),
        # Priority 100: "answer: X"
        (
            100,
            re.compile(
                rf"(?i:answer)\s*:\s*.{{0,50}}?{letter_pattern}",
                re.IGNORECASE | re.DOTALL,
            ),
            "answer_colon",
        ),
        # Priority 150: "answer X"
        (
            150,
            re.compile(rf"(?i:answer)\s+{letter_pattern}", re.IGNORECASE),
            "answer_space",
        ),
        # Priority 200: Response starts with letter
        (
            200,
            re.compile(rf"^\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE),
            "start",
        ),
        # Priority 210: Letter at start of any line
        (
            210,
            re.compile(rf"\n\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE),
            "line_start",
        ),
        # Priority 250: Standalone letter with word boundaries
        (250, re.compile(rf"\b{letter_pattern}\b", re.IGNORECASE), "standalone"),
    ]

    return patterns


def extract_mcqa_answer_with_fallback(
    response: str,
    num_choices: int = 4,
    fallback_patterns: list = None,
    debug: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Extract MCQA answer using answer tags first, then fallback patterns.

    Args:
        response: The model's response (content after </think> in thinking mode)
        num_choices: Number of valid choices
        fallback_patterns: Pre-built fallback patterns (optional, will be built if not provided)
        debug: Whether to print debug information

    Returns:
        Tuple of (extracted_letter or None, extraction_method)
    """
    if not response:
        return None, "empty_response"

    valid_letters = set(ascii_uppercase[:num_choices])

    # PRIMARY: Try <answer></answer> tags first
    letter, method = extract_letter_from_answer_tag(response, valid_letters, debug)
    if letter:
        return letter, method

    # FALLBACK: Use regex patterns
    if fallback_patterns is None:
        fallback_patterns = build_mcqa_fallback_patterns(num_choices)

    for priority, pattern, method_name in fallback_patterns:
        matches = pattern.findall(response)
        if matches:
            # Get the last match for answer patterns (final answer is most reliable)
            match = (
                matches[-1]
                if method_name
                in ["final_answer_is", "the_answer_is", "answer_colon", "answer_space"]
                else matches[0]
            )

            if isinstance(match, tuple):
                match = match[0]
            letter = match.strip("()").upper()

            if letter in valid_letters:
                if debug:
                    print(
                        f"    Extracted '{letter}' using fallback method '{method_name}' (priority {priority})"
                    )
                return letter, f"fallback_{method_name}"

    # Last resort: find any valid letter (take the last one)
    for letter in reversed(list(valid_letters)):
        if letter in response.upper():
            if debug:
                print(f"    Extracted '{letter}' using fallback 'last_valid_letter'")
            return letter, "fallback_last_valid_letter"

    return None, "no_match"


# =============================================================================
# MATH ANSWER VERIFICATION HELPERS
# =============================================================================
# These functions use the math_verify library for robust mathematical answer
# verification. They support \boxed{} extraction, symbolic comparison, and
# string normalization fallback.

# Regex for extracting \boxed{} content
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# Global ProcessPoolExecutor for math verification (avoids timeouts)
_math_executor: Optional[ProcessPoolExecutor] = None


def get_math_executor(max_workers: int = 64) -> ProcessPoolExecutor:
    """
    Get or create the global ProcessPoolExecutor for math verification.

    Using a process pool protects against hangs from sympy/latex parsing.

    Args:
        max_workers: Maximum number of worker processes

    Returns:
        ProcessPoolExecutor instance
    """
    global _math_executor
    if _math_executor is None:
        _math_executor = ProcessPoolExecutor(max_workers=max_workers)
    return _math_executor


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract all \\boxed{} answers from text.

    Args:
        text: The text to search for boxed answers

    Returns:
        List of extracted boxed contents
    """
    return BOXED_PATTERN.findall(text)


def extract_first_boxed_answer(
    response: str, after_think: bool = True, debug: bool = False
) -> Tuple[Optional[str], str, bool]:
    """
    Extract the first \\boxed{} answer from a response.

    Follows the rule: only accept if there's exactly ONE boxed answer
    after the reasoning tags (if thinking mode). Multiple boxed answers = failure.

    Supports both <think></think> and <|start_of_scratchpad|><|end_of_scratchpad|> formats.

    Args:
        response: The model's full response
        after_think: Whether to only look after reasoning tags
        debug: Whether to print debug information

    Returns:
        Tuple of (extracted_answer or None, extraction_method, has_multiple_boxed)
    """
    # Get content to search
    if after_think:
        # Try to extract content after </think> first
        match = THINK_CONTENT_AFTER_PATTERN.search(response)
        if match:
            search_content = match.group(1)
        else:
            # Try <|end_of_scratchpad|> tags
            match = SCRATCHPAD_CONTENT_AFTER_PATTERN.search(response)
            if match:
                search_content = match.group(1)
            else:
                # No reasoning tags, use full response
                search_content = response
    else:
        search_content = response

    # Find all boxed answers
    boxed_answers = extract_boxed_answers(search_content)

    if len(boxed_answers) == 0:
        if debug:
            print("    No \\boxed{} found in response")
        return None, "no_boxed", False

    if len(boxed_answers) > 1:
        if debug:
            print(f"    Multiple \\boxed{{}} found ({len(boxed_answers)}) - rejecting")
        return None, "multiple_boxed", True

    # Exactly one boxed answer
    answer = boxed_answers[0].strip()
    if debug:
        preview = answer[:50] + "..." if len(answer) > 50 else answer
        print(f"    Extracted '{preview}' from \\boxed{{}}")

    return answer, "boxed", False


def math_normalize_string(text: str) -> str:
    """
    Normalize a math answer string for comparison.

    This is a fallback when symbolic verification fails.
    Based on lighteval's math_normalizer.

    Args:
        text: The text to normalize

    Returns:
        Normalized string
    """
    if not text:
        return ""

    # Remove outer whitespace
    text = text.strip()

    # Remove \boxed{} wrapper if present
    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        text = boxed_match.group(1)

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove common LaTeX commands that don't affect value
    text = re.sub(r"\\[,;:!]", "", text)  # \, \; etc.
    text = re.sub(r"\\quad|\\qquad", " ", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)  # \text{...} -> ...
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)

    # Normalize math operators
    text = text.replace("\\times", "*")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\div", "/")
    text = text.replace("\\pm", "+-")

    # Remove $ signs
    text = text.replace("$", "")

    # Remove trailing punctuation
    text = text.rstrip(".,;:")

    # Lowercase for comparison
    text = text.lower()

    return text.strip()


def compare_math_strings(pred: str, gold: str) -> bool:
    """
    Compare two math answer strings after normalization.

    This is a fallback when symbolic verification fails.

    Args:
        pred: Predicted answer
        gold: Gold answer

    Returns:
        True if answers match, False otherwise
    """
    pred_norm = math_normalize_string(pred)
    gold_norm = math_normalize_string(gold)

    if not pred_norm:
        return False

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Try numeric comparison
    try:
        # Handle commas in numbers
        pred_clean = pred_norm.replace(",", "").replace(" ", "")
        gold_clean = gold_norm.replace(",", "").replace(" ", "")

        pred_num = float(pred_clean)
        gold_num = float(gold_clean)

        # Exact numeric match
        if pred_num == gold_num:
            return True

        # Small relative error (for floating point)
        if gold_num != 0 and abs(pred_num - gold_num) / abs(gold_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Try integer comparison (for AIME-style 0-999)
    try:
        pred_int = int(float(pred_norm.replace(",", "")))
        gold_int = int(float(gold_norm.replace(",", "")))
        if pred_int == gold_int:
            return True
    except (ValueError, TypeError):
        pass

    return False


def _score_math_answer_worker(
    gold: str, response: str, wrap_gold_boxed: bool = True
) -> Tuple[Optional[bool], str]:
    """
    Worker function for scoring math answers (runs in separate process).

    This function is designed to run in a ProcessPoolExecutor to protect
    against hangs from sympy/latex parsing.

    Args:
        gold: The gold answer
        response: The model's response (content to extract answer from)
        wrap_gold_boxed: Whether to wrap gold in \\boxed{} if not present

    Returns:
        Tuple of (is_correct or None, method_used)
    """
    if not MATH_VERIFY_AVAILABLE:
        # Fallback to string comparison
        boxed = extract_boxed_answers(response)
        if boxed:
            return compare_math_strings(boxed[0], gold), "string_fallback_no_lib"
        return None, "no_math_verify"

    try:
        # Prepare gold answer
        if wrap_gold_boxed and "\\boxed" not in gold:
            gold_text = f"\\boxed{{{gold}}}"
        else:
            gold_text = gold

        # Parse gold
        gold_parsed = parse(
            gold_text,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) == 0:
            # Gold couldn't be parsed, try string comparison
            boxed = extract_boxed_answers(response)
            if boxed:
                return (
                    compare_math_strings(boxed[0], gold),
                    "string_fallback_gold_parse",
                )
            return None, "gold_parse_failed"

        # Parse response
        response_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        if len(response_parsed) == 0:
            # Response couldn't be parsed, try string comparison
            boxed = extract_boxed_answers(response)
            if boxed:
                return (
                    compare_math_strings(boxed[0], gold),
                    "string_fallback_response_parse",
                )
            return None, "response_parse_failed"

        # Verify match
        is_correct = verify(response_parsed, gold_parsed)
        return is_correct, "math_verify"

    except TimeoutException:
        # Timeout during parsing/verification, try string comparison
        boxed = extract_boxed_answers(response)
        if boxed:
            return compare_math_strings(boxed[0], gold), "string_fallback_timeout"
        return None, "timeout"
    except Exception as e:
        # Any other error, try string comparison
        boxed = extract_boxed_answers(response)
        if boxed:
            return compare_math_strings(boxed[0], gold), "string_fallback_error"
        return None, f"error_{type(e).__name__}"


def score_math_answer(
    gold: str,
    response: str,
    after_think: bool = True,
    wrap_gold_boxed: bool = True,
    executor: Optional[ProcessPoolExecutor] = None,
    debug: bool = False,
) -> Tuple[Optional[bool], str, bool]:
    """
    Score a math answer using math_verify with process isolation.

    This is the main function for scoring math answers. It:
    1. Extracts content after </think> if thinking mode
    2. Checks for multiple \\boxed{} (fails if multiple)
    3. Uses math_verify for symbolic comparison
    4. Falls back to string normalization if that fails

    Args:
        gold: The gold answer
        response: The model's full response
        after_think: Whether to extract content after </think>
        wrap_gold_boxed: Whether to wrap gold in \\boxed{} if not present
        executor: Optional ProcessPoolExecutor to use
        debug: Whether to print debug information

    Returns:
        Tuple of (is_correct or None, method_used, has_multiple_boxed)
    """
    # Get content to score (check for both think and scratchpad tags)
    if after_think:
        match = THINK_CONTENT_AFTER_PATTERN.search(response)
        if match:
            score_content = match.group(1)
        else:
            # Try scratchpad tags
            match = SCRATCHPAD_CONTENT_AFTER_PATTERN.search(response)
            if match:
                score_content = match.group(1)
            else:
                score_content = response
    else:
        score_content = response

    # Check for multiple boxed answers
    boxed_answers = extract_boxed_answers(score_content)
    if len(boxed_answers) > 1:
        if debug:
            print(f"    Multiple \\boxed{{}} found ({len(boxed_answers)}) - rejecting")
        return None, "multiple_boxed", True

    if len(boxed_answers) == 0:
        if debug:
            print("    No \\boxed{} found")
        return None, "no_boxed", False

    # Use executor if provided, otherwise run directly
    if executor is not None:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                executor,
                _score_math_answer_worker,
                gold,
                score_content,
                wrap_gold_boxed,
            )
            # Note: This needs to be awaited in async context
            # For sync usage, call _score_math_answer_worker directly
            is_correct, method = future.result(timeout=30)
        except Exception as e:
            if debug:
                print(f"    Executor error: {e}")
            # Fallback to string comparison
            if boxed_answers:
                is_correct = compare_math_strings(boxed_answers[0], gold)
                method = "string_fallback_executor_error"
            else:
                return None, f"executor_error_{type(e).__name__}", False
    else:
        is_correct, method = _score_math_answer_worker(
            gold, score_content, wrap_gold_boxed
        )

    if debug:
        print(f"    Score: {is_correct} (method: {method})")

    return is_correct, method, False


async def score_math_answer_async(
    gold: str,
    response: str,
    after_think: bool = True,
    wrap_gold_boxed: bool = True,
    executor: Optional[ProcessPoolExecutor] = None,
    debug: bool = False,
) -> Tuple[Optional[bool], str, bool]:
    """
    Async version of score_math_answer for use in async evaluation loops.

    Uses ProcessPoolExecutor to run math verification in separate process,
    protecting against hangs.

    Args:
        gold: The gold answer
        response: The model's full response
        after_think: Whether to extract content after </think>
        wrap_gold_boxed: Whether to wrap gold in \\boxed{} if not present
        executor: Optional ProcessPoolExecutor to use
        debug: Whether to print debug information

    Returns:
        Tuple of (is_correct or None, method_used, has_multiple_boxed)
    """
    import asyncio

    # Get content to score (check for both think and scratchpad tags)
    if after_think:
        match = THINK_CONTENT_AFTER_PATTERN.search(response)
        if match:
            score_content = match.group(1)
        else:
            # Try scratchpad tags
            match = SCRATCHPAD_CONTENT_AFTER_PATTERN.search(response)
            if match:
                score_content = match.group(1)
            else:
                score_content = response
    else:
        score_content = response

    # Check for multiple boxed answers
    boxed_answers = extract_boxed_answers(score_content)
    if len(boxed_answers) > 1:
        if debug:
            print(f"    Multiple \\boxed{{}} found ({len(boxed_answers)}) - rejecting")
        return None, "multiple_boxed", True

    if len(boxed_answers) == 0:
        if debug:
            print("    No \\boxed{} found")
        return None, "no_boxed", False

    # Get executor
    if executor is None:
        executor = get_math_executor()

    try:
        loop = asyncio.get_event_loop()
        is_correct, method = await loop.run_in_executor(
            executor, _score_math_answer_worker, gold, score_content, wrap_gold_boxed
        )
    except Exception as e:
        if debug:
            print(f"    Executor error: {e}")
        # Fallback to string comparison
        if boxed_answers:
            is_correct = compare_math_strings(boxed_answers[0], gold)
            method = "string_fallback_executor_error"
        else:
            return None, f"executor_error_{type(e).__name__}", False

    if debug:
        print(f"    Score: {is_correct} (method: {method})")

    return is_correct, method, False


def format_math_answer_instruction(include_hope: bool = True) -> str:
    """
    Get the standard instruction for math answer format.

    Based on lighteval's AIME prompt which works well with math_verify.

    Args:
        include_hope: Whether to include "I hope it is correct" suffix

    Returns:
        Instruction string
    """
    if include_hope:
        return (
            "The last line of your response should be of the following format: "
            "'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct' "
            "(without quotes) where ANSWER is just the final number or expression that solves the problem."
        )
    else:
        return (
            "Put your final answer in \\boxed{} format. "
            "For example: \\boxed{42} or \\boxed{\\frac{1}{2}}"
        )


# =============================================================================
# SYSTEM PROMPT AND CONFIGURATION HELPERS
# =============================================================================
# These functions handle common system prompt creation patterns used across
# evaluation environments with thinking mode support.


def create_system_content(
    thinking_mode: bool,
    custom_thinking_prompt: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Create system message content based on thinking mode configuration.

    This is the standard pattern used across all eval environments:
    - In thinking mode: thinking_prompt + optional system_prompt
    - In non-thinking mode: just the system_prompt (or None)

    Args:
        thinking_mode: Whether thinking mode is enabled
        custom_thinking_prompt: Optional custom thinking prompt (uses default if None)
        custom_system_prompt: Optional additional system prompt

    Returns:
        System content string, or None if no content needed
    """
    if thinking_mode:
        thinking_prompt = get_default_thinking_prompt(custom_thinking_prompt)
        if custom_system_prompt:
            return f"{thinking_prompt}\n\n{custom_system_prompt}"
        return thinking_prompt
    return custom_system_prompt


# =============================================================================
# RESULTS SAVING UTILITIES
# =============================================================================
# Common patterns for saving evaluation results to disk.


def save_eval_results(
    save_dir: str,
    metrics: Dict,
    results: List[Dict],
    metrics_filename: str = "metrics.json",
    results_filename: str = "results.jsonl",
    print_confirmation: bool = True,
) -> Tuple[str, str]:
    """
    Save evaluation results to disk in standard format.

    Creates two files:
    - metrics.json: Summary metrics dict
    - results.jsonl: Per-item results (one JSON object per line)

    Args:
        save_dir: Directory to save results to (created if doesn't exist)
        metrics: Dictionary of evaluation metrics
        results: List of per-item result dictionaries
        metrics_filename: Name for metrics file
        results_filename: Name for results file
        print_confirmation: Whether to print confirmation message

    Returns:
        Tuple of (metrics_path, results_path)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(save_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save detailed results
    results_path = os.path.join(save_dir, results_filename)
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    if print_confirmation:
        print(f"Results saved to {save_dir}")

    return metrics_path, results_path


def load_eval_results(
    save_dir: str,
    metrics_filename: str = "metrics.json",
    results_filename: str = "results.jsonl",
) -> Tuple[Dict, List[Dict]]:
    """
    Load evaluation results from disk.

    Args:
        save_dir: Directory containing results
        metrics_filename: Name of metrics file
        results_filename: Name of results file

    Returns:
        Tuple of (metrics dict, list of result dicts)
    """
    # Load metrics
    metrics_path = os.path.join(save_dir, metrics_filename)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Load detailed results
    results_path = os.path.join(save_dir, results_filename)
    results = []
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return metrics, results


# =============================================================================
# COMMON EVALUATION UTILITIES
# =============================================================================
# Helper functions used across multiple eval environments.


def calculate_accuracy(
    results: List[Dict],
    score_key: str = "is_correct",
    filter_fn: Optional[callable] = None,
) -> float:
    """
    Calculate accuracy from a list of result dictionaries.

    Args:
        results: List of result dictionaries
        score_key: Key to look up score/correctness (should be bool or 0/1)
        filter_fn: Optional function to filter results before calculation

    Returns:
        Accuracy as float between 0 and 1
    """
    if filter_fn:
        results = [r for r in results if filter_fn(r)]

    if not results:
        return 0.0

    correct = sum(1 for r in results if r.get(score_key, False))
    return correct / len(results)


def group_results_by_key(results: List[Dict], key: str) -> Dict[str, List[Dict]]:
    """
    Group results by a specific key value.

    Useful for computing per-category or per-subset metrics.

    Args:
        results: List of result dictionaries
        key: Key to group by

    Returns:
        Dictionary mapping key values to lists of results
    """
    grouped = {}
    for r in results:
        value = r.get(key, "unknown")
        if value not in grouped:
            grouped[value] = []
        grouped[value].append(r)
    return grouped


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.

    Args:
        value: Float value (0-1 scale)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "75.50%")
    """
    return f"{value * 100:.{decimals}f}%"


def print_eval_summary(title: str, metrics: Dict[str, float], width: int = 60) -> None:
    """
    Print a formatted evaluation summary.

    Args:
        title: Summary title
        metrics: Dictionary of metric name -> value
        width: Width of separator lines
    """
    print(f"\n{'='*width}")
    print(title)
    print(f"{'='*width}")
    for name, value in metrics.items():
        if isinstance(value, float) and 0 <= value <= 1:
            print(f"  {name}: {format_percentage(value)}")
        else:
            print(f"  {name}: {value}")
    print(f"{'='*width}\n")
