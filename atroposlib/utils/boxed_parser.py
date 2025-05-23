import logging
import re
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse
from math_verify.errors import TimeoutException

logger = logging.getLogger(__name__)

def _simple_boxed_extraction(text: str) -> Optional[str]:
    """Simple regex-based fallback for extracting \boxed{} content."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extracts the content from the first \\boxed{} expression found in the text.

    Args:
        text: The input string to parse.

    Returns:
        The content within the \boxed{} expression as a string, or None if 
        not found, parsing fails, or the content is not a simple string.
    """
    if not text or "\\boxed{" not in text:
        logger.debug(f"No \\boxed{{}} found in text: {text}")
        return None

    try:
        extraction_config = LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                equations=False,
                boxed="all",
                units=False,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False
        )
        
        try:
            parsed_elements = parse(
                text,
                extraction_config=[extraction_config],
                extraction_mode="first_match"
            )
        except ValueError as e:
            if "signal only works in main thread" in str(e):
                logger.debug(f"Signal-based timeout not available in current thread, using simple fallback parser")
                return _simple_boxed_extraction(text)
            else:
                raise

        if parsed_elements:
            extracted_item = parsed_elements[0]
            if isinstance(extracted_item, (str, int, float)):
                return str(extracted_item)
            elif isinstance(extracted_item, list) and len(extracted_item) == 1:
                return str(extracted_item[0])
            else:
                logger.warning(f"Boxed content parsed into unexpected type: {type(extracted_item)}. Value: {extracted_item}")
                return str(extracted_item) 

        return None

    except TimeoutException:
        logger.error("Timeout during parsing for boxed content.")
        return None
    except Exception as e:
        logger.error(f"Error parsing for boxed content: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_cases = [
        ("Some text with \\boxed{123.45} and more.", "123.45"),
        ("No boxed content here.", None),
        ("\\boxed{-0.5} is the answer.", "-0.5"),
        ("Blah \\boxed{text here} blah", "text here"),
        ("Leading text \\boxed{  spaced value  } trailing", "spaced value"),
        ("No box", None),
        ("Multiple: \\boxed{first} then \\boxed{second}", "first"),
        ("Invalid \\boxed{", None),
        ("Escaped like in prompts: \\\\boxed{9.2}", "9.2"),
        ("Text with \\boxed{True} value", "True"),
    ]

    for text, expected in test_cases:
        result = extract_boxed_content(text)
        print(f"Input: {repr(text)}, Expected: {expected}, Got: {result}, Match: {result == expected}")
        assert result == expected, f"Failed for {text}"
    
    print("\nAll tests passed!")
