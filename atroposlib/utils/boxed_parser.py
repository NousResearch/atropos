import logging
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse
from math_verify.errors import TimeoutException

logger = logging.getLogger(__name__)

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extracts the content from the first \\boxed{} expression found in the text.

    Args:
        text: The input string to parse.

    Returns:
        The content within the \boxed{} expression as a string, or None if 
        not found, parsing fails, or the content is not a simple string.
    """
    if not text or "\\boxed{" not in text: # Quick check to avoid unnecessary parsing
        logger.debug(f"No \\boxed{{}} found in text: {text}")
        input("Press Enter to continue...")
        return None

    try:
        # Configure extraction to find boxed content.
        # NormalizationConfig can be tailored if specific math normalizations are needed,
        # but for simple string extraction, defaults are often fine.
        extraction_config = LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,          # Number In Text Substitution
                malformed_operators=False,
                basic_latex=True,    # Basic LaTeX normalizations
                equations=False,     # Don't interpret as equations unless needed
                boxed="all",         # Crucial: tells it to look for boxed content
                units=False,         # Disable unit processing unless needed
            ),
            boxed_match_priority=0, # Prioritize boxed matches
            try_extract_without_anchor=False # Avoids extracting non-boxed items if \boxed is missing
        )
        
        parsed_elements = parse(
            text,
            extraction_config=[extraction_config],
            extraction_mode="first_match" # We only care about the first valid boxed expression
        )

        # math_verify.parse returns a list of extracted elements.
        # If a boxed expression was found, it should be the first (and only, with first_match)
        # element, and it's usually a string if it was just text inside.
        if parsed_elements:
            extracted_item = parsed_elements[0]
            # The structure of extracted_item can vary. For simple \boxed{text},
            # it might be the string itself or a dict/object wrapping it.
            # We need to inspect how math_verify returns simple boxed strings.
            # Based on typical usage (like in accuracy_reward), it might be a string directly
            # or might require accessing a specific field if it's a more complex structure.
            # For \boxed{NUMBER}, it usually comes out as a number type.
            # For \boxed{TEXT}, it usually comes out as a string.
            
            # If it's a list of sympy expressions (which happens if it can be parsed as math)
            # we will just stringify it for now if it's simple.
            # For our Q-value purpose (a float), this should be sufficient as it's parsed later.
            if isinstance(extracted_item, (str, int, float)):
                return str(extracted_item)
            elif isinstance(extracted_item, list) and len(extracted_item) == 1:
                 # Potentially a list containing one sympy expression
                return str(extracted_item[0])
            else:
                logger.warning(f"Boxed content parsed into unexpected type: {type(extracted_item)}. Value: {extracted_item}")
                # Attempt to stringify it as a fallback, but this might not always be what's desired.
                return str(extracted_item) 

        return None # No boxed content found

    except TimeoutException:
        logger.error("Timeout during parsing for boxed content.")
        return None
    except Exception as e:
        logger.error(f"Error parsing for boxed content: {e}", exc_info=True)
        return None

# Example Usage (for testing):
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_cases = [
        ("Some text with \\boxed{123.45} and more.", "123.45"),
        ("No boxed content here.", None),
        ("\\boxed{-0.5} is the answer.", "-0.5"),
        ("Blah \\boxed{text here} blah", "text here"),
        ("Leading text \\boxed{  spaced value  } trailing", "spaced value"), # math_verify usually strips spaces
        ("No box", None),
        ("Multiple: \\boxed{first} then \\boxed{second}", "first"),
        ("Invalid \\boxed{", None), # Should be handled by parser
        ("Escaped like in prompts: \\\\boxed{9.2}", "9.2"), # Simulating prompt string literal
        ("Text with \\boxed{True} value", "True"),
    ]

    for text, expected in test_cases:
        # For __main__ test, simulate how it might look in a raw string literal if that's the source
        # The parser itself expects the literal single backslash version.
        # Python string literals: r"\boxed{}" is one backslash, "\\boxed{}" is one backslash.
        # The key is what `text` variable holds *at runtime*.
        
        # If text comes from an f-string like f"... \\boxed{{value}} ...", then `text` will contain `\boxed{value}`.
        # The test cases above are defined with Python string escaping, so `\\` becomes `\`.
        
        result = extract_boxed_content(text)
        print(f"Input: {repr(text)}, Expected: {expected}, Got: {result}, Match: {result == expected}")
        assert result == expected, f"Failed for {text}"
    
    print("\nAll tests passed!")
