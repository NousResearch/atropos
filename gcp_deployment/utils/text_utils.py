import re


def transform_npc_speech(text: str) -> str:
    """
    Transforms text to uppercase and removes all vowels (A, E, I, O, U).
    """
    if not isinstance(text, str):
        return ""
    # Convert to uppercase first
    text_upper = text.upper()
    # Remove vowels
    # re.sub is generally efficient for this.
    text_no_vowels = re.sub(r"[AEIOU]", "", text_upper)
    return text_no_vowels


if __name__ == "__main__":
    test_phrases = [
        "Hello World",
        "This is a test message for the LLM society.",
        "AEIOUaeiou",
        "Rhythm and blues!",
        "12345 XYZ",
        None,  # Test None input
        "",
    ]
    print("Testing NPC Speech Transformation:")
    for phrase in test_phrases:
        original = str(phrase) if phrase is not None else "None (literal)"
        transformed = transform_npc_speech(phrase)
        print("  Original: "{original}" -> Transformed: "{transformed}')
