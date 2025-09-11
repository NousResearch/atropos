import re


def extract_memory_block(text: str) -> str | None:
    """Extract the first <memory>...</memory> block content from the text.

    Returns None if no well-formed block is found.
    """
    if not text:
        return None
    m = re.search(r"<memory>(.*?)</memory>", text, flags=re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    return content if content else None


def validate_memory_content(content: str) -> bool:
    """Basic validation for memory content: non-empty and not excessively long."""
    if not content:
        return False
    # Reject extremely long memory blocks (> 2k chars) to avoid runaway outputs
    return len(content) <= 2000

