"""
Scoring helpers for the Solidity Smart Contract Audit environment.

Provides multi-component reward calculation for vulnerability detection:
- Vulnerability detection (binary match)
- Category matching (fuzzy string)
- Description quality (keyword Jaccard similarity)
- Format compliance (valid YAML, all fields present, boxed format)
"""

import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

import yaml

REQUIRED_FIELDS = {"vulnerable", "category", "description", "fix"}

WEIGHT_VULNERABLE = 0.25
WEIGHT_CATEGORY = 0.35
WEIGHT_DESCRIPTION = 0.25
WEIGHT_FORMAT = 0.15


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} in the response text.

    Handles nested braces by counting brace depth.

    Returns:
        The inner content string, or None if no valid boxed block found.
    """
    match = re.search(r"\\boxed\s*\{", text)
    if not match:
        return None

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[start : i - 1].strip()


def parse_audit_yaml(yaml_str: str) -> Optional[Dict]:
    """Parse YAML string into a dict, returning None on failure."""
    try:
        result = yaml.safe_load(yaml_str)
        if isinstance(result, dict):
            return result
        return None
    except (yaml.YAMLError, ValueError):
        return None


def extract_audit_response(text: str) -> Tuple[Optional[Dict], bool]:
    """Extract and parse the audit response from model output.

    Returns:
        Tuple of (parsed_dict or None, boxed_found: bool)
    """
    boxed = extract_boxed_content(text)
    if boxed is None:
        return None, False

    parsed = parse_audit_yaml(boxed)
    return parsed, True


def normalize_bool(value) -> Optional[bool]:
    """Normalize a value to boolean. Handles strings like 'true', 'yes', etc."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def score_vulnerability_detection(predicted: Dict, actual_vulnerable: bool) -> float:
    """Score whether the model correctly identified if the code is vulnerable.

    Returns:
        1.0 for correct detection, 0.0 for incorrect.
    """
    pred_value = normalize_bool(predicted.get("vulnerable"))
    if pred_value is None:
        return 0.0
    return 1.0 if pred_value == actual_vulnerable else 0.0


def score_category_match(predicted_category: str, actual_category: str) -> float:
    """Score category match using fuzzy string similarity.

    Uses SequenceMatcher ratio for fuzzy matching.

    Returns:
        Float between 0.0 and 1.0.
    """
    pred = predicted_category.strip().lower().replace(" ", "_").replace("-", "_")
    actual = actual_category.strip().lower().replace(" ", "_").replace("-", "_")

    if pred == actual:
        return 1.0

    return SequenceMatcher(None, pred, actual).ratio()


def _tokenize(text: str) -> set:
    """Tokenize text into a set of lowercase words."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_description_quality(predicted_desc: str, actual_desc: str) -> float:
    """Score description quality using keyword Jaccard similarity.

    Returns:
        Float between 0.0 and 1.0.
    """
    pred_tokens = _tokenize(predicted_desc)
    actual_tokens = _tokenize(actual_desc)

    if not pred_tokens and not actual_tokens:
        return 1.0
    if not pred_tokens or not actual_tokens:
        return 0.0

    intersection = pred_tokens & actual_tokens
    union = pred_tokens | actual_tokens
    return len(intersection) / len(union)


def score_format_compliance(text: str) -> float:
    """Score format compliance of the response.

    Checks:
    - boxed block present (0.4 of format score)
    - Valid YAML inside boxed (0.3 of format score)
    - All required fields present (0.3 of format score)

    Returns:
        Float between 0.0 and 1.0.
    """
    score = 0.0

    boxed = extract_boxed_content(text)
    if boxed is None:
        return 0.0
    score += 0.4

    parsed = parse_audit_yaml(boxed)
    if parsed is None:
        return score
    score += 0.3

    present_fields = set(parsed.keys()) & REQUIRED_FIELDS
    field_ratio = len(present_fields) / len(REQUIRED_FIELDS)
    score += 0.3 * field_ratio

    return score


def compute_total_reward(
    predicted: Optional[Dict],
    actual_vulnerable: bool,
    actual_category: str,
    actual_description: str,
    raw_response: str,
) -> float:
    """Compute the weighted total reward for a single response.

    Components:
    - vulnerable match:     0.25
    - category match:       0.35
    - description quality:  0.25
    - format compliance:    0.15

    Returns:
        Float between 0.0 and 1.0.
    """
    format_score = score_format_compliance(raw_response)

    if predicted is None:
        return WEIGHT_FORMAT * format_score

    vuln_score = score_vulnerability_detection(predicted, actual_vulnerable)

    pred_category = str(predicted.get("category", ""))
    cat_score = score_category_match(pred_category, actual_category)

    pred_desc = str(predicted.get("description", ""))
    desc_score = score_description_quality(pred_desc, actual_description)

    total = (
        WEIGHT_VULNERABLE * vuln_score
        + WEIGHT_CATEGORY * cat_score
        + WEIGHT_DESCRIPTION * desc_score
        + WEIGHT_FORMAT * format_score
    )

    return total
