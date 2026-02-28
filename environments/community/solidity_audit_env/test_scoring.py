"""
Unit tests for the Solidity audit scoring module.
"""

import pytest
from scoring import (
    compute_total_reward,
    extract_audit_response,
    extract_boxed_content,
    normalize_bool,
    parse_audit_yaml,
    score_category_match,
    score_description_quality,
    score_format_compliance,
    score_vulnerability_detection,
)

# --- extract_boxed_content ---


class TestExtractBoxedContent:
    def test_valid_boxed(self):
        text = r"Some text \boxed{hello world} more text"
        assert extract_boxed_content(text) == "hello world"

    def test_boxed_with_yaml(self):
        text = r"""\boxed{
vulnerable: true
category: reentrancy
description: "The function is vulnerable"
fix: "Use checks-effects-interactions"
}"""
        result = extract_boxed_content(text)
        assert result is not None
        assert "vulnerable: true" in result

    def test_nested_braces(self):
        text = r"\boxed{outer {inner} end}"
        assert extract_boxed_content(text) == "outer {inner} end"

    def test_no_boxed(self):
        text = "No boxed content here"
        assert extract_boxed_content(text) is None

    def test_unclosed_boxed(self):
        text = r"\boxed{unclosed"
        assert extract_boxed_content(text) is None

    def test_empty_boxed(self):
        text = r"\boxed{}"
        assert extract_boxed_content(text) == ""


# --- parse_audit_yaml ---


class TestParseAuditYaml:
    def test_valid_yaml(self):
        yaml_str = """vulnerable: true
category: reentrancy
description: "Test description"
fix: "Test fix"
"""
        result = parse_audit_yaml(yaml_str)
        assert result is not None
        assert result["vulnerable"] is True
        assert result["category"] == "reentrancy"

    def test_invalid_yaml(self):
        yaml_str = "{{{{not yaml"
        assert parse_audit_yaml(yaml_str) is None

    def test_yaml_returns_string(self):
        yaml_str = "just a string"
        assert parse_audit_yaml(yaml_str) is None

    def test_empty_string(self):
        assert parse_audit_yaml("") is None


# --- extract_audit_response ---


class TestExtractAuditResponse:
    def test_valid_response(self):
        text = r"""\boxed{
vulnerable: true
category: reentrancy
description: "Vulnerable withdraw"
fix: "Use CEI pattern"
}"""
        parsed, found = extract_audit_response(text)
        assert found is True
        assert parsed is not None
        assert parsed["vulnerable"] is True
        assert parsed["category"] == "reentrancy"

    def test_no_boxed(self):
        parsed, found = extract_audit_response("No boxed here")
        assert found is False
        assert parsed is None

    def test_invalid_yaml_in_boxed(self):
        text = r"\boxed{: : : invalid}"
        parsed, found = extract_audit_response(text)
        assert found is True
        assert parsed is None


# --- normalize_bool ---


class TestNormalizeBool:
    def test_bool_true(self):
        assert normalize_bool(True) is True

    def test_bool_false(self):
        assert normalize_bool(False) is False

    def test_string_true(self):
        assert normalize_bool("true") is True
        assert normalize_bool("True") is True
        assert normalize_bool("yes") is True
        assert normalize_bool("YES") is True

    def test_string_false(self):
        assert normalize_bool("false") is False
        assert normalize_bool("no") is False

    def test_int(self):
        assert normalize_bool(1) is True
        assert normalize_bool(0) is False

    def test_invalid(self):
        assert normalize_bool("maybe") is None
        assert normalize_bool(None) is None


# --- score_vulnerability_detection ---


class TestScoreVulnerabilityDetection:
    def test_correct_vulnerable(self):
        score = score_vulnerability_detection({"vulnerable": True}, True)
        assert score == 1.0

    def test_correct_not_vulnerable(self):
        score = score_vulnerability_detection({"vulnerable": False}, False)
        assert score == 1.0

    def test_incorrect(self):
        score = score_vulnerability_detection({"vulnerable": True}, False)
        assert score == 0.0

    def test_missing_field(self):
        score = score_vulnerability_detection({}, True)
        assert score == 0.0

    def test_string_bool(self):
        score = score_vulnerability_detection({"vulnerable": "true"}, True)
        assert score == 1.0


# --- score_category_match ---


class TestScoreCategoryMatch:
    def test_exact_match(self):
        assert score_category_match("reentrancy", "reentrancy") == 1.0

    def test_case_insensitive(self):
        assert score_category_match("Reentrancy", "reentrancy") == 1.0

    def test_underscore_normalization(self):
        assert score_category_match("access control", "access_control") == 1.0

    def test_hyphen_normalization(self):
        assert score_category_match("access-control", "access_control") == 1.0

    def test_different_categories(self):
        score = score_category_match("reentrancy", "overflow")
        assert score < 0.5

    def test_similar_categories(self):
        score = score_category_match("reentrancy", "reentranc")
        assert score > 0.7


# --- score_description_quality ---


class TestScoreDescriptionQuality:
    def test_identical(self):
        desc = "The withdraw function calls external address before updating state"
        assert score_description_quality(desc, desc) == 1.0

    def test_partial_overlap(self):
        pred = "The function has a reentrancy vulnerability in the withdraw method"
        actual = (
            "The withdraw function calls an external address before updating balance"
        )
        score = score_description_quality(pred, actual)
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        pred = "alpha beta gamma"
        actual = "delta epsilon zeta"
        score = score_description_quality(pred, actual)
        assert score == 0.0

    def test_empty_both(self):
        assert score_description_quality("", "") == 1.0

    def test_empty_one(self):
        assert score_description_quality("", "something") == 0.0
        assert score_description_quality("something", "") == 0.0


# --- score_format_compliance ---


class TestScoreFormatCompliance:
    def test_perfect_format(self):
        text = r"""\boxed{
vulnerable: true
category: reentrancy
description: "test"
fix: "test fix"
}"""
        score = score_format_compliance(text)
        assert score == pytest.approx(1.0)

    def test_no_boxed(self):
        assert score_format_compliance("no boxed here") == 0.0

    def test_boxed_invalid_yaml(self):
        text = r"\boxed{: : : invalid}"
        score = score_format_compliance(text)
        assert score == pytest.approx(0.4)

    def test_boxed_valid_yaml_missing_fields(self):
        text = r"""\boxed{
vulnerable: true
category: reentrancy
}"""
        score = score_format_compliance(text)
        # boxed (0.4) + valid yaml (0.3) + 2/4 fields (0.3 * 0.5 = 0.15)
        assert score == pytest.approx(0.85)

    def test_boxed_valid_yaml_all_fields(self):
        text = r"""\boxed{
vulnerable: true
category: reentrancy
description: "d"
fix: "f"
}"""
        assert score_format_compliance(text) == pytest.approx(1.0)


# --- compute_total_reward ---


class TestComputeTotalReward:
    def test_perfect_response(self):
        predicted = {
            "vulnerable": True,
            "category": "reentrancy",
            "description": "The withdraw function calls external address before updating state",
            "fix": "Move state update before external call",
        }
        raw = r"""\boxed{
vulnerable: true
category: reentrancy
description: "The withdraw function calls external address before updating state"
fix: "Move state update before external call"
}"""
        reward = compute_total_reward(
            predicted=predicted,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="The withdraw function calls external address before updating state",
            raw_response=raw,
        )
        assert reward == pytest.approx(1.0)

    def test_no_parsed_response(self):
        reward = compute_total_reward(
            predicted=None,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="Test description",
            raw_response="no boxed content",
        )
        assert reward == 0.0

    def test_wrong_vulnerability_detection(self):
        predicted = {
            "vulnerable": False,
            "category": "reentrancy",
            "description": "The withdraw function calls external address before updating state",
            "fix": "Move state update before external call",
        }
        raw = r"""\boxed{
vulnerable: false
category: reentrancy
description: "The withdraw function calls external address before updating state"
fix: "Move state update before external call"
}"""
        reward = compute_total_reward(
            predicted=predicted,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="The withdraw function calls external address before updating state",
            raw_response=raw,
        )
        # vulnerable wrong (0.0 * 0.25), rest correct
        assert reward == pytest.approx(0.75)

    def test_wrong_category(self):
        predicted = {
            "vulnerable": True,
            "category": "overflow",
            "description": "The withdraw function calls external address before updating state",
            "fix": "Move state update",
        }
        raw = r"""\boxed{
vulnerable: true
category: overflow
description: "The withdraw function calls external address before updating state"
fix: "Move state update"
}"""
        reward = compute_total_reward(
            predicted=predicted,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="The withdraw function calls external address before updating state",
            raw_response=raw,
        )
        # category will have low fuzzy score
        assert reward < 0.9

    def test_empty_response(self):
        reward = compute_total_reward(
            predicted=None,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="Test",
            raw_response="",
        )
        assert reward == 0.0

    def test_partial_format_only(self):
        # Boxed with invalid YAML — only format partial score counts
        raw = r"\boxed{not valid yaml : : :}"
        reward = compute_total_reward(
            predicted=None,
            actual_vulnerable=True,
            actual_category="reentrancy",
            actual_description="Test",
            raw_response=raw,
        )
        # Only format score: 0.15 * 0.4 = 0.06
        assert reward == pytest.approx(0.15 * 0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
