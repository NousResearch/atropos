"""Tests for jsonl2html.py message format handling."""

import json


from atroposlib.frontend.jsonl2html import create_html_for_group, generate_html


class TestCreateHtmlForGroup:
    """Test create_html_for_group with different message formats."""

    def test_string_messages_format(self):
        """Test with original string messages format (backward compatibility)."""
        group_data = {
            "messages": ["Hello, world!", "This is a test message."],
            "scores": [0.5, 0.8],
        }

        html = create_html_for_group(group_data, 0)

        assert html is not None
        assert "Hello, world!" in html
        assert "This is a test message." in html
        assert "0.5" in html
        assert "0.8" in html

    def test_nested_messages_format(self):
        """Test with nested conversation format: List[List[Message]]."""
        group_data = {
            "messages": [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ],
                [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            ],
            "scores": [1.0, 0.9],
        }

        html = create_html_for_group(group_data, 0)

        assert html is not None
        # Check that content is rendered
        assert "You are a helpful assistant" in html
        assert "What is 2+2?" in html
        assert "2+2 equals 4" in html
        assert "Hello!" in html
        assert "Hi there!" in html
        # Check that roles are rendered
        assert "System" in html
        assert "User" in html
        assert "Assistant" in html

    def test_empty_messages(self):
        """Test with empty messages list."""
        group_data = {
            "messages": [],
            "scores": [],
        }

        html = create_html_for_group(group_data, 0)

        # Should return empty string for no items
        assert html == ""

    def test_mismatched_lengths(self):
        """Test with mismatched messages and scores lengths."""
        group_data = {
            "messages": ["Message 1", "Message 2", "Message 3"],
            "scores": [0.5],  # Only one score
        }

        # Should handle gracefully by using minimum length
        html = create_html_for_group(group_data, 0)

        assert "Message 1" in html
        assert "Message 2" not in html  # Should be skipped
        assert "Message 3" not in html  # Should be skipped


class TestGenerateHtml:
    """Test the full generate_html function."""

    def test_generate_html_with_nested_messages(self, tmp_path):
        """Test full HTML generation with nested message format."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"

        test_data = {
            "messages": [
                [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]
            ],
            "scores": [0.75],
        }

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(test_data) + "\n")

        # Generate HTML
        output_file = tmp_path / "test.html"
        generate_html(str(jsonl_file), str(output_file))

        # Verify output exists and contains expected content
        assert output_file.exists()
        html_content = output_file.read_text()
        assert "Test question" in html_content
        assert "Test answer" in html_content

    def test_generate_html_with_string_messages(self, tmp_path):
        """Test full HTML generation with string message format (backward compat)."""
        jsonl_file = tmp_path / "test_strings.jsonl"

        test_data = {
            "messages": ["Simple string message"],
            "scores": [0.5],
        }

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(test_data) + "\n")

        output_file = tmp_path / "test_strings.html"
        generate_html(str(jsonl_file), str(output_file))

        assert output_file.exists()
        html_content = output_file.read_text()
        assert "Simple string message" in html_content
