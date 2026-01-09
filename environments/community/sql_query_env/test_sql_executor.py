"""
Unit tests for SQL executor module.
"""

from sql_executor import (
    create_table_from_wikisql,
    execute_sql,
    extract_boxed_sql,
    normalize_result,
    results_match,
)


class TestCreateTableFromWikiSQL:
    """Tests for create_table_from_wikisql function."""

    def test_basic_table_creation(self):
        """Test creating a simple table."""
        header = ["name", "age", "city"]
        rows = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "Los Angeles"],
        ]

        conn = create_table_from_wikisql(header, rows)
        result = execute_sql(conn, "SELECT * FROM data")
        conn.close()

        assert result is not None
        assert len(result) == 2

    def test_table_with_special_column_names(self):
        """Test columns with spaces and special characters."""
        header = ["Player Name", "Goals-Scored", "Team (2024)"]
        rows = [["Messi", "10", "Inter Miami"]]

        conn = create_table_from_wikisql(header, rows)
        # Column names should be sanitized
        result = execute_sql(conn, "SELECT * FROM data")
        conn.close()

        assert result is not None
        assert len(result) == 1

    def test_empty_table(self):
        """Test creating a table with no rows."""
        header = ["col1", "col2"]
        rows = []

        conn = create_table_from_wikisql(header, rows)
        result = execute_sql(conn, "SELECT * FROM data")
        conn.close()

        assert result == []


class TestExecuteSQL:
    """Tests for execute_sql function."""

    def test_valid_select(self):
        """Test a valid SELECT query."""
        header = ["name", "age"]
        rows = [["Alice", "30"], ["Bob", "25"]]

        conn = create_table_from_wikisql(header, rows)
        result = execute_sql(conn, "SELECT name FROM data WHERE age = '30'")
        conn.close()

        assert result == [("Alice",)]

    def test_invalid_sql_returns_none(self):
        """Test that invalid SQL returns None."""
        header = ["a"]
        rows = [["1"]]

        conn = create_table_from_wikisql(header, rows)
        result = execute_sql(conn, "INVALID SQL SYNTAX")
        conn.close()

        assert result is None

    def test_aggregation(self):
        """Test COUNT aggregation."""
        header = ["category"]
        rows = [["A"], ["A"], ["B"]]

        conn = create_table_from_wikisql(header, rows)
        result = execute_sql(conn, "SELECT COUNT(*) FROM data WHERE category = 'A'")
        conn.close()

        assert result == [(2,)]


class TestExtractBoxedSQL:
    """Tests for extract_boxed_sql function."""

    def test_simple_boxed(self):
        """Test extracting from simple boxed format."""
        text = "The answer is \\boxed{SELECT * FROM data}"
        result = extract_boxed_sql(text)
        assert result == "SELECT * FROM data"

    def test_boxed_with_think_tags(self):
        """Test extracting when thinking tags are present."""
        text = """<think>
Let me think about this query...
</think>

\\boxed{SELECT name FROM data WHERE age > 25}"""
        result = extract_boxed_sql(text)
        assert result == "SELECT name FROM data WHERE age > 25"

    def test_no_boxed_returns_none(self):
        """Test that missing boxed returns None."""
        text = "SELECT * FROM data"
        result = extract_boxed_sql(text)
        assert result is None

    def test_empty_boxed(self):
        """Test empty boxed returns None."""
        text = "\\boxed{}"
        result = extract_boxed_sql(text)
        assert result is None


class TestResultsMatch:
    """Tests for results_match function."""

    def test_identical_results(self):
        """Test matching identical results."""
        result1 = [("Alice", "30"), ("Bob", "25")]
        result2 = [("Alice", "30"), ("Bob", "25")]
        assert results_match(result1, result2) is True

    def test_different_order(self):
        """Test matching results in different order."""
        result1 = [("Alice",), ("Bob",)]
        result2 = [("Bob",), ("Alice",)]
        assert results_match(result1, result2) is True

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        result1 = [("ALICE",)]
        result2 = [("alice",)]
        assert results_match(result1, result2) is True

    def test_different_results(self):
        """Test non-matching results."""
        result1 = [("Alice",)]
        result2 = [("Bob",)]
        assert results_match(result1, result2) is False

    def test_none_result(self):
        """Test that None results don't match."""
        assert results_match(None, [("a",)]) is False
        assert results_match([("a",)], None) is False


class TestNormalizeResult:
    """Tests for normalize_result function."""

    def test_normalization(self):
        """Test result normalization."""
        result = [("Alice", "30"), ("BOB", "25")]
        normalized = normalize_result(result)

        # Should be lowercase and sorted
        assert normalized == [("25", "bob"), ("30", "alice")] or normalized == [
            ("alice", "30"),
            ("bob", "25"),
        ]


class TestIntegration:
    """Integration tests for the full scoring pipeline."""

    def test_correct_sql_scores_1(self):
        """Test that matching SQL results score correctly."""
        header = ["name", "age"]
        rows = [["Alice", "30"], ["Bob", "25"]]

        conn = create_table_from_wikisql(header, rows)

        gold_sql = "SELECT name FROM data WHERE age = '30'"
        generated_sql = "SELECT name FROM data WHERE age = '30'"

        gold_result = execute_sql(conn, gold_sql)
        gen_result = execute_sql(conn, generated_sql)
        conn.close()

        assert results_match(gen_result, gold_result) is True

    def test_semantically_equivalent_sql(self):
        """Test that semantically equivalent but differently written SQL matches."""
        header = ["name", "age"]
        rows = [["Alice", "30"], ["Bob", "25"]]

        conn = create_table_from_wikisql(header, rows)

        gold_sql = "SELECT name FROM data WHERE age = '30'"
        # Different whitespace/formatting but same result
        generated_sql = "select NAME from data where AGE='30'"

        gold_result = execute_sql(conn, gold_sql)
        gen_result = execute_sql(conn, generated_sql)
        conn.close()

        assert results_match(gen_result, gold_result) is True

    def test_wrong_sql_fails(self):
        """Test that incorrect SQL doesn't match."""
        header = ["name", "age"]
        rows = [["Alice", "30"], ["Bob", "25"]]

        conn = create_table_from_wikisql(header, rows)

        gold_sql = "SELECT name FROM data WHERE age = '30'"
        generated_sql = "SELECT name FROM data WHERE age = '25'"

        gold_result = execute_sql(conn, gold_sql)
        gen_result = execute_sql(conn, generated_sql)
        conn.close()

        assert results_match(gen_result, gold_result) is False
